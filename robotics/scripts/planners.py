"""
Pure C-space path planning (joint space to joint space).

This module handles RRT/PRM planning in configuration space.
For IK-based planning (Cartesian to joint space), see ik_simple.py and ik_goal_region.py

Architecture:
- planners.py: Configuration space planning (q -> q)
- ik_simple.py: IK + Planning with closest solution (frame -> q)
- ik_goal_region.py: IK + Planning with GoalSampleableRegion (frame -> q)

Reference:
- OMPL RRT: https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRT.html
- OMPL PRM: https://ompl.kavrakilab.org/classompl_1_1geometric_1_1PRM.html

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
from typing import Optional, List, Dict
from ompl import base as ob
from ompl import geometric as og

from utils.logger import ProjectLogger
from utils.mujoco_utils import is_q_valid, is_q_valid_with_held_object, Q_HOME, load_config, get_p2p_waypoints

# Planner constants
PLANNER_TIMEOUT = 15.0  # Increased for held object transport
JOINT_BOUNDS_LOW = -np.pi
JOINT_BOUNDS_HIGH = np.pi
NUM_JOINTS = 6


def normalize_angles(q: np.ndarray) -> np.ndarray:
    """
    Normalize joint angles to [-pi, pi].

    Args:
        q: Joint configuration

    Returns:
        Normalized configuration
    """
    return np.arctan2(np.sin(q), np.cos(q))

class StateValidator:
    """
    OMPL state validity checker for collision detection.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        num_joint: Number of joints (6 for UR5)
        target_object: Object being approached (collisions ignored)
        held_object: Object being held during transport
        robot_rtb: roboticstoolbox robot (for FK when held_object is set)
    """

    def __init__(self, d, m, num_joint: int, target_object: str = None,
                 held_object: str = None, robot_rtb=None):
        self.d = d
        self.m = m
        self.num_joint = num_joint
        self.target_object = target_object
        self.held_object = held_object
        self.robot_rtb = robot_rtb
        self.all_trajectories = []

    def __call__(self, state) -> bool:
        """
        Check if state is collision-free.
        """
        q_pose = [state[i] for i in range(self.num_joint)]

        if self.held_object and self.robot_rtb:
            return is_q_valid_with_held_object(
                d=self.d, m=self.m, q=q_pose,
                robot_rtb=self.robot_rtb,
                held_object=self.held_object,
                target_object=self.target_object
            )
        else:
            return is_q_valid(d=self.d, m=self.m, q=q_pose,
                             target_object=self.target_object)


class PathPlanner:
    """
    Configuration space path planner using OMPL RRT/PRM.

    Usage:
        planner = PathPlanner()
        path = planner.plan(d, m, start_q, goal_q)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PathPlanner.

        Args:
            config: Optional config dict
        """
        self.config = config or load_config()
        self.logger = ProjectLogger.get_instance()
        self.timeout = PLANNER_TIMEOUT

    def plan(self, d, m, start_q: np.ndarray, goal_q: np.ndarray,
             planner_type: str = "rrt", robot=None, target_frame=None,
             held_object: str = None,
             target_object: str = None) -> Optional[List[np.ndarray]]:
        """
        Configuration space planning (joint space to joint space).

        Supports:
        - 'rrt': Rapidly-exploring Random Tree
        - 'prm': Probabilistic Roadmap
        - 'p2p': Pre-recorded waypoints from config

        Args:
            d: MuJoCo data
            m: MuJoCo model
            start_q: Starting joint configuration
            goal_q: Goal joint configuration
            planner_type: "rrt", "prm", or "p2p"
            robot: for p2p
            target_frame: SE3 target frame (for p2p)
            held_object: Object being held (for collision check)
            target_object: Object to ignore in collision check

        Returns:
            List of joint configurations or None

        Reference:
        - https://ompl.kavrakilab.org/planners.html
        """
        # P2P uses pre-recorded waypoints
        if planner_type.lower() == "p2p":
            return self._generate_p2p_path(robot, start_q, target_frame, "box")

        self.logger.log_planning_start(planner_type, start_q, goal_q,
                                       target_object, held_object)

        start_q = normalize_angles(np.array(start_q))
        goal_q = normalize_angles(np.array(goal_q))

        # Get robot_rtb for FK (held object collision check)
        robot_rtb = robot.robot_ur5 if robot and hasattr(robot, 'robot_ur5') else None

        # Setup state space
        space = ob.RealVectorStateSpace(NUM_JOINTS)
        bounds = ob.RealVectorBounds(NUM_JOINTS)
        bounds.setLow(JOINT_BOUNDS_LOW)
        bounds.setHigh(JOINT_BOUNDS_HIGH)
        space.setBounds(bounds)

        ss = og.SimpleSetup(space)
        validator = StateValidator(d, m, NUM_JOINTS, target_object,
                                   held_object, robot_rtb)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
        ss.getSpaceInformation().setStateValidityCheckingResolution(0.005)

        start = ob.State(space)
        goal = ob.State(space)
        for i in range(NUM_JOINTS):
            start[i] = start_q[i]
            goal[i] = goal_q[i]

        ss.setStartAndGoalStates(start, goal)

        # Select planner
        if planner_type.lower() == "prm":
            planner_obj = og.PRM(ss.getSpaceInformation())
        else:
            planner_obj = og.RRT(ss.getSpaceInformation())

        ss.setPlanner(planner_obj)
        solved = ss.solve(self.timeout)

        if solved:
            path = ss.getSolutionPath()
            self.logger.log_planning_result(True, path.length(),
                                           path.getStateCount(), planner_type)

            trajectory = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                q = np.array([state[j] for j in range(NUM_JOINTS)])
                trajectory.append(q)

            return trajectory

        self.logger.error(f"{planner_type.upper()} failed within timeout!")
        self.logger.log_planning_result(False, planner_type=planner_type)
        return None

    def _generate_p2p_path(self, robot, start_q: np.ndarray,
                            target_frame, obj_name: str) -> List[np.ndarray]:
        """
        Generate P2P path using pre-recorded waypoints.

        Args:
            robot: Robot wrapper
            start_q: Starting configuration
            target_frame: Target frame (unused for p2p)
            obj_name: Object name to get waypoints for

        Returns:
            List of waypoints + home
        """
        self.logger.info(f"P2P Path Generation for: {obj_name}")

        waypoints = get_p2p_waypoints(self.config, obj_name)

        if not waypoints:
            self.logger.warning(f"No waypoints found for {obj_name}")
            return []

        path = waypoints + [Q_HOME]
        self.logger.info(f"P2P generated {len(path)} waypoints")

        return path


