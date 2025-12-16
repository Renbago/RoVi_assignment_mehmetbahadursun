"""
Goal Region IK Planning Module - OMPL GoalSampleableRegion approach.

This module provides IK planning using OMPL's GoalSampleableRegion interface.
The approach:
1. Pre-compute all valid IK solutions for target frame
2. RRT samples from these goals during planning
3. Finds the easiest-to-reach configuration automatically

For direct IK approach, see ik_simple.py

Reference:
- OMPL GoalSampleableRegion: https://ompl.kavrakilab.org/classompl_1_1base_1_1GoalSampleableRegion.html
- OMPL IK Planning: https://ompl.kavrakilab.org/ConstrainedPlanningImplicitParallel.html
- roboticstoolbox IK: https://petercorke.github.io/robotics-toolbox-python/

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import random
from typing import Optional, List, Dict
from ompl import base as ob
from ompl import geometric as og
import spatialmath as sm

from utils.logger import ProjectLogger
from utils.mujoco_utils import (
    is_q_valid, is_q_valid_with_held_object, load_config
)
from scripts.planners import (
    normalize_angles, StateValidator,
    PLANNER_TIMEOUT, JOINT_BOUNDS_LOW, JOINT_BOUNDS_HIGH, NUM_JOINTS
)


class IKGoalRegion(ob.GoalSampleableRegion):
    """
    OMPL GoalSampleableRegion for multiple IK solutions.

    Pre-computes all valid IK solutions, then RRT samples from goals
    to find the easiest-to-reach configuration.

    Reference:
    - https://ompl.kavrakilab.org/classompl_1_1base_1_1GoalSampleableRegion.html

    IK Sampling Strategy:
    - UR5 has up to 8 unique IK solutions
    - Sweeping J1 explores different arm configurations
    - 16+ samples ensure we don't miss valid solutions
    """

    def __init__(self, si, robot, target_frame, d, m,
                 n_samples: int = 64, target_object: str = None,
                 held_object: str = None, sweep_enabled: bool = False,
                 sweep_degrees: float = 30,
                 logger: ProjectLogger = None):
        """
        Args:
            si: OMPL SpaceInformation
            robot: roboticstoolbox robot (for IK)
            target_frame: SE3 target pose
            d, m: MuJoCo data/model
            n_samples: Number of IK attempts
            target_object: Object to ignore in collision
            held_object: Object being held
            sweep_enabled: Enable approach direction sweep
            sweep_degrees: Sweep range in degrees
            logger: ProjectLogger instance
        """
        super(IKGoalRegion, self).__init__(si)
        self.setThreshold(0.01)

        self.robot = robot
        self.target_frame = target_frame
        self.d = d
        self.m = m
        self.target_object = target_object
        self.held_object = held_object
        self.sweep_enabled = sweep_enabled
        self.sweep_degrees = sweep_degrees
        self.logger = logger or ProjectLogger.get_instance()
        self.valid_solutions = []

        self._compute_ik_solutions(n_samples)

    def _compute_ik_solutions(self, n_samples: int):
        """
        Compute IK solutions with J1 and approach direction sweep.
        be care of using direction sweep, I've been added
        for the problem of IK sometimes always fails to find
        a solution. so Instead of deleting i just stayed as a parameter
        Referanced the exercise_3
        """
        self.logger.info("GoalRegion: Computing IK solutions")
        self.logger.debug(f"Target: [{self.target_frame.t[0]}, "
                         f"{self.target_frame.t[1]}, {self.target_frame.t[2]}]")

        if self.sweep_enabled:
            self.logger.debug(f"Approach sweep: +-{self.sweep_degrees} deg")

        ik_converged = 0
        collision_failed = 0

        sweep_range = np.radians(self.sweep_degrees) if self.sweep_enabled else 0

        for i in range(n_samples):
            # Approach direction sweep
            if self.sweep_enabled and n_samples > 1:
                approach_offset = -sweep_range + (2 * sweep_range * i / (n_samples - 1))
            else:
                approach_offset = 0

            rotated_target = sm.SE3.Rz(approach_offset) * self.target_frame
            rotated_target = sm.SE3.Rt(R=rotated_target.R, t=self.target_frame.t, check=False)

            # J1 sweep
            theta = -np.pi + (2 * np.pi * i / n_samples)
            q_init = np.zeros(6)
            q_init[0] = theta

            result = self.robot.ik_LM(Tep=rotated_target, q0=q_init)

            if result[1]:
                ik_converged += 1
                q_solution = result[0]

                # Collision check
                if self.held_object:
                    is_valid = is_q_valid_with_held_object(
                        self.d, self.m, q_solution,
                        robot_rtb=self.robot,
                        held_object=self.held_object,
                        target_object=self.target_object
                    )
                else:
                    is_valid = is_q_valid(self.d, self.m, q_solution,
                                          target_object=self.target_object)

                if is_valid:
                    is_unique = all(
                        np.linalg.norm(q_solution - existing) >= 0.1
                        for existing in self.valid_solutions
                    )
                    if is_unique:
                        self.valid_solutions.append(q_solution)
                else:
                    collision_failed += 1

        # Log results
        self.logger.log_ik_sampling(
            n_samples, ik_converged, collision_failed, len(self.valid_solutions)
        )

        if ik_converged == 0:
            self.logger.error("IK failed!!! Target might be out of reach")
        elif len(self.valid_solutions) == 0:
            self.logger.error("All IK solutions collided with obstacles")

    def distanceGoal(self, state):
        """
        Distance from state to nearest valid IK solution.
        """
        if not self.valid_solutions:
            return float('inf')

        q = np.array([state[i] for i in range(6)])
        return min(np.linalg.norm(q - sol) for sol in self.valid_solutions)

    def sampleGoal(self, state):
        """
        Sample a random valid IK solution.

        IMPORTANT: Must use index assignment (state[i] = val),
        do not use direct assignment (state = ...) due to OMPL Python bindings.
        """
        if self.valid_solutions:
            sol = random.choice(self.valid_solutions)
            for i in range(6):
                state[i] = sol[i]

    def maxSampleCount(self):
        """
        Number of unique goal samples available.
        """
        return len(self.valid_solutions) if self.valid_solutions else 0

    def canSample(self):
        """
        checking valid_positions
        """
        return len(self.valid_solutions) > 0

    def couldSample(self):
        """
        Is goal sampling possible ?
        """
        return True

    def hasValidSolution(self):
        """
        Check if at least one valid solution exists
        """
        return len(self.valid_solutions) > 0

    def getSolutionCount(self):
        """
        Return number of valid IK solutions.
        """
        return len(self.valid_solutions)

    def getBestSolution(self, start_q: np.ndarray) -> Optional[np.ndarray]:
        """
        Get solution closest to start configuration
        """
        if not self.valid_solutions:
            return None
        return min(self.valid_solutions, key=lambda s: np.linalg.norm(s - start_q))

    def getAllSolutions(self) -> List[np.ndarray]:
        """
        Return all valid solutions
        """
        return self.valid_solutions.copy()


def plan_with_goal_region(d, m, start_q: np.ndarray, target_frame,
                           robot, planner_type: str = "rrt",
                           target_object: str = None,
                           held_object: str = None) -> Optional[List[np.ndarray]]:
    """
    GoalSampleableRegion approach:
    1. Compute multiple IK solutions
    2. RRT samples from valid goals
    3. Finds easiest-to-reach configuration

    Args:
        d: MuJoCo data
        m: MuJoCo model
        start_q: Starting joint configuration
        target_frame: SE3 Cartesian target pose
        robot: roboticstoolbox robot (for IK)
        planner_type: "rrt" or "prm"
        target_object: Object being approached (collisions ignored)
        held_object: Object being held (for collision check)

    Returns:
        List of joint configurations (trajectory) or None

    Reference:
    - https://ompl.kavrakilab.org/classompl_1_1base_1_1GoalSampleableRegion.html
    """
    logger = ProjectLogger.get_instance()
    config = load_config()

    # Get IK settings from config
    ik_config = config.get('ik', {})
    sampling_config = ik_config.get('sampling', {})
    n_samples = sampling_config.get('n_samples', 64)

    sweep_config = sampling_config.get('sweep', {})
    sweep_enabled = sweep_config.get('enabled', False)
    sweep_degrees = sweep_config.get('degrees', 30)

    logger.log_planning_start("goal_region", start_q, target_frame,
                               target_object, held_object)

    start_q = normalize_angles(np.array(start_q))

    # Setup state space
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)

    # Setup space information
    si = ob.SpaceInformation(space)
    validator = StateValidator(
        d, m, NUM_JOINTS,
        target_object=target_object,
        held_object=held_object,
        robot_rtb=robot
    )
    si.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    si.setStateValidityCheckingResolution(0.005)
    si.setup()

    # Create goal region
    goal = IKGoalRegion(
        si, robot, target_frame, d, m,
        n_samples=n_samples,
        target_object=target_object,
        held_object=held_object,
        sweep_enabled=sweep_enabled,
        sweep_degrees=sweep_degrees,
        logger=logger
    )

    if not goal.hasValidSolution():
        logger.error("No valid IK solutions found for target frame!")
        logger.log_planning_result(False, planner_type="goal_region")
        return None

    logger.info(f"GoalRegion: {goal.getSolutionCount()} valid IK solutions")

    # Setup problem definition
    pdef = ob.ProblemDefinition(si)

    start = ob.State(space)
    for i in range(NUM_JOINTS):
        start[i] = start_q[i]
    pdef.addStartState(start)
    pdef.setGoal(goal)

    # Choose planner
    if planner_type.lower() == "prm":
        planner = og.PRM(si)
    else:
        planner = og.RRT(si)

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Solve
    solved = planner.solve(PLANNER_TIMEOUT)

    if solved and pdef.hasExactSolution():
        path = pdef.getSolutionPath()
        logger.log_planning_result(True, path.length(),
                                   path.getStateCount(), "goal_region")

        trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(NUM_JOINTS)])
            trajectory.append(q)

        logger.debug(f"Final J1: {np.degrees(trajectory[-1][0]):.1f} deg")
        return trajectory

    if solved:
        logger.warning("Only APPROXIMATE solution found - rejecting!")

    logger.error(f"{planner_type.upper()} failed within timeout!")
    logger.log_planning_result(False, planner_type="goal_region")
    return None
