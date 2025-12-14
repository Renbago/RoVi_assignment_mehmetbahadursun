"""
This file is for planners so everything can be added here
dynamicly change the planner type

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
import spatialmath as sm
from utils.mujoco_utils import is_q_valid, is_q_valid_with_held_object, Q_HOME
from scripts.ik_goal_region import IKGoalRegion

# Planner settings
PLANNER_TIMEOUT = 5.0
JOINT_BOUNDS_LOW = -np.pi
JOINT_BOUNDS_HIGH = np.pi
NUM_JOINTS = 6


class StateValidator:
    """
    OMPL state validity checker for collision detection.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        num_joint: Number of joints (6 for UR5)
        target_object: Object being approached (collisions ignored)
        held_object: Object being held during transport (for held object collision check)
        robot_rtb: roboticstoolbox robot (needed for FK when held_object is set)
    """
    def __init__(self, d, m, num_joint, target_object=None, held_object=None, robot_rtb=None):
        self.d = d
        self.m = m
        self.num_joint = num_joint
        self.target_object = target_object
        self.held_object = held_object
        self.robot_rtb = robot_rtb
        self.all_trajectories = []

    def __call__(self, state):
        q_pose = [state[i] for i in range(self.num_joint)]

        if self.held_object and self.robot_rtb:
            # Object is being held - check collision INCLUDING the held object
            return is_q_valid_with_held_object(
                d=self.d, m=self.m, q=q_pose,
                robot_rtb=self.robot_rtb,
                held_object=self.held_object,
                target_object=self.target_object
            )
        else:
            # Normal case - no object held
            return is_q_valid(d=self.d, m=self.m, q=q_pose, target_object=self.target_object) 

def generate_p2p_path(robot, start_q, target_frame, obj_name="box"):
    """
    Generate P2P path using pre-recorded waypoints from config.
    Returns list of joint configurations for the trajectory.
    """
    from utils import load_config, get_p2p_waypoints

    config = load_config()
    waypoints = get_p2p_waypoints(config, obj_name)

    if not waypoints:
        print(f"[P2P] No waypoints found for {obj_name}, using empty path")
        return []

    # Append home position at the end
    path = waypoints + [Q_HOME]

    print(f"[P2P] Generated {len(path)} waypoints for {obj_name}")
    return path

def plan(d, m, start_q, goal_q, planner_type="rrt", robot=None, target_frame=None):
    """
    Unified planner function.
    For now only supports but i want to add more:
    - 'rrt', 'prm', 'p2p':
    """
    
    # P2P - uses pre-recorded waypoints
    if planner_type.lower() == "p2p":
        return generate_p2p_path(robot, start_q, target_frame, obj_name="box")

    # RRT / PRM
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)
    
    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, NUM_JOINTS)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))

    # Increase collision checking resolution
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

    start = ob.State(space)
    goal = ob.State(space)

    for i in range(NUM_JOINTS):
        start[i] = start_q[i]
        goal[i] = goal_q[i]

    ss.setStartAndGoalStates(start, goal)

    if planner_type.lower() == "prm":
        planner = og.PRM(ss.getSpaceInformation())
    else:
        planner = og.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner)

    solved = ss.solve(PLANNER_TIMEOUT)

    if solved:
        # Simplify path (short-cutting to remove unnecessary waypoints)
        ss.simplifySolution()

        path = ss.getSolutionPath()
        print(f"[Planner] Path length: {path.length():.2f}, states: {path.getStateCount()}")

        solution_trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q_pose = [state[i] for i in range(space.getDimension())]
            solution_trajectory.append(np.array(q_pose))

        return solution_trajectory
    else:
        print(f"{planner_type} failed to find a path.")
        return None


def plan_simple_ik(d, m, start_q, target_frame, robot, planner_type="rrt", target_object=None, held_object=None, n_samples=16):
    """
    Smart IK approach:
    1. Try multiple IK solutions (J1 sweep like GoalSampleableRegion)
    2. Find all collision-free solutions
    3. Pick the CLOSEST one to current position (shortest path)
    4. Use basic RRT to reach that goal

    Pros: Robust (multiple IK) + Short paths (closest selection)
    Cons: Slightly slower than single IK

    Args:
        d: MuJoCo data
        m: MuJoCo model
        start_q: Starting joint configuration
        target_frame: SE3 Cartesian target pose
        robot: roboticstoolbox robot (for IK)
        planner_type: "rrt" or "prm"
        target_object: Name of target object (collisions ignored)
        held_object: Name of object being held (for collision check)
        n_samples: Number of IK attempts with different J1 seeds

    Returns:
        List of joint configurations (trajectory) or None
    """
    print(f"[SmartIK] Computing {n_samples} IK solutions...")
    if held_object:
        print(f"[SmartIK] Held object: {held_object}")
    print(f"[SmartIK] Target position: {target_frame.t}")

    valid_solutions = []
    ik_converged = 0
    collision_failed = 0

    # Sweep J1 (base joint) to find multiple IK solutions
    for i in range(n_samples):
        theta = -np.pi + (2 * np.pi * i / n_samples)
        q_init = np.zeros(6)
        q_init[0] = theta

        result = robot.ik_LM(Tep=target_frame, q0=q_init)

        if result[1]:  # IK converged
            ik_converged += 1
            goal_q = result[0]

            # Check if collision-free
            if held_object:
                is_valid = is_q_valid_with_held_object(d, m, goal_q, robot, held_object, target_object)
            else:
                is_valid = is_q_valid(d, m, goal_q, target_object)

            if is_valid:
                # Check uniqueness
                is_unique = True
                for existing in valid_solutions:
                    if np.linalg.norm(goal_q - existing) < 0.1:
                        is_unique = False
                        break

                if is_unique:
                    valid_solutions.append(goal_q)
            else:
                collision_failed += 1

    if not valid_solutions:
        print(f"[SmartIK] No valid IK solutions! (IK converged: {ik_converged}, collision failed: {collision_failed})")
        return None

    # Pick the CLOSEST solution to current position
    distances = [np.linalg.norm(sol - start_q) for sol in valid_solutions]
    best_idx = np.argmin(distances)
    goal_q = valid_solutions[best_idx]

    print(f"[SmartIK] Found {len(valid_solutions)} valid, picked closest (dist: {distances[best_idx]:.2f})")
    print(f"[SmartIK] Goal J1: {np.degrees(goal_q[0]):.1f}deg")

    # Use basic RRT to reach the goal
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, NUM_JOINTS, target_object, held_object, robot)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))

    # Increase collision checking resolution (default is 0.05 = 5%)
    # Lower value = more checks = safer paths but slower
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)  # 1%

    start = ob.State(space)
    goal = ob.State(space)
    for i in range(NUM_JOINTS):
        start[i] = start_q[i]
        goal[i] = goal_q[i]

    ss.setStartAndGoalStates(start, goal)

    if planner_type.lower() == "prm":
        planner_obj = og.PRM(ss.getSpaceInformation())
    else:
        planner_obj = og.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner_obj)
    solved = ss.solve(PLANNER_TIMEOUT)

    if solved:
        # Simplify path (short-cutting to remove unnecessary waypoints)
        ss.simplifySolution()

        path = ss.getSolutionPath()
        print(f"[SmartIK] Path length: {path.length():.2f}, states: {path.getStateCount()}")

        trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(NUM_JOINTS)])
            trajectory.append(q)
        return trajectory

    print("[SmartIK] RRT failed to find path!")
    return None


def plan_with_goal_region(d, m, start_q, target_frame, robot, planner_type="rrt", target_object=None, held_object=None):
    """
    GoalSampleableRegion approach:
    1. Compute multiple IK solutions (16 attempts)
    2. RRTConnect samples from valid goals
    3. Finds easiest-to-reach configuration

    Pros: Finds optimal IK solution automatically
    Cons: More complex, paths may be longer

    Args:
        d: MuJoCo data
        m: MuJoCo model
        start_q: Starting joint configuration
        target_frame: SE3 Cartesian target pose (NOT joint goal!)
        robot: roboticstoolbox robot (for IK)
        planner_type: "rrt" or "prm"
        target_object: Name of target object (collisions with it are ignored)
        held_object: Name of object being held during transport (for collision check)

    Returns:
        List of joint configurations (trajectory) or None
    """
    # Setup state space
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)

    # Setup space information with collision checker
    # Pass target_object so we only ignore collisions with THAT object
    # Pass held_object for transport collision checking
    si = ob.SpaceInformation(space)
    validator = StateValidator(
        d, m, NUM_JOINTS,
        target_object=target_object,
        held_object=held_object,
        robot_rtb=robot
    )
    si.setStateValidityChecker(ob.StateValidityCheckerFn(validator))

    # Increase collision checking resolution
    si.setStateValidityCheckingResolution(0.01)

    si.setup()

    # Create goal region with multiple IK solutions
    goal = IKGoalRegion(si, robot, target_frame, d, m, target_object=target_object, held_object=held_object)

    if not goal.hasValidSolution():
        print("[Planner] No valid IK solutions found for target!")
        return None

    # Setup problem definition
    pdef = ob.ProblemDefinition(si)

    # Set start state
    start = ob.State(space)
    for i in range(NUM_JOINTS):
        start[i] = start_q[i]
    pdef.addStartState(start)

    # Set goal REGION (not single state!)
    pdef.setGoal(goal)

    # Choose planner
    # RRTConnect works well with goal regions - bidirectional search
    if planner_type.lower() == "prm":
        planner = og.PRM(si)
    else:
        planner = og.RRTConnect(si)

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Solve
    solved = planner.solve(PLANNER_TIMEOUT)

    if solved:
        path = pdef.getSolutionPath()

        # Simplify path (short-cutting to remove unnecessary waypoints)
        simplifier = og.PathSimplifier(si)
        simplifier.simplify(path, 1.0)  # 1 second timeout

        print(f"[GoalRegion] Path length: {path.length():.2f}, states: {path.getStateCount()}")

        # Extract trajectory
        trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(NUM_JOINTS)])
            trajectory.append(q)

        # Log which goal was reached
        final_q = trajectory[-1]
        print(f"[GoalRegion] Final config J1: {np.degrees(final_q[0]):.1f}deg")

        return trajectory
    else:
        print(f"[GoalRegion] {planner_type} failed to find a path.")
        return None


def plan_to_frame(d, m, start_q, target_frame, robot, planner_type="rrt",
                  target_object=None, held_object=None, ik_mode="simple"):
    """
    Unified planning function - selects between IK modes.

    Args:
        ik_mode: "simple" (Exercise 9 style) or "goal_region" (GoalSampleableRegion)
        ... other args same as plan_simple_ik/plan_with_goal_region

    Returns:
        List of joint configurations (trajectory) or None
    """
    if ik_mode == "goal_region":
        return plan_with_goal_region(
            d, m, start_q, target_frame, robot, planner_type, target_object, held_object
        )
    else:
        # Default to simple IK
        return plan_simple_ik(
            d, m, start_q, target_frame, robot, planner_type, target_object, held_object
        )
