"""
Simple IK Planning Module - Direct IK with sampling approach.

This module provides IK planning using multiple IK samples and RRT path planning.
The approach:
1. Sample multiple IK solutions (J1 sweep + optional approach direction sweep)
2. Filter for collision-free solutions
3. Pick the CLOSEST one to current configuration
4. Use RRT to reach that goal

For GoalSampleableRegion approach, see ik_goal_region.py

Reference:
- roboticstoolbox IK: https://petercorke.github.io/robotics-toolbox-python/
- UR5 IK Solutions: https://alexanderelias.com/ur5-ik/
- OMPL RRT: https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRT.html

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import spatialmath as sm
from typing import Optional, List, Dict
from ompl import base as ob
from ompl import geometric as og
from scripts.ik_goal_region import plan_with_goal_region

from utils.logger import ProjectLogger
from utils.mujoco_utils import (
    is_q_valid, is_q_valid_with_held_object, load_config
)
from scripts.planners import (
    normalize_angles, StateValidator,
    PLANNER_TIMEOUT, JOINT_BOUNDS_LOW, JOINT_BOUNDS_HIGH, NUM_JOINTS
)


def plan_simple_ik(d, m, start_q: np.ndarray, target_frame,
                   robot, planner_type: str = "rrt",
                   target_object: str = None, held_object: str = None,
                   n_samples: int = None, config: Dict = None) -> Optional[List[np.ndarray]]:
    """
    Smart IK approach:
    1. Try multiple IK solutions (J1 sweep)
    2. Find all collision-free solutions
    3. Pick the CLOSEST one to current position
    4. Use RRT to reach that goal

    Args:
        d: MuJoCo data
        m: MuJoCo model
        start_q: Starting joint configuration
        target_frame: SE3 Cartesian target pose
        robot: roboticstoolbox robot (for IK)
        planner_type: "rrt" or "prm"
        target_object: Name of target object (collisions ignored)
        held_object: Name of object being held
        n_samples: Override number of IK attempts
        config: Optional config dict

    Returns:
        List of joint configurations (trajectory) or None
    """
    logger = ProjectLogger.get_instance()

    if config is None:
        config = load_config()

    # Get IK settings from config
    ik_config = config.get('ik', {})
    sampling_config = ik_config.get('sampling', {})

    if n_samples is None:
        n_samples = sampling_config.get('n_samples', 64)

    sweep_config = sampling_config.get('sweep', {})
    sweep_enabled = sweep_config.get('enabled', False)
    sweep_degrees = sweep_config.get('degrees', 30)

    start_q = normalize_angles(np.array(start_q))

    logger.log_planning_start("simple_ik", start_q, target_frame,
                               target_object, held_object)
    logger.info(f"IK Samples: {n_samples}")

    if sweep_enabled:
        logger.debug(f"Approach sweep: {sweep_degrees} deg")
    else:
        logger.debug("Approach sweep: DISABLED")

    # Sample IK solutions
    valid_solutions = _sample_ik_solutions(
        start_q, target_frame, robot, d, m, n_samples,
        target_object, held_object, sweep_enabled, sweep_degrees, logger
    )

    if not valid_solutions:
        logger.error("No valid IK solutions found!")
        logger.log_planning_result(False, planner_type="simple_ik")
        return None

    # Pick closest solution
    goal_q = _select_closest_solution(valid_solutions, start_q, logger)
    logger.debug(f"Goal J1: {np.degrees(goal_q[0]):.1f} deg")

    # RRT to goal
    return _plan_rrt_to_goal(
        d, m, start_q, goal_q, robot, planner_type,
        target_object, held_object, "simple_ik", logger
    )


def plan_to_frame(d, m, start_q: np.ndarray, target_frame,
                  robot, planner_type: str = "rrt",
                  target_object: str = None, held_object: str = None,
                  ik_mode: str = None) -> Optional[List[np.ndarray]]:
    """
    Unified planning function - selects between IK modes.

    AUTOMATIC RETRY: If planning fails with sweep disabled, automatically
    retries with sweep enabled.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        start_q: Starting joint configuration
        target_frame: SE3 Cartesian target pose
        robot: roboticstoolbox robot (for IK)
        planner_type: "rrt" or "prm"
        target_object: Name of target object (collisions ignored)
        held_object: Name of object being held (for collision check)
        ik_mode: "simple" or "goal_region" (default from config)

    Returns:
        List of joint configurations (trajectory) or None

    Reference:
    - Automatic retry: https://ompl.kavrakilab.org/FAQ.html
    """
    logger = ProjectLogger.get_instance()
    config = load_config()

    ik_config = config.get('ik', {})
    if ik_mode is None:
        ik_mode = ik_config.get('mode', 'simple')

    logger.info(f"IK Mode: {ik_mode.upper()}, Planner: {planner_type.upper()}")

    if ik_mode == "goal_region":
        result = plan_with_goal_region(
            d, m, start_q, target_frame, robot, planner_type,
            target_object, held_object
        )
    else:
        result = plan_simple_ik(
            d, m, start_q, target_frame, robot, planner_type,
            target_object, held_object, config=config
        )

    # AUTOMATIC RETRY with sweep if failed
    sweep_config = ik_config.get('sampling', {}).get('sweep', {})
    sweep_enabled = sweep_config.get('enabled', False)
    sweep_degrees = sweep_config.get('degrees', 30)

    if result is None and not sweep_enabled:
        logger.warning("RETRY: Planning failed with sweep DISABLED")
        logger.warning(f"Retrying with approach sweep ENABLED (+-{sweep_degrees} deg)...")

        result = _plan_with_retry_sweep(
            d, m, start_q, target_frame, robot, planner_type,
            target_object, held_object, config, logger
        )

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _sample_ik_solutions(start_q: np.ndarray, target_frame,
                          robot, d, m, n_samples: int,
                          target_object: str, held_object: str,
                          sweep_enabled: bool, sweep_degrees: float,
                          logger) -> List[np.ndarray]:
    """
    Sample multiple IK solutions with J1 sweep and approach direction sweep.

    Args:
        start_q: Starting configuration (used as seed base)
        target_frame: SE3 target pose
        robot: roboticstoolbox robot for IK
        d, m: MuJoCo data/model
        n_samples: Number of attempts
        target_object: Object to ignore in collision
        held_object: Object being held
        sweep_enabled: Whether to sweep approach direction
        sweep_degrees: Sweep range in degrees
        logger: ProjectLogger instance

    Returns:
        List of valid, unique IK solutions
    """
    valid_solutions = []
    ik_converged = 0
    collision_failed = 0

    sweep_range = np.radians(sweep_degrees) if sweep_enabled else 0

    for i in range(n_samples):
        # Approach direction sweep
        if sweep_enabled and n_samples > 1:
            approach_offset = -sweep_range + (2 * sweep_range * i / (n_samples - 1))
        else:
            approach_offset = 0

        # Rotate target frame (orientation only, keep position)
        rotated_target = sm.SE3.Rz(approach_offset) * target_frame
        rotated_target = sm.SE3.Rt(R=rotated_target.R, t=target_frame.t, check=False)

        # Vary J1 seed
        q_init = start_q.copy()
        theta_offset = -np.pi + (2 * np.pi * i / n_samples)
        q_init[0] = start_q[0] + theta_offset

        result = robot.ik_LM(Tep=rotated_target, q0=q_init)

        if result[1]:  # IK converged
            ik_converged += 1
            goal_q = result[0]

            # Collision check
            if held_object:
                is_valid = is_q_valid_with_held_object(
                    d, m, goal_q, robot_rtb=robot,
                    held_object=held_object, target_object=target_object
                )
            else:
                is_valid = is_q_valid(d, m, goal_q, target_object)

            if is_valid:
                # Check uniqueness
                is_unique = all(
                    np.linalg.norm(goal_q - existing) >= 0.1
                    for existing in valid_solutions
                )
                if is_unique:
                    valid_solutions.append(goal_q)
            else:
                collision_failed += 1

    # Log results
    logger.log_ik_sampling(
        n_samples, ik_converged, collision_failed, len(valid_solutions)
    )

    return valid_solutions


def _select_closest_solution(solutions: List[np.ndarray],
                              start_q: np.ndarray, logger) -> np.ndarray:
    """
    Select the IK solution closest to current configuration.

    Args:
        solutions: List of valid IK solutions
        start_q: Current joint configuration
        logger: ProjectLogger instance

    Returns:
        Closest solution
    """
    distances = [np.linalg.norm(sol - start_q) for sol in solutions]
    best_idx = np.argmin(distances)

    logger.info(f"Selected closest solution (distance: {distances[best_idx]:.4f} rad)")

    return solutions[best_idx]


def _plan_rrt_to_goal(d, m, start_q: np.ndarray, goal_q: np.ndarray,
                      robot, planner_type: str,
                      target_object: str, held_object: str,
                      log_label: str, logger) -> Optional[List[np.ndarray]]:
    """
    Plan RRT path from start to goal configuration.

    Args:
        d, m: MuJoCo data/model
        start_q: Start configuration
        goal_q: Goal configuration
        robot: Robot for FK (collision check)
        planner_type: "rrt" or "prm"
        target_object: Object to ignore
        held_object: Object being held
        log_label: Label for logging
        logger: ProjectLogger instance

    Returns:
        Trajectory or None
    """
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, NUM_JOINTS, target_object, held_object, robot)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.005)

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
        # Reject approximate solutions
        if not ss.haveExactSolutionPath():
            logger.warning("Only APPROXIMATE solution - rejecting!")
            logger.log_planning_result(False, planner_type=log_label)
            return None

        path = ss.getSolutionPath()
        logger.log_planning_result(True, path.length(),
                                   path.getStateCount(), log_label)

        trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(NUM_JOINTS)])
            trajectory.append(q)

        logger.debug(f"Trajectory waypoints: {len(trajectory)}")
        return trajectory

    logger.error("RRT failed to find path!")
    logger.log_planning_result(False, planner_type=log_label)
    return None


def _plan_with_retry_sweep(d, m, start_q: np.ndarray, target_frame,
                            robot, planner_type: str,
                            target_object: str,
                            held_object: str,
                            config: Dict, logger) -> Optional[List[np.ndarray]]:
    """
    Retry planning with sweep force-enabled.

    Args:
        Same as plan_simple_ik + config + logger

    Returns:
        Trajectory or None
    """
    start_q = normalize_angles(np.array(start_q))

    ik_config = config.get('ik', {})
    sampling_config = ik_config.get('sampling', {})
    n_samples = sampling_config.get('n_samples', 64)
    sweep_degrees = sampling_config.get('sweep', {}).get('degrees', 30)

    logger.info(f"Retry with sweep: +-{sweep_degrees} deg, samples: {n_samples}")

    # Sample with sweep FORCE ENABLED
    valid_solutions = _sample_ik_solutions(
        start_q, target_frame, robot, d, m, n_samples,
        target_object, held_object, sweep_enabled=True,
        sweep_degrees=sweep_degrees, logger=logger
    )

    if not valid_solutions:
        logger.error("Retry FAILED: No valid IK solutions even with sweep!")
        return None

    goal_q = _select_closest_solution(valid_solutions, start_q, logger)

    # RRT with extended timeout
    retry_timeout = PLANNER_TIMEOUT * 2
    logger.info(f"Retry RRT with timeout: {retry_timeout}s")

    # Direct RRT call with extended timeout
    space = ob.RealVectorStateSpace(NUM_JOINTS)
    bounds = ob.RealVectorBounds(NUM_JOINTS)
    bounds.setLow(JOINT_BOUNDS_LOW)
    bounds.setHigh(JOINT_BOUNDS_HIGH)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, NUM_JOINTS, target_object, held_object, robot)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.005)

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
    solved = ss.solve(retry_timeout)

    if solved and ss.haveExactSolutionPath():
        path = ss.getSolutionPath()
        logger.log_planning_result(True, path.length(),
                                   path.getStateCount(), "retry_sweep")

        trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(NUM_JOINTS)])
            trajectory.append(q)

        logger.info("RETRY SUCCESSFUL with sweep enabled!")
        return trajectory

    logger.error("Retry FAILED: RRT couldn't find path!")
    return None
