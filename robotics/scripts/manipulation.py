"""
Handles manipulation tasks - pick and place operations.

Important:
GRASP_CONFIG's are imported from utils.mujoco_utils - check there!

Architecture:
- Main functions: pick_object, place_object, return_to_home
- Helper functions (prefix _): At the bottom of file

Reference:
- roboticstoolbox: https://petercorke.github.io/robotics-toolbox-python/
- spatialmath: https://bdaiinstitute.github.io/spatialmath-python/

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import spatialmath as sm
from typing import Optional, Dict, List, Any, Tuple

from scripts.planners import PathPlanner, Q_HOME
from scripts.ik_simple import plan_to_frame
from utils.logger import ProjectLogger
from utils.mujoco_utils import (
    get_mjobj_frame, load_config, get_ik_mode,
    GRASP_CONFIG, DEFAULT_GRASP_CONFIG,
    compute_real_grasp_offset, clear_grasp_offset
)


# Timing settings
DESCENT_TIME = 500
LIFT_TIME = 500
GRIPPER_TIME = 300
GRIPPER_RELEASE_TIME = 800
PATH_TIME = 300
PLACE_PATH_TIME = 600

# Gripper settings
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 255

# Approach settings
APPROACH_HEIGHT = 0.15
SIDE_GRASP_MIN_HEIGHT = 0.10  # Minimum Z for side grasp -bypassing the .xml-


def execute_movement_wrapper(robot, d, m, viewer, execute_fn,
                              log_list: List = None, boundary_list: List = None):
    """
    Helper to visualize, execute and log movement.
    """
    robot.visualize_trajectory(viewer, use_queue=True, sample_rate=5)
    execute_fn(robot, robot.queue.copy(), d, m, viewer)
    
    if log_list is not None:
        log_list.extend([q.copy() for q, _ in robot.queue])
        if boundary_list is not None:
            boundary_list.append(len(log_list))
    
    robot.queue.clear()

def execute_manipulation_sequence(robot, d, m, viewer, execute_fn,
                                   path, target_frame, approach_frame,
                                   gripper_value, obj_name, action_verb,
                                   path_t=300, gripper_t=None,
                                   log_list=None, boundary_list=None,
                                   lift_frame=None, skip_lift=False):
    """
    Executes the common sequence (exercise 9 style - simple IK):
    1. Follow Path
    2. Descend to Target
    3. Actuate Gripper
    4. Lift or Retreat

    Args:
        skip_lift: If True, skip vertical LIFT and do RETREAT instead.

        GRASP (skip_lift=False):
            Gripper closes -> LIFT (vertical, collision-checked)

        RELEASE (skip_lift=True):
            Gripper opens -> RETREAT (gripper local -Z, APPROACH_HEIGHT distance)
            -> return_to_home handles the rest

        the retreat purpose is robot is too close to placed object for RRT to find
        valid start state. Retreat creates clearance first.
    """
    logger = ProjectLogger.get_instance()

    if not path:
        return False

    # 1. Execute approach path
    robot.add_planned_path(path)
    robot.visualize_trajectory(viewer, use_queue=False)
    robot.move_j_via(points=path, t=path_t)
    execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    logger.debug("Frame Comparison:")
    logger.debug(f"Approach pos: [{approach_frame.t[0]}, {approach_frame.t[1]}, {approach_frame.t[2]}]")
    logger.debug(f"Target pos:   [{target_frame.t[0]}, {target_frame.t[1]}, {target_frame.t[2]}]")

    # 2. Descend to target
    if not _plan_and_execute_descent(robot, d, m, viewer, execute_fn,
                                      target_frame, obj_name,
                                      log_list, boundary_list):
        return False

    # 3. Actuate gripper
    close_gripper = (gripper_value == GRIPPER_CLOSE)
    _actuate_gripper(robot, d, m, viewer, execute_fn,
                     close=close_gripper, obj_name=obj_name,
                     log_list=log_list, boundary_list=boundary_list,
                     gripper_t=gripper_t)

    # 4. Lift or Retreat
    if skip_lift:
        _plan_and_execute_retreat(robot, d, m, viewer, execute_fn,
                                  obj_name, log_list, boundary_list)
    else:
        actual_lift_frame = lift_frame if lift_frame is not None else approach_frame
        _plan_and_execute_lift(robot, d, m, viewer, execute_fn,
                               actual_lift_frame, obj_name,
                               log_list, boundary_list)

    return True


def pick_object(robot, obj_name, planner_type, d, m, viewer, execute_fn,
                log_list=None, boundary_list=None):
    """
    Pick the object from its current location.

    Sequence: Approach -> Descend -> Grasp -> Lift
    """
    # Initialize parameters
    params = _init_grasp_params(obj_name)
    logger = params['logger']

    logger.info(f"Picking object: {obj_name}")
    logger.log_manipulation_phase("approach", obj_name, {"action": "PICK"})

    start_q = robot.get_current_q()

    # Get object frame
    try:
        obj_frame = get_mjobj_frame(model=m, data=d, obj_name=obj_name)
    except Exception as e:
        logger.error(f"Object {obj_name} not found: {e}")
        return False

    # Compute grasp and approach frames
    grasp_frame = _compute_grasp_frame(obj_frame, params)
    approach_frame, lift_frame = _compute_approach_frame(grasp_frame, params)

    logger.debug(f"Object frame pos: [{obj_frame.t[0]}, {obj_frame.t[1]}, {obj_frame.t[2]}]")
    logger.debug(f"Grasp frame pos:  [{grasp_frame.t[0]}, {grasp_frame.t[1]}, {grasp_frame.t[2]}]")
    logger.debug(f"Approach pos:     [{approach_frame.t[0]}, {approach_frame.t[1]}, {approach_frame.t[2]}]")

    # Get IK mode from config
    config = load_config()
    ik_mode = get_ik_mode(config)
    logger.info(f"Planning approach: {planner_type.upper()}, IK mode: {ik_mode}")

    """
    IMPORTANT:
    In approach path, we check collision with EVERYTHING including target object
    We dont check it only during DESCENT (which is when gripper touches object)

    target_object: None means: All objects are collision obstacles (CORRECT for approach)
    target_object: obj_name means: Object ignored (ONLY for descent/grasp)

    held_object computed by compute_real_grasp_offset after gripper closed in pick_object
    held_object: None means: No object held (CORRECT for approach)
    held_object: obj_name means: RRT checks obj collision using REAL offset (ONLY for transport)

    This ensures object doesn't hit table/walls/other objects during transport.
    """
    # Plan approach path (check ALL collisions)
    path = plan_to_frame(
        d=d, m=m,
        start_q=start_q,
        target_frame=approach_frame,
        robot=robot.robot_ur5,
        planner_type=planner_type,
        target_object=None,
        held_object=None,
        ik_mode=ik_mode
    )
    
    return execute_manipulation_sequence(
        robot, d, m, viewer, execute_fn,
        path=path,
        target_frame=grasp_frame,
        approach_frame=approach_frame,
        gripper_value=GRIPPER_CLOSE,
        obj_name=obj_name,
        action_verb="grasp",
        path_t=PATH_TIME,
        log_list=log_list,
        boundary_list=boundary_list,
        lift_frame=lift_frame
    )

def place_object(robot, obj_name, planner_type, d, m, viewer, execute_fn,
                 log_list=None, boundary_list=None):
    """
    Place the held object at drop location.

    Sequence: Transport -> Descend -> Release -> Retreat
    """
    params = _init_grasp_params(obj_name)
    logger = params['logger']

    logger.log_manipulation_phase("transport", obj_name, {"action": "PLACE"})

    start_q = robot.get_current_q()

    # Get drop frame
    drop_frame_name = f"drop_point_{obj_name}"
    drop_base_frame = get_mjobj_frame(model=m, data=d, obj_name=drop_frame_name)

    # Adjust for side_grasp minimum height
    if params['side_grasp']:
        target_z = max(drop_base_frame.t[2], SIDE_GRASP_MIN_HEIGHT)
        drop_base_adjusted = sm.SE3.Rt(
            R=drop_base_frame.R,
            t=np.array([drop_base_frame.t[0], drop_base_frame.t[1], target_z]),
            check=False
        )
        drop_pose = drop_base_adjusted * sm.SE3.Tz(params['grasp_tz'])
    else:
        drop_pose = drop_base_frame * sm.SE3.Tz(params['grasp_tz'])

    # Compute drop target frame
    drop_target_frame = (drop_pose *
                         sm.SE3.Rz(params['grasp_rz']) *
                         sm.SE3.Rx(-params['grasp_rx']) *
                         sm.SE3.Tz(params['grasp_td']))

    # Always approach from above for drop
    approach_pos = drop_target_frame.t + np.array([0, 0, APPROACH_HEIGHT])
    drop_approach_frame = sm.SE3.Rt(R=drop_target_frame.R, t=approach_pos, check=False)

    logger.debug(f"Drop target pos:   [{drop_target_frame.t[0]}, {drop_target_frame.t[1]}, {drop_target_frame.t[2]}]")
    logger.debug(f"Drop approach pos: [{drop_approach_frame.t[0]}, {drop_approach_frame.t[1]}, {drop_approach_frame.t[2]}]")

    # Get IK mode from config
    config = load_config()
    ik_mode = get_ik_mode(config)
    logger.info(f"Planning transport: {planner_type.upper()}, IK mode: {ik_mode}")

    # Plan transport path (held_object for collision check)
    path = plan_to_frame(
        d=d, m=m,
        start_q=start_q,
        target_frame=drop_approach_frame,
        robot=robot.robot_ur5,
        planner_type=planner_type,
        target_object=None,
        held_object=obj_name,
        ik_mode=ik_mode
    )

    # Execute with skip_lift=True (retreat instead of lift)
    return execute_manipulation_sequence(
        robot, d, m, viewer, execute_fn,
        path=path,
        target_frame=drop_target_frame,
        approach_frame=drop_approach_frame,
        gripper_value=GRIPPER_OPEN,
        obj_name=obj_name,
        action_verb="release",
        path_t=PLACE_PATH_TIME,
        gripper_t=GRIPPER_RELEASE_TIME,
        log_list=log_list,
        boundary_list=boundary_list,
        skip_lift=True
    )


def return_to_home(robot, planner_type, d, m, viewer, execute_fn,
                   log_list=None, boundary_list=None, held_object=None):
    """
    Return robot to home position after placing object.
    The purpose of this robot couldnt go the second object after replace this its always
    failing idk so i added this function after i can able to solve the IK collision problem
    # TODO: If planning goes well remove this function
    """
    logger = ProjectLogger.get_instance()

    robot.clear_trajectories(viewer)

    logger.info("RETURNING TO HOME POSITION")
    if held_object:
        logger.info(f"  (Still holding: {held_object})")

    start_q = robot.get_current_q()
    goal_q = Q_HOME

    planner = PathPlanner()
    path = planner.plan(
        d=d, m=m, start_q=start_q, goal_q=goal_q,
        planner_type=planner_type, robot=robot, held_object=held_object
    )

    if path:
        robot.add_planned_path(path)
        robot.visualize_trajectory(viewer, use_queue=False)
        robot.move_j_via(points=path, t=PATH_TIME)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        logger.info("Successfully returned to home position")
        return True

    logger.error("Failed to return to home - no collision-free path!")
    return False



def _init_grasp_params(obj_name: str) -> Dict[str, Any]:
    """
    Initialize common grasp parameters for pick/place operations.

    Args:
        obj_name: Name of the object (box, cylinder, t_block)

    Returns:
        Dict with keys: logger, grasp_config, approach_height,
                       side_grasp, grasp_rx, grasp_rz, grasp_tz, grasp_td
    """
    logger = ProjectLogger.get_instance()
    grasp_config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)

    return {
        'logger': logger,
        'grasp_config': grasp_config,
        'approach_height': APPROACH_HEIGHT,
        'side_grasp': grasp_config.get('side_grasp', False),
        'grasp_rx': grasp_config.get('rx', np.pi),
        'grasp_rz': grasp_config.get('rz', 0),
        'grasp_tz': grasp_config.get('tz', 0),
        'grasp_td': grasp_config.get('td', 0),
    }


def _compute_grasp_frame(obj_frame, params: Dict) -> sm.SE3:
    """
    Compute grasp frame from object frame and grasp parameters.

    Args:
        obj_frame: SE3 frame of the object
        params: Dictionary from _init_grasp_params()

    Returns:
        SE3 grasp frame
    """
    grasp_pose = obj_frame * sm.SE3.Tz(params['grasp_tz'])
    grasp_frame = (grasp_pose *
                   sm.SE3.Rz(params['grasp_rz']) *
                   sm.SE3.Rx(-params['grasp_rx']) *
                   sm.SE3.Tz(params['grasp_td']))
    return grasp_frame


def _compute_approach_frame(grasp_frame, params: Dict) -> Tuple[sm.SE3, Optional[sm.SE3]]:
    """
    Compute approach frame from grasp frame and parameters.

    Args:
        grasp_frame: SE3 grasp frame
        params: Dictionary from _init_grasp_params()

    Returns:
        Tuple of (approach_frame, lift_frame or None)
    """
    if params['side_grasp']:
        approach_frame = grasp_frame * sm.SE3.Tz(-params['approach_height'])
        lift_pos = grasp_frame.t + np.array([0, 0, params['approach_height']])
        lift_frame = sm.SE3.Rt(R=grasp_frame.R, t=lift_pos, check=False)
    else:
        approach_pos = grasp_frame.t + np.array([0, 0, params['approach_height']])
        approach_frame = sm.SE3.Rt(R=grasp_frame.R, t=approach_pos, check=False)
        lift_frame = None

    return approach_frame, lift_frame


def _execute_path_via(robot, path: List[np.ndarray], t: int,
                       d, m, viewer, execute_fn,
                       log_list: List = None, boundary_list: List = None):
    """
    Execute a path segment using move_j_via (NOT for loop).

    Args:
        robot: Robot instance
        path: List of joint configurations
        t: Total time for path execution
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        log_list: Optional trajectory log list
        boundary_list: Optional boundary list
    """
    if not path:
        return

    robot.move_j_via(points=path, t=t)
    execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)


def _actuate_gripper(robot, d, m, viewer, execute_fn,
                      close: bool, obj_name: str,
                      log_list: List = None, boundary_list: List = None,
                      gripper_t: int = None):
    """
    Close or open gripper with physics settling.

    Args:
        robot: Robot instance
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        close: True for grasp, False for release
        obj_name: Object name for offset computation
        log_list: Optional trajectory log
        boundary_list: Optional boundary list
        gripper_t: Gripper actuation time (default GRIPPER_TIME)
    """
    logger = ProjectLogger.get_instance()

    gripper_value = GRIPPER_CLOSE if close else GRIPPER_OPEN
    action = "grasp" if close else "release"
    actual_t = gripper_t if gripper_t is not None else GRIPPER_TIME

    logger.log_manipulation_phase(action, obj_name)
    robot.set_gripper(gripper_value, t=actual_t)
    execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    # After grasp we computing the real offset for accurate collision checking with obstacle
    # Because of we are giving that knowledge like preparing we created knew joint
    # and giving the IK of that object space for calculating the IK- RRT for collision-free
    # After release it we are clearing the offset
    if close:
        compute_real_grasp_offset(d, m, robot.robot_ur5, obj_name)
    else:
        clear_grasp_offset(obj_name)


def _plan_and_execute_descent(robot, d, m, viewer, execute_fn,
                               target_frame, obj_name: str,
                               log_list: List = None, boundary_list: List = None) -> bool:
    """
    Plan and execute descent to grasp position.

    Args:
        robot: Robot instance
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        target_frame: SE3 target (grasp) frame
        obj_name: Object name
        log_list, boundary_list: Optional logging

    Returns:
        True if successful, False otherwise
    """
    logger = ProjectLogger.get_instance()
    logger.log_manipulation_phase("descent", obj_name, {"target_pos": target_frame.t})

    q_current = robot.get_current_q()

    descent_path = plan_to_frame(
        d=d, m=m, start_q=q_current, target_frame=target_frame,
        robot=robot.robot_ur5, planner_type="rrt",
        target_object=obj_name, held_object=None
    )

    if descent_path:
        _execute_path_via(robot, descent_path, DESCENT_TIME,
                         d, m, viewer, execute_fn, log_list, boundary_list)
        logger.info(f"Descent completed with {len(descent_path)} waypoints")
        return True

    logger.error("Descent RRT failed - no collision-free path!")
    return False


def _plan_and_execute_lift(robot, d, m, viewer, execute_fn,
                            lift_frame, obj_name: str,
                            log_list: List = None, boundary_list: List = None) -> bool:
    """
    Plan and execute lift after grasping.

    Args:
        robot: Robot instance
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        lift_frame: SE3 lift target frame
        obj_name: Object name (held object)
        log_list, boundary_list: Optional logging

    Returns:
        True if successful
    """
    logger = ProjectLogger.get_instance()
    logger.log_manipulation_phase("lift", obj_name, {"target_pos": lift_frame.t})

    q_current = robot.get_current_q()

    # Use IK + RRT for collision-free lift path
    # held_object=obj_name: RRT checks obj collision using REAL offset
    # can be accesable from config.yaml name is: objects_to_move
    # computed by compute_real_grasp_offset after gripper closed
    lift_path = plan_to_frame(
        d=d, m=m, start_q=q_current, target_frame=lift_frame,
        robot=robot.robot_ur5, planner_type="rrt",
        target_object=None, held_object=obj_name
    )

    if lift_path:
        _execute_path_via(robot, lift_path, LIFT_TIME,
                         d, m, viewer, execute_fn, log_list, boundary_list)
        logger.info(f"Lift completed with {len(lift_path)} waypoints")
        return True

    # Fallback to simple interpolation
    logger.warning("Lift RRT failed - using simple interpolation")
    ik_result = robot.robot_ur5.ik_LM(Tep=lift_frame, q0=q_current)
    q_lift = ik_result[0]
    robot.move_j(start_q=q_current, end_q=q_lift, t=LIFT_TIME)
    execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
    return True


def _plan_and_execute_retreat(robot, d, m, viewer, execute_fn,
                               obj_name: str,
                               log_list: List = None, boundary_list: List = None) -> bool:
    """
    Plan and execute retreat after placing (moves back from placed object).

    After release, retreat along gripper's approach direction to avoid collision
    because robot is still close to placed object so whenever RRT tries to handle
    its collide so cannot created path. For the solution using that.

    For Side grasp we need to retreat horizontally which is gripper local -Z direction
    For Top grasp we need to retreat vertically which is world +Z direction

    Args:
        robot: Robot instance
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        obj_name: Object name (just placed)
        log_list, boundary_list: Optional logging

    Returns:
        True if successful
    """
    logger = ProjectLogger.get_instance()
    logger.log_manipulation_phase("retreat", obj_name)

    q_current = robot.get_current_q()
    current_frame = robot.robot_ur5.fkine(q_current)
    retreat_frame = current_frame * sm.SE3.Tz(-APPROACH_HEIGHT)

    logger.debug(f"Current pos: [{current_frame.t[0]:.4f}, {current_frame.t[1]:.4f}, {current_frame.t[2]:.4f}]")
    logger.debug(f"Retreat pos: [{retreat_frame.t[0]:.4f}, {retreat_frame.t[1]:.4f}, {retreat_frame.t[2]:.4f}]")

    retreat_path = plan_to_frame(
        d=d, m=m, start_q=q_current, target_frame=retreat_frame,
        robot=robot.robot_ur5, planner_type="rrt",
        target_object=obj_name, held_object=None
    )

    if retreat_path:
        _execute_path_via(robot, retreat_path, DESCENT_TIME,
                         d, m, viewer, execute_fn, log_list, boundary_list)
        logger.info("Retreat completed - ready for RRT to home")
        return True

    logger.warning("Retreat path failed - home RRT might fail!")
    return False
