"""
Handles manipulation tasks basically pick and place.

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import scripts.planners as planners
from scripts.planners import plan_to_frame
from utils.mujoco_utils import get_mjobj_frame, load_config, get_ik_mode

# Motion timing (steps)
DESCENT_TIME = 500
LIFT_TIME = 500
GRIPPER_TIME = 300
PATH_TIME = 300
PLACE_PATH_TIME = 600

# Gripper values
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 255

# Approach
APPROACH_HEIGHT = 0.10
SIDE_GRASP_MIN_HEIGHT = 0.10  # Minimum Z for side grasp (gripper can't reach ground level)

# Grasp configurations per object type
# side_grasp: True = approach horizontally (gripper local Z), False = approach vertically (world Z)
GRASP_CONFIG = {
    "box": {"rx": np.pi, "side_grasp": False},                                      # Top-down
    "cylinder": {"rx": np.pi / 2, "rz": -np.pi/2, "tz": 0.0, "side_grasp": True},  # Side grasp at center
    "t_block": {"rx": np.pi, "side_grasp": False},                                  # Top-down
}
DEFAULT_GRASP_CONFIG = {"rx": np.pi, "rz": 0, "tz": 0, "side_grasp": False}


def execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list=None, boundary_list=None):
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
                                path_t=300, log_list=None, boundary_list=None,
                                lift_frame=None):
    """
    Executes the common sequence (exercise 9 style - simple IK):
    1. Follow Path
    2. Descend to Target
    3. Actuate Gripper
    4. Lift to lift_frame (defaults to approach_frame)
    """
    if path:
        robot.add_planned_path(path)
        robot.visualize_trajectory(viewer, use_queue=False)
        robot.move_j_via(points=path, t=path_t)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

        # Descend (simple IK like exercise 9)
        print(f"descending to {action_verb} the {obj_name}")
        q_current = robot.get_current_q()
        q_target = robot.robot_ur5.ik_LM(Tep=target_frame, q0=q_current)[0]
        robot.move_j(start_q=q_current, end_q=q_target, t=DESCENT_TIME)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

        # Actuate Gripper
        print(f"{action_verb}ing the {obj_name}")
        robot.set_gripper(gripper_value, t=GRIPPER_TIME)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

        # Lift (use lift_frame if provided, otherwise approach_frame)
        actual_lift_frame = lift_frame if lift_frame is not None else approach_frame
        print(f"lifting the {obj_name}")
        q_current = robot.get_current_q()
        q_lift = robot.robot_ur5.ik_LM(Tep=actual_lift_frame, q0=q_current)[0]
        robot.move_j(start_q=q_current, end_q=q_lift, t=LIFT_TIME)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

        return True
    return False

def pick_object(robot, obj_name, planner_type, d, m, viewer, execute_fn, log_list=None, boundary_list=None):

    print(f"\n--- Picking the {obj_name} ---")
    start_q = robot.get_current_q()
    
    try:
        obj_frame = get_mjobj_frame(model=m, data=d, obj_name=obj_name)
    except Exception as e:
        print(f"Object {obj_name} not found: {e}")
        return False

    # Get grasp config for this object type
    config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)
    grasp_rx = config.get("rx", np.pi)
    grasp_rz = config.get("rz", 0)   # Approach direction
    grasp_tz = config.get("tz", 0)   # Vertical offset before rotation
    side_grasp = config.get("side_grasp", False)

    grasp_pose = obj_frame * sm.SE3.Tz(grasp_tz)
    grasp_frame = grasp_pose * sm.SE3.Rz(grasp_rz) * sm.SE3.Rx(-grasp_rx)

    # Approach position: side_grasp uses gripper's local Z (horizontal pullback)
    #                    top-down uses world Z (vertical lift)
    if side_grasp:
        approach_frame = grasp_frame * sm.SE3.Tz(-APPROACH_HEIGHT)  # Horizontal pullback
        # For lift after grasping: use VERTICAL lift (world Z) to avoid collision
        lift_pos = grasp_frame.t + np.array([0, 0, APPROACH_HEIGHT])
        lift_frame = sm.SE3(np.vstack([np.hstack([grasp_frame.R, lift_pos.reshape(3,1)]), [0,0,0,1]]))
    else:
        # For top-down: move up in world Z, keeping same orientation
        approach_pos = grasp_frame.t + np.array([0, 0, APPROACH_HEIGHT])
        approach_frame = sm.SE3(np.vstack([np.hstack([grasp_frame.R, approach_pos.reshape(3,1)]), [0,0,0,1]]))
        lift_frame = None  # Use approach_frame for lift

    # Debug: print grasp position
    print(f"  Object frame pos: {obj_frame.t}")
    print(f"  Grasp frame pos: {grasp_frame.t}")
    print(f"  Approach frame pos: {approach_frame.t}")

    # DEBUG for cylinder object i couldnt find the damn problem
    if obj_name == "cylinder":
        q_user = np.array([-2.89018, -2.01056, -2.13656, -2.07339, -1.50792, 0.06283])
        correct_frame = robot.robot_ur5.fkine(q_user)
        print(f"\n  === DEBUG: FK Comparison ===")
        print(f"  User keyframe FK pos: {correct_frame.t}")
        print(f"  User keyframe FK rot:\n{correct_frame.R}")
        print(f"  Our grasp frame pos: {grasp_frame.t}")
        print(f"  Our grasp frame rot:\n{grasp_frame.R}")
        print(f"  Position diff: {np.linalg.norm(correct_frame.t - grasp_frame.t):.4f}m")
        print(f"  ==============================\n")

    # Get IK mode from config
    config = load_config()
    ik_mode = get_ik_mode(config)
    print(f"moving to approach position with {planner_type} (ik_mode: {ik_mode})")

    path = plan_to_frame(
        d=d, m=m,
        start_q=start_q,
        target_frame=approach_frame,
        robot=robot.robot_ur5,
        planner_type=planner_type,
        target_object=obj_name,
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
        lift_frame=lift_frame  # Vertical lift for side_grasp
    )

def place_object(robot, obj_name, planner_type, d, m, viewer, execute_fn, log_list=None, boundary_list=None):
    """
    Robust Place Sequence:
    1. Approach Drop
    2. Descend
    3. Release
    4. Lift
    """
    print(f"\n--- Dropping {obj_name} ---")
    start_q = robot.get_current_q()

    # Get grasp config - MUST match pick_object orientation
    config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)
    grasp_rx = config.get("rx", np.pi)
    grasp_rz = config.get("rz", 0)
    grasp_tz = config.get("tz", 0)
    side_grasp = config.get("side_grasp", False)

    # Get drop frame
    drop_frame_name = f"drop_point_{obj_name}"
    drop_base_frame = get_mjobj_frame(model=m, data=d, obj_name=drop_frame_name)

    # For side_grasp objects (cylinder), place at minimum height - gripper can't reach ground level
    if side_grasp:
        target_z = max(drop_base_frame.t[2], SIDE_GRASP_MIN_HEIGHT)
        drop_base_adjusted = sm.SE3.Rt(R=drop_base_frame.R, t=np.array([drop_base_frame.t[0], drop_base_frame.t[1], target_z]))
        drop_pose = drop_base_adjusted * sm.SE3.Tz(grasp_tz)
    else:
        drop_pose = drop_base_frame * sm.SE3.Tz(grasp_tz)

    drop_target_frame = drop_pose * sm.SE3.Rz(grasp_rz) * sm.SE3.Rx(-grasp_rx)

    # For DROP: always approach from above (vertical) - easier to reach
    approach_pos = drop_target_frame.t + np.array([0, 0, APPROACH_HEIGHT])
    drop_approach_frame = sm.SE3.Rt(R=drop_target_frame.R, t=approach_pos)

    print(f"  Drop target pos: {drop_target_frame.t}")
    print(f"  Drop approach pos: {drop_approach_frame.t}")
    
    # Get IK mode from config
    yaml_config = load_config()
    ik_mode = get_ik_mode(yaml_config)
    print(f"Moving to drop approach with {planner_type} (ik_mode: {ik_mode})")

    # During transport: object is HELD, not being approached
    # held_object enables collision checking WITH the held object
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

    return execute_manipulation_sequence(
        robot, d, m, viewer, execute_fn,
        path=path,
        target_frame=drop_target_frame,
        approach_frame=drop_approach_frame,
        gripper_value=GRIPPER_OPEN,
        obj_name=obj_name,
        action_verb="release",
        path_t=PLACE_PATH_TIME,  # Slower for carry or else it might drop it
        log_list=log_list,
        boundary_list=boundary_list
    )

def return_to_home(robot, planner_type, d, m, viewer, execute_fn, log_list=None, boundary_list=None):
    """
    Return robot to home position after placing object.
    The purpose of this robot couldnt go the second object after replace this its always
    failing idk so i added this function after i can able to solve the IK collision problem
    # TODO: If planning goes well remove this function
    """
    # Clear visualization before returning home
    robot.clear_trajectories(viewer)

    print(f"\n--- Returning to home position ---")
    start_q = robot.get_current_q()
    goal_q = planners.Q_HOME

    path = planners.plan(d=d, m=m, start_q=start_q, goal_q=goal_q, planner_type=planner_type, robot=robot)

    if path:
        robot.add_planned_path(path)
        robot.visualize_trajectory(viewer, use_queue=False)
        robot.move_j_via(points=path, t=PATH_TIME)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        print("Returned to home position")
        return True
    else:
        print("Failed to return to home - no path found")
        return False
