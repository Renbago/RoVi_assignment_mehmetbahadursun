"""
Handles manipulation tasks basically pick and place.
It acts as a task orchestrator that:
1. Calculates necessary poses (Grasp, Approach, Drop).
2. Calls the Planner (RRT/PRM/P2P) to generate safe paths.
3. Executes the movement sequence (Approach -> Grasp -> Lift -> Move -> Release).

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import scripts.helpers.planners as planners
from robot import get_mjobj_frame

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
                                path_t=300, log_list=None, boundary_list=None):
    """
    Executes the common sequence: 
    1. Follow Path (Approach)
    2. Descend to Target
    3. Actuate Gripper
    4. Lift to Approach
    """
    if path:
        robot.add_planned_path(path)
        robot.visualize_trajectory(viewer, use_queue=False)
        robot.move_j_via(points=path, t=path_t)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        
        # Descend
        print(f"descending to {action_verb} the {obj_name}")
        q_current = robot.get_current_q()
        q_target = robot.robot_ur5.ik_LM(Tep=target_frame, q0=q_current)[0]
        robot.move_j(start_q=q_current, end_q=q_target, t=500)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        
        # Actuate Gripper
        print(f"{action_verb}ing the {obj_name}")
        robot.set_gripper(gripper_value, t=100)
        execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        
        # Lift
        print(f"lifting the {obj_name}")
        q_current = robot.get_current_q()
        q_lift = robot.robot_ur5.ik_LM(Tep=approach_frame, q0=q_current)[0]
        robot.move_j(start_q=q_current, end_q=q_lift, t=500)
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

    # Grasp is Object Frame rotated 180 (x-axis) so z-axis points down
    # but we can change it here i mean might be the other side would be better
    grasp_frame = obj_frame * sm.SE3.Rx(-np.pi)
    
    # 10cm above grasp seems all right but meh can be better
    approach_frame = grasp_frame * sm.SE3.Tz(-0.10)
    
    print(f"moving to approach position with this planner: {planner_type}")
    # Calculate IK for approach 
    goal_q_approach = robot.robot_ur5.ik_LM(Tep=approach_frame, q0=start_q)[0]
    
    # creating the plan
    path = planners.plan(d=d, m=m, start_q=start_q, goal_q=goal_q_approach, planner_type=planner_type, robot=robot, target_frame=approach_frame)
    
    return execute_manipulation_sequence(
        robot, d, m, viewer, execute_fn,
        path=path,
        target_frame=grasp_frame,
        approach_frame=approach_frame,
        gripper_value=255, # Close
        obj_name=obj_name,
        action_verb="grasp",
        path_t=300,
        log_list=log_list,
        boundary_list=boundary_list
    )

def place_object(robot, obj_name, planner_type, d, m, viewer, execute_fn, log_list=None, boundary_list=None):
    """
    Robust Place Sequence:
    1. Approach Drop (RRT)
    2. Descend
    3. Release
    4. Lift
    """
    print(f"\n--- Dropping {obj_name} ---")
    start_q = robot.get_current_q()
    
    # Get drop frame
    drop_frame_name = f"drop_point_{obj_name}"
    
    drop_target_frame = get_mjobj_frame(model=m, data=d, obj_name=drop_frame_name)

    drop_target_frame = drop_target_frame * sm.SE3.Rx(-np.pi) 
    drop_approach_frame = drop_target_frame * sm.SE3.Tz(-0.10)
    
    # Plan to drop approach
    print(f"Moving to drop approach with planner type: {planner_type}")
    goal_q_drop_approach = robot.robot_ur5.ik_LM(Tep=drop_approach_frame, q0=start_q)[0]
    
    path = planners.plan(d=d, m=m, start_q=start_q, goal_q=goal_q_drop_approach, planner_type=planner_type, robot=robot, target_frame=drop_approach_frame)
    
    return execute_manipulation_sequence(
        robot, d, m, viewer, execute_fn,
        path=path,
        target_frame=drop_target_frame,
        approach_frame=drop_approach_frame,
        gripper_value=0, # Open
        obj_name=obj_name,
        action_verb="release",
        path_t=600, # Slower for carry or else its dropping the object :D i couldnt find a solution
        log_list=log_list,
        boundary_list=boundary_list
    )
