"""
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import queue
import mujoco
import numpy as np
import time
import os
import mujoco.viewer
import glfw
import roboticstoolbox as rtb
import spatialmath as sm
import spatialmath.base as smb
from spatialmath import SE3
from spatialmath.base import trinterp, trnorm
import matplotlib.pyplot as plt
from typing import List
from PIL import Image

from robot import *
from cam import *
from scripts.helpers import (
    plan, 
    StateValidator, 
    plot_multi_joint_trajectory, 
    pick_object, 
    place_object
)


# params for sequence tracking
sequence_counter = 0
all_trajectories = []
movement_boundaries = [0]
sequence_names = []


def execute_movement(robot, cmd_queue, d, m, viewer):
    """
    execute a single movement sequence and return when completed
    """
    while len(cmd_queue) > 0 and viewer.is_running():
        cmd_element, cmd_queue = cmd_queue[0], cmd_queue[1:]
        desired_cmd, gripper_value = cmd_element

        if isinstance(desired_cmd, np.ndarray):
            # This allows the physics engine to calculate contacts/grasping correctly.
            # ur_set_qpos: kinematically sets position (teleportation) will NOT grab (physics ignored)
            # ur_ctrl_qpos: applies forces to reach position (control) WILL grab (physics enabled)
            ur_ctrl_qpos(data=d, q_desired=desired_cmd)

        if gripper_value is not None:
            hande_ctrl_qpos(data=d, gripper_value=gripper_value)

        mujoco.mj_step(m, d)
        viewer.sync()
    return cmd_queue


def run_sequence(robot, start_q, goal_q, d, m, viewer, planner_type="rrt"):
    """
    run a single planning + execution
    automatically increase sequence counter and logs as sequence_1, sequence_2, etc.
    """
    global sequence_counter, all_trajectories, movement_boundaries, sequence_names
    
    sequence_counter += 1
    seq_name = f"sequence_{sequence_counter}"
    sequence_names.append(seq_name)
    
    print(f"\n=== {seq_name} ===")
    
    path = plan(d=d, m=m, start_q=start_q, goal_q=goal_q, planner_type=planner_type)
    
    if path:
        robot.add_planned_path(path)
        robot.move_j_via(points=path, t=300)
        
        # Visualize dense path (small circles trace)
        robot.visualize_trajectory(viewer, use_queue=True, sample_rate=5)
        
        cmd_queue = robot.queue.copy()
        all_trajectories.extend([q.copy() for q, _ in robot.queue])
        robot.queue.clear()
        
        execute_movement(robot, cmd_queue, d, m, viewer)
        movement_boundaries.append(len(all_trajectories))
        print(f"{seq_name} complete!")
        return True
    else:
        print(f"{seq_name} failed - no path found")
        return False


def save_results(time_step):
    """
    save all necessary data to outputs/
    """
    global all_trajectories, movement_boundaries, sequence_names
    
    if all_trajectories:
        print("\n--- Saving outputs ---")
        np.save("outputs/trajectory.npy", {
            "trajectories": all_trajectories,
            "boundaries": movement_boundaries,
            "sequences": sequence_names
        }, allow_pickle=True)
        print(f"trajectory is saved: {len(all_trajectories)} points: {sequence_names}")
        print(f"boundaries: {movement_boundaries}")
        
        plot_multi_joint_trajectory(
            trajectory=all_trajectories,
            t_f=len(all_trajectories) * time_step,
            save_path="outputs/trajectory_plot.png",
            show=False
        )
        print("\ntrajectory plot is saved to outputs/trajectory_plot.png\n")


if __name__ == "__main__":
    # Initialize OpenGL context first
    # mj.GLContext(max_width=1280, max_height=720)  # Adjust size as needed

    # import os
    # os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "lesson_3"))

    model_path = "scene_obstacles_lecture_6.xml"  # Replace with your XML file
    time_step = 0.002 # Defined in scene.xml 
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    key_queue = queue.Queue()

    with mujoco.viewer.launch_passive(model=m, 
                                      data=d, 
                                      key_callback=lambda key: key_queue.put(key)
                                      ) as viewer:
        
        # Home position for the scene
        target_pos = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])  # UR home position
        ur_set_qpos(data=d, q_desired=target_pos)
        hande_ctrl_qpos(data=d, gripper_value=0) # Open gripper


        sim_start = time.time()
        while time.time() - sim_start < 3.0:
            mujoco.mj_step(m, d)
            viewer.sync()

        # the main execution part
        robot = UR5robot(data=d, model=m)
        planner_type = input("Select planner (rrt/prm/p2p): ").strip().lower() or "rrt"

        # just call run_sequence() for each movement you want
    
        # Pick up box
        pick_object(robot, "box", planner_type, d, m, viewer, execute_movement, all_trajectories, movement_boundaries)
        
        # Drop box
        place_object(robot, "box", planner_type, d, m, viewer, execute_movement, all_trajectories, movement_boundaries)

        # save results
        save_results(time_step)

        # keep viewer open
        print("\nSimulation complete. Close viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.01)