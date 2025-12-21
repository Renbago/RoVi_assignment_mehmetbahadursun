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
from scripts import (
    StateValidator,
    plot_multi_joint_trajectory,
    plot_all_joints_derivatives,
    plot_combined_trajectory,
    pick_object,
    place_object,
    return_to_home,
    execute_p2p_sequence,
    cv_demo_program
)
from utils import (
    load_config, get_objects_to_move, get_model_path, get_default_planner,
    get_trajectory_logger
)


# params for sequence tracking
sequence_counter = 0
all_trajectories = []
movement_boundaries = [0]
sequence_names = []


def execute_movement(robot, cmd_queue, d, m, viewer):
    """
    Execute a single movement sequence and return when completed.

    Args:
        robot: UR5robot instance
        cmd_queue: Queue of (joint_config, gripper_value) commands
        d: MuJoCo data
        m: MuJoCo model
        viewer: MuJoCo viewer

    Returns:
        Remaining command queue (empty if all executed)
    """
    while len(cmd_queue) > 0 and viewer.is_running():
        cmd_element, cmd_queue = cmd_queue[0], cmd_queue[1:]
        desired_cmd, gripper_value = cmd_element

        if isinstance(desired_cmd, np.ndarray):
            # ur_ctrl_qpos: applies forces to reach position (control) - WILL grab (physics enabled)
            # ur_set_qpos: kinematically sets position (teleportation) - will NOT grab (physics ignored)
            ur_ctrl_qpos(data=d, q_desired=desired_cmd)

        if gripper_value is not None:
            hande_ctrl_qpos(data=d, gripper_value=gripper_value)

        mujoco.mj_step(m, d)
        viewer.sync()
    return cmd_queue


def save_object_results(obj_name, trajectory, planner_type, time_step):
    """
    Save results for a single object.
    Creates: outputs/{planner}_{obj}.png (Position, Velocity, Accel, Jerk in one plot)
    """
    if not trajectory:
        return

    prefix = f"{planner_type}_{obj_name}"
    print(f"\n--- Saving {obj_name} outputs ---")

    # Save trajectory data
    np.save(f"outputs/{prefix}_trajectory.npy", np.array(trajectory))
    print(f"  Saved: outputs/{prefix}_trajectory.npy ({len(trajectory)} points)")

    # Save single plot with Position, Velocity, Acceleration, Jerk
    plot_all_joints_derivatives(
        trajectory=trajectory,
        dt=time_step,
        title_prefix=f"{planner_type.upper()} - {obj_name}",
        save_path=f"outputs/{prefix}.png",
        show=False
    )


def save_combined_results(trajectories_dict, planner_type, time_step):
    """
    Save combined plot for all objects side by side.
    Creates: outputs/{planner}_combined.png
    """
    if not trajectories_dict:
        return

    print("\n--- Saving combined outputs ---")

    # Save combined plot (all objects side by side)
    plot_combined_trajectory(
        trajectories_dict=trajectories_dict,
        dt=time_step,
        planner_type=planner_type,
        save_path=f"outputs/{planner_type}_combined.png",
        show=False
    )


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    model_path = get_model_path(config)
    objects_to_move = get_objects_to_move(config)
    default_planner = get_default_planner(config)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    key_queue = queue.Queue()

    with mujoco.viewer.launch_passive(model=m, 
                                      data=d, 
                                      key_callback=lambda key: key_queue.put(key)
                                      ) as viewer:
        
        # Home position for the scene
        ur_set_qpos(data=d, q_desired=UR5robot.Q_HOME)
        hande_ctrl_qpos(data=d, gripper_value=0)  # Open gripper
        cv_demo_program(d, m)

        sim_start = time.time()
        while time.time() - sim_start < 3.0:
            mujoco.mj_step(m, d)
            viewer.sync()

        # the main execution part
        robot = UR5robot(data=d, model=m)
        planner_type = input(f"Select planner (rrt/prm/p2p) [{default_planner}]: ").strip().lower() or default_planner

        # Initialize trajectory data logger for JSON export
        traj_logger = get_trajectory_logger()
        traj_logger.start_session(planner_type, config)

        # Store trajectories per object for combined plot
        trajectories_per_object = {}

        # Pick and place ALL objects in sequence
        for obj_name in objects_to_move:
            print(f"\n{'='*50}")
            print(f"Processing: {obj_name}")
            print(f"{'='*50}")

            # Track trajectory for this object separately
            obj_trajectory = []
            obj_boundaries = [0]

            # P2P uses pre-recorded waypoints from config.yaml
            if planner_type.lower() == "p2p":
                p2p_success = execute_p2p_sequence(robot, obj_name, d, m, viewer, execute_movement, obj_trajectory, obj_boundaries)
                if not p2p_success:
                    print(f"[ERROR] P2P sequence failed for {obj_name}")
            else:
                # RRT/PRM path planning
                # Pick up object
                pick_success = pick_object(robot, obj_name, planner_type, d, m, viewer, execute_movement, obj_trajectory, obj_boundaries)

                # Place object (with retry on failure)
                place_success = False
                if pick_success:
                    place_success = place_object(robot, obj_name, planner_type, d, m, viewer, execute_movement, obj_trajectory, obj_boundaries)

                    # Retry once if place failed
                    if not place_success:
                        print(f"[RETRY] Place failed for {obj_name} - retrying...")
                        place_success = place_object(robot, obj_name, planner_type, d, m, viewer, execute_movement, obj_trajectory, obj_boundaries)

                        if not place_success:
                            print(f"[ERROR] Place retry failed for {obj_name} - stopping execution")
                            raise RuntimeError(f"Could not place {obj_name} after retry")
                else:
                    print(f"Skipping place for {obj_name} - pick failed")

            # Return to home position for next object
            return_to_home(robot, planner_type, d, m, viewer, execute_movement, obj_trajectory, obj_boundaries)

            # Save results for this object
            save_object_results(obj_name, obj_trajectory, planner_type, UR5robot.TIME_STEP)

            # Log to JSON for later visualization/analysis
            traj_logger.add_trajectory(
                obj_name=obj_name,
                trajectory=obj_trajectory,
                planner_type=planner_type,
                duration=len(obj_trajectory) * UR5robot.TIME_STEP,
                metadata={'boundaries': obj_boundaries}
            )

            # Store for combined plot
            trajectories_per_object[obj_name] = obj_trajectory.copy()

            # Also add to global tracking
            all_trajectories.extend(obj_trajectory)
            movement_boundaries.append(len(all_trajectories))

            # Clear visualization for next object (only clears path spheres, not actual objects)
            robot.clear_trajectories(viewer)

        # Save combined plot (all objects side by side)
        save_combined_results(trajectories_per_object, planner_type, UR5robot.TIME_STEP)

        # Save JSON session data for later analysis
        json_path = traj_logger.save()
        if json_path:
            print(f"\n--- JSON Data Saved ---")
            print(f"  {json_path}")

        # keep viewer open
        print("\nSimulation complete. Close viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.01)