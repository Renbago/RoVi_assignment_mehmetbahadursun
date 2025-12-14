"""
path trajectory visualization for robot.
https://pypi.org/project/mjc-viewer/
https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/viewer.py

Functions to visualize the 3D path of the end-effector.
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import numpy as np
import mujoco as mj
from mjc_viewer import Serializer, Trajectory


def draw_path_in_viewer(viewer, positions, color=(0, 1, 0, 1), size=0.01):
    """
    draw spheres at each position in the mujoco viewer.
    uses mjvScene to add temporary geometries.
    """
    scene = viewer._scene
    
    for pos in positions:

        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32)
        )
        scene.ngeom += 1

def draw_line_path(viewer, positions, color=(1, 0.5, 0, 1), width=0.005):
    """
    Draw lines connecting waypoints.
    """
    scene = viewer._scene
    
    for i in range(len(positions) - 1):
        
        p1 = np.array(positions[i])
        p2 = np.array(positions[i + 1])
        
        center = (p1 + p2) / 2
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            continue
            
        # Create rotation matrix to align capsule with direction
        z_axis = direction / length
        # Find perpendicular axes
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross([0, 0, 1], z_axis)
        else:
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rot_mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
        
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            size=[width, length/2, 0],
            pos=center,
            mat=rot_mat,
            rgba=np.array(color, dtype=np.float32)
        )
        scene.ngeom += 1

def visualize_planned_path(viewer, robot, trajectory, show_spheres=True, show_lines=True):
    """
    Main function to visualize a planned trajectory.
    """
    if len(trajectory) == 0:
        return
        
    # Get end-effector positions using robot's method (subscribed logic)
    ee_positions = robot.get_ee_positions(trajectory=trajectory)
    
    if show_lines and len(ee_positions) > 1:
        draw_line_path(viewer, ee_positions, color=(1, 0.5, 0, 0.8))
    
    if show_spheres:
        # Draw start point (green)
        draw_path_in_viewer(viewer, [ee_positions[0]], color=(0, 1, 0, 1), size=0.015)
        
        # Draw intermediate points (yellow)
        if len(ee_positions) > 2:
            draw_path_in_viewer(viewer, ee_positions[1:-1], color=(1, 1, 0, 0.7), size=0.008)
        
        # Draw end point (red)
        draw_path_in_viewer(viewer, [ee_positions[-1]], color=(1, 0, 0, 1), size=0.015)
    
    print(f"Visualized path with {len(ee_positions)} waypoints")
