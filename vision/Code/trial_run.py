#!/usr/bin/env python3
"""
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import os
import open3d as o3d
import numpy as np
import copy

import do_pe
import settings
from helpers import add_noise, computeError

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

scene_id = settings.indexes[1]
noise_level = settings.noise_levels[-1]

def main():
    print(f"\n------------------------------------------------------------")
    print(f"POSE ESTIMATION - Scene {scene_id}, Noise {noise_level}")
    print(f"------------------------------------------------------------")

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print(f"\n[1/4] Loading data...")

    scene_file = settings.input_folder + f'point_cloud_{scene_id:04}.pcd'
    scene_pointcloud = o3d.io.read_point_cloud(scene_file)
    print(f"  Scene: {len(scene_pointcloud.points)} pts")

    scene_pointcloud_noisy = add_noise(scene_pointcloud, 0, noise_level)

    object_mesh = o3d.io.read_triangle_mesh(settings.input_folder + "duck.stl")
    object_pointcloud = object_mesh.sample_points_poisson_disk(10000)
    print(f"  Object: {len(object_pointcloud.points)} pts")

    o3d.visualization.draw_geometries([object_pointcloud, scene_pointcloud_noisy], window_name='Pre alignment')

    # =========================================================================
    # STEP 2: Pose estimation
    # =========================================================================
    print(f"\n[2/4] Running pose estimation...")

    estimated_pose = do_pe.do_pose_estimation(scene_pointcloud_noisy, object_pointcloud)

    # =========================================================================
    # STEP 3: Evaluate
    # =========================================================================
    print(f"\n[3/4] Evaluating...")

    ground_truth = np.loadtxt(settings.input_folder + f"gt_{scene_id:04}.txt")
    error_angle, error_pos = computeError(ground_truth, estimated_pose)

    ROT_THRESHOLD = 5.0
    POS_THRESHOLD = 5.0

    if error_angle <= ROT_THRESHOLD and error_pos <= POS_THRESHOLD:
        color = bcolors.OKGREEN
        result = "PASS"
    else:
        color = bcolors.FAIL
        result = "FAIL"

    print(f"\n{color}Result: {result}{bcolors.ENDC}")
    print(f"  Rotation: {error_angle} / {ROT_THRESHOLD} deg")
    print(f"  Position: {error_pos} / {POS_THRESHOLD} mm")

    # =========================================================================
    # STEP 4: Visualization
    # =========================================================================
    print(f"\n[4/4] Final visualization...")

    object_pointcloud.colors = o3d.utility.Vector3dVector(np.zeros_like(object_pointcloud.points) + [0, 255, 0])

    o3d.visualization.draw_geometries([copy.deepcopy(object_pointcloud).transform(estimated_pose), scene_pointcloud_noisy], window_name='Final alignment')

    o3d.visualization.draw_geometries([copy.deepcopy(object_pointcloud).transform(ground_truth), scene_pointcloud_noisy], window_name='Perfect alignment')

if __name__ == "__main__":
    main()
