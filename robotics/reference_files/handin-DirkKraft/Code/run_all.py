#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import time

import do_pe
import helpers
import settings

indexes = settings.indexes
noise_levels = settings.noise_levels

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

def main():
    for index in indexes:
        for iteration in range(0,5):
            for noise_sigma in noise_levels:
                if index != indexes[0] and noise_sigma != 0.0:
                    continue

                scene_pointcloud_file_name = settings.input_folder + f'point_cloud_{index:04}.pcd'
                scene_pointcloud = o3d.io.read_point_cloud(scene_pointcloud_file_name)
                
                scene_pointcloud_noisy = helpers.add_noise(scene_pointcloud, 0, noise_sigma)
                
                object_mesh = o3d.io.read_triangle_mesh(settings.input_folder + "duck.stl")
                object_pointcloud = object_mesh.sample_points_poisson_disk(10000)

                # o3d.visualization.draw_geometries([object_pointcloud, scene_pointcloud_noisy], window_name='Pre alignment')

                st = time.time()
                estimated_pose = do_pe.do_pose_estimation(scene_pointcloud_noisy, object_pointcloud)
                et = time.time()

                elapsed_time = et - st

                np.savetxt(settings.results_folder + f'estimate_{index:04}_{noise_sigma}_{iteration}.txt',
                        estimated_pose)        

                # print("Final pose")
                # print (estimated_pose)

                ground_truth = np.loadtxt(settings.input_folder + f'gt_{index:04}.txt')

                # print("Ground truth")
                # print(ground_truth)

                error_angle, error_pos = helpers.computeError(ground_truth,estimated_pose)

                if error_angle > 5 or error_pos > 5:
                    color = bcolors.FAIL
                else:
                    color = bcolors.OKGREEN

                print(color, index, iteration, noise_sigma, error_angle, error_pos, elapsed_time)

                f = open(settings.results_folder + f'time_{index:04}_{noise_sigma}_{iteration}.txt', "w")
                f.write(str(elapsed_time) + '\n')
                f.close()


                # object_pointcloud.colors = o3d.utility.Vector3dVector(np.zeros_like(object_pointcloud.points) + [255,0,0])
                # o3d.visualization.draw_geometries([object_pointcloud.transform(estimated_pose), scene_pointcloud_noisy], window_name='Final alignment')

if __name__ == "__main__":
    main()
