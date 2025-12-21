#!/usr/bin/env python3

import helpers
import numpy as np
import matplotlib.pyplot as plt
from helpers import filter_errors
import settings


max_rotation_error = 5
max_position_error = 5

results_success = []
indexes_all = []
indexes_filtered_all = []
position_errors_all = []
position_errors_filtered_all = []
rotation_errors_all = []
rotation_errors_filtered_all = []
times = []

indexes = settings.indexes
noise_sigma = settings.noise_levels[0]

for index, true_index in enumerate(indexes,1):
    errors = []
    for iteration in range(0,5):
        
        estimated_pose = np.loadtxt(settings.results_folder + f'estimate_{true_index:04}_{noise_sigma}_{iteration}.txt')
        ground_truth = np.loadtxt(settings.input_folder + f'gt_{true_index:04}.txt')
        f = open(settings.results_folder + f'time_{true_index:04}_{noise_sigma}_{iteration}.txt')
        time = float(f.read())
        error_angle, error_pos = helpers.computeError(ground_truth,estimated_pose)

        #print (index, iteration, error_angle, error_pos)
        errors.append([error_angle, error_pos, index])
        times.append(time)

        position_errors_all.append(error_pos)
        rotation_errors_all.append(error_angle)
        indexes_all.append(index)

    filtered_errors = filter_errors(errors, max_rotation_error, max_position_error)

    for err in filtered_errors:
        rotation_errors_filtered_all.append(err[0])
        position_errors_filtered_all.append(err[1])
        indexes_filtered_all.append(err[2])

    succes_percentage = len(filtered_errors) * 100.0 / len(errors)
    #print(index, succes_percentage)

    results_success.append(succes_percentage)

index_labels = ["Scene " + str(i) for i in indexes]
indexes = np.arange(1, len(indexes)+1)

plt.scatter(indexes, results_success)
plt.ylim(0, 105)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Successful pose estimates [%]")
plt.title("Successful pose estimates per scene\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '10-ScenesSuccessrate.png', bbox_inches="tight")

plt.figure() 
plt.scatter(indexes_all, position_errors_all, marker='x')
plt.ylim(0)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Position error [mm]")
plt.title("Position errors per scene (unfiltered)\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '11-ScenesPositionErrorAll.png', bbox_inches="tight")

plt.figure() 
plt.scatter(indexes_all, rotation_errors_all, marker='x')
plt.ylim(0)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Rotation error [deg]")
plt.title("Rotation errors per scene (unfiltered)\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '13-ScenesRotationErrorAll.png', bbox_inches="tight")

plt.figure() 
plt.scatter(indexes_filtered_all, position_errors_filtered_all, marker='x')
plt.ylim(0)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Position error [mm]")
plt.title("Position errors per scene\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '12-ScenesPositionErrorFiltered.png', bbox_inches="tight")

plt.figure() 
plt.scatter(indexes_filtered_all, rotation_errors_filtered_all, marker='x')
plt.ylim(0)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Rotation error [deg]")
plt.title("Rotation errors per scene\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '14-ScenesRotationErrorFiltered.png', bbox_inches="tight")

plt.figure()
plt.scatter(indexes_all, times, marker='x')
plt.ylim(0)
plt.xticks(indexes, index_labels)
plt.xlabel("Scene index")
plt.ylabel("Runtimes [s]")
plt.title("Runtime per scene (unfiltered)\n$Noise=" + str(noise_sigma) + "$")
#plt.show()
plt.savefig(settings.graph_folder + '15-ScenesRuntime.png', bbox_inches="tight")
    
