#!/usr/bin/env python3

import helpers
import numpy as np
import matplotlib.pyplot as plt
from helpers import filter_errors
import settings


max_rotation_error = 5
max_position_error = 5

results_success = []
noises_all = []
noises_filtered_all = []
position_errors_all = []
position_errors_filtered_all = []
rotation_errors_all = []
rotation_errors_filtered_all = []
times = []

index = settings.indexes[0]
noise_sigmas = settings.noise_levels

for noise_sigma, true_noise_sigma in enumerate(noise_sigmas, 1):
    errors = []
    for iteration in range(0,5):

        estimated_pose = np.loadtxt(settings.results_folder + f'estimate_{index:04}_{true_noise_sigma}_{iteration}.txt')
        ground_truth = np.loadtxt(settings.input_folder + f'gt_{index:04}.txt')
        f = open(settings.results_folder + f'time_{index:04}_{true_noise_sigma}_{iteration}.txt')
        time = float(f.read())
        error_angle, error_pos = helpers.computeError(ground_truth,estimated_pose)

        #print (index, iteration, error_angle, error_pos)
        errors.append([error_angle, error_pos, noise_sigma])
        times.append(time)

        position_errors_all.append(error_pos)
        rotation_errors_all.append(error_angle)
        noises_all.append(noise_sigma)

    filtered_errors = filter_errors(errors, max_rotation_error, max_position_error)

    for err in filtered_errors:
        rotation_errors_filtered_all.append(err[0])
        position_errors_filtered_all.append(err[1])
        noises_filtered_all.append(err[2])

    succes_percentage = len(filtered_errors) * 100.0 / len(errors)
    #print(index, noise_sigma, succes_percentage)

    results_success.append(succes_percentage)

noise_level_labels = [str(i) for i in noise_sigmas]
noise_sigmas = np.arange(1, len(noise_sigmas)+1)

plt.scatter(noise_sigmas, results_success)
plt.ylim(0, 105)
plt.xlabel("Noise level")
plt.ylabel("Successful pose estimates [%]")
plt.title("Successful pose estimates per noise level\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '20-NoiseSuccessrate.png', bbox_inches="tight")

plt.figure() 
plt.scatter(noises_all, position_errors_all, marker='x')
plt.ylim(0)
plt.xlabel("Noise level")
plt.ylabel("Position error [mm]")
plt.title("Position errors per noise level (unfiltered)\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '21-NoisePositionErrorAll.png', bbox_inches="tight")

plt.figure() 
plt.scatter(noises_all, rotation_errors_all, marker='x')
plt.ylim(0)
plt.xlabel("Noise level")
plt.ylabel("Rotation error [deg]")
plt.title("Rotation errors per noise level (unfiltered)\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '23-NoiseRotationErrorAll.png', bbox_inches="tight")

plt.figure() 
plt.scatter(noises_filtered_all, position_errors_filtered_all, marker='x')
plt.ylim(0)
plt.xlabel("Noise level")
plt.ylabel("Position error [mm]")
plt.title("Position errors per noise level\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '22-NoisePositionErrorFiltered.png', bbox_inches="tight")

plt.figure() 
plt.scatter(noises_filtered_all, rotation_errors_filtered_all, marker='x')
plt.ylim(0)
plt.xlabel("Noise level")
plt.ylabel("Rotation error [deg]")
plt.title("Rotation errors per noise level\nwith $max\\ rotation\\ error=" + str(max_rotation_error) + "^\\circ$ and $max\\ position\\ error=" + str(max_position_error) + "\\,mm$\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '24-NoiseRotationErrorFiltered.png', bbox_inches="tight")

plt.figure()
plt.scatter(noises_all, times, marker='x')
plt.ylim(0)
plt.xlabel("Noise level")
plt.ylabel("Runtimes [s]")
plt.title("Runtime per noise level (unfiltered)\n$Scene=" + str(index) + "$")
plt.xticks(noise_sigmas, noise_level_labels)
# plt.xscale("log")
#plt.show()
plt.savefig(settings.graph_folder + '25-NoiseRuntime.png', bbox_inches="tight")
    
