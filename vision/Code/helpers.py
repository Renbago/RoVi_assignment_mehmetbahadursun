import numpy as np
import copy
from spatialmath import SE3, SO3
from spatialmath.base import trnorm
import open3d as o3d
import math

def add_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def numpyToSE3(transform_np):
    assert(transform_np.shape[0] == 4)
    assert(transform_np.shape[1] == 4)

    transform_se3 = SE3(trnorm(transform_np))

    return transform_se3

def computeError(ground_truth, estimate_pose):
    gt_se3 = numpyToSE3(ground_truth)
    ep_se3 = numpyToSE3(estimate_pose)

    # Rotation error in degrees
    error_angle = gt_se3.angdist(ep_se3) * 180.0 / math.pi
    # Position error in mm
    error_pos = np.linalg.norm(gt_se3.t - ep_se3.t, 2) * 1000

    return error_angle, float(error_pos)

def filter_errors(errors, max_rotation_error, max_position_error):
    result = []
    for e in errors:
        if e[0] > max_rotation_error or e[1] > max_position_error:
            continue
        else:
            result.append(e)
    return result
