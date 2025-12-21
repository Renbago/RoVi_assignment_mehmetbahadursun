"""
Vision Pipeline Helpers.

Contains shared functions for pose estimation pipeline:
- Error computation
- Noise addition
- Point cloud visualization

Reference:
- https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
- https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import copy
import math
from spatialmath import SE3
from spatialmath.base import trnorm
import open3d as o3d


def add_noise(pcd, mu, sigma):
    """Add Gaussian noise to point cloud."""
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


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def show_point_clouds(geometries, title, info=None, debug=True):

    if not debug:
        return

    print(f"\n[STEP] {title}")
    if info:
        for line in info:
            print(f"  {line}")
    o3d.visualization.draw_geometries(geometries, window_name=title)


def color_by_normal_direction(pcd):
    """
    Color points by normal direction relative to center.

    Inward: Red
    Outward: Blue
    """
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    centroid = np.mean(pts, axis=0)
    to_point = pts - centroid
    dot = np.sum(to_point * normals, axis=1)

    colors = np.zeros((len(pts), 3))
    colors[dot < 0] = [1, 0, 0]
    colors[dot >= 0] = [0, 0, 1]

    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    n_inward = int(np.sum(dot < 0))
    n_outward = int(np.sum(dot >= 0))

    return pcd_colored, n_inward, n_outward


def dbscan_with_colors(pcd, eps=0.03, min_points=10):
    """
    DBSCAN clustering with random colors per cluster.
    """
    pcd_copy = copy.deepcopy(pcd)

    labels = np.array(pcd_copy.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=True
    ))

    max_label = labels.max()
    n_clusters = max_label + 1
    n_noise = int(np.sum(labels == -1))

    np.random.seed(42)
    cluster_colors = np.random.random((max_label + 1, 3))

    colors = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        if labels[i] >= 0:
            colors[i] = cluster_colors[labels[i]]
        else:
            colors[i] = [0, 0, 0]

    pcd_copy.colors = o3d.utility.Vector3dVector(colors)

    cluster_sizes = []
    for c in range(n_clusters):
        cluster_sizes.append(int(np.sum(labels == c)))

    stats = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes
    }

    return pcd_copy, labels, stats


def create_correspondence_lines(obj_pcd, scn_pcd, obj_fpfh, scn_fpfh, n_show=-1):
    """
    Create cyan lines showing FPFH feature correspondences.
    
    Args:
        n_show: Number of correspondences to show (-1 or None = all)
    """
    obj_f = np.asarray(obj_fpfh.data).T
    scn_f = np.asarray(scn_fpfh.data).T

    corr_list = []
    for j in range(obj_f.shape[0]):
        dist = np.sum((obj_f[j] - scn_f)**2, axis=-1)
        kmin = np.argmin(dist)
        corr_list.append([j, kmin, dist[kmin]])

    corr_list.sort(key=lambda x: x[2])

    obj_pts = np.asarray(obj_pcd.points)
    scn_pts = np.asarray(scn_pcd.points)

    obj_vis = copy.deepcopy(obj_pcd)
    scn_vis = copy.deepcopy(scn_pcd)

    if n_show is None or n_show < 0:
        n_show = len(corr_list)
    else:
        n_show = min(n_show, len(corr_list))

    lines = []
    colors = []

    for i in range(n_show):
        obj_idx, scn_idx, _ = corr_list[i]
        lines.append([obj_idx, len(obj_pts) + scn_idx])
        colors.append([0, 1, 1])

    combined_pts = np.vstack([obj_pts, scn_pts])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(combined_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    corr_dists = [c[2] for c in corr_list]
    corr_stats = {
        "total": len(corr_list),
        "shown": n_show,
        "dist_min": float(min(corr_dists)),
        "dist_max": float(max(corr_dists)),
        "dist_mean": float(np.mean(corr_dists)),
        "dist_median": float(np.median(corr_dists))
    }

    return line_set, obj_vis, scn_vis, corr_stats
