"""
Implementation based on lecture code's:
- ex2_pose_est_global_solution.py (RANSAC + FPFH)
- ex1_pose_est_local_solution.py (ICP)

Which I added:
- Instead of classic ransac: CorrespondenceCheckerBasedOnEdgeLength(0.9) to catch 180° flips
- MST-based normal orientation for consistent FPFH
- MST-based normal orientation per-cluster (DBSCAN) for better local consistency

Note from author:
- I've tried multiple normal orientation methods.
- For fixing the problem in max noise +10 hours researching and parameter tuning but
- It does always because of the noise is random sometimes all right sometimes not.
- So I quit to find a perfect solution for that specific case, and i let it go
- With current version if it does not match it doesnt move the duck so i believe its better
- In the past scenerios which I tuned it was finding sometimes and not but nearly always had a wrong placement
- Because of FPFH normal inconsistency.

Reference's are mentioned in the function's but main references are:
- https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
- https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import open3d as o3d
import copy
import numpy as np

from helpers import (
    show_point_clouds, dbscan_with_colors,
    color_by_normal_direction, create_correspondence_lines
)


DEBUG = True

"""
- Formulas parameters are referenced from open3d tutorial:
- https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
"""

# Voxel downsampling
VOXEL_SIZE = 0.005  # 5mm

# Normal estimation (radius = voxel × 2.0)
NORMAL_RADIUS_FACTOR = 2.0
NORMAL_MAX_NN = 30  # Open3D default

# MST normal orientation
NORMAL_K = 30  # More neighbors = better graph connectivity 
MST_LAMBDA_PENALTY = 10.0  # Tangent plane distance penalty
MST_COS_ALPHA_TOL = 0.5    # Angular threshold

# FPFH features (radius = voxel × 5.0)
FPFH_RADIUS_FACTOR = 5.0
FPFH_MAX_NN = 100

# RANSAC global registration (distance = voxel × 1.5)
RANSAC_DISTANCE_FACTOR = 1.5
RANSAC_EDGE_LENGTH = 0.9  # Catches flips
# RANSAC_NORMAL_ANGLE = np.pi / 5
RANSAC_MAX_ITERATION = 100000  # Open3D default
RANSAC_CONFIDENCE = 0.999  # Open3D default

# ICP local refinement
# For noisy data: threshold should be >= noise level
# noise=0.005 needs to be need threshold 5-10mm = 1.0-2.0
# But open3d uses 0.4 * voxel for clean data
ICP_DISTANCE_FACTOR = 1.2
ICP_MAX_ITERATION = 1000000

# DBSCAN clustering
DBSCAN_EPS = 0.02
DBSCAN_MIN_POINTS = 10

# Statistical outlier removal 
# - Used for removing noised table values from the duck for better calculation of normals and FPFH features
# Reference: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
OUTLIER_NB_NEIGHBORS = 30  # Balance between local and global
OUTLIER_STD_RATIO = 0.5  # Less aggressive (was 0.5)


# =============================================================================
# PREPROCESSING
# =============================================================================

def segment_table(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Remove table plane from scene using RANSAC plane segmentation.

    Reference:
    - http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    return objects_pcd


def estimate_normals(obj, scn):
    """
    Estimate normals for point clouds (without orientation fix).

    IMPORTANT:
    The covariance analysis algorithm produces two opposite directions as normal candidates.
    Without knowing the global structure of the geometry, both can be correct.
    This is known as the normal orientation problem.

    Reference:
    - https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    """
    radius = VOXEL_SIZE * NORMAL_RADIUS_FACTOR

    obj.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=NORMAL_MAX_NN))
    scn.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=NORMAL_MAX_NN))


def orient_normals_mst(obj, scn, labels=None):
    """
    Option 1: MST-based propagation.

    If labels provided: 
        MST will be calculate per-cluster
    Else:
        MST will be referenced all scene pcd

    MST ensures local consistency but the problem is:
    Direction is arbitrary (as research its seed dependent)
    I've tried lots of other ways but in the end same thing.

    Reference:
    - https://cims.nyu.edu/gcl/papers/2021-Dipole.pdf
    - https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    - Hoppe et al. 1992 "Surface Reconstruction from Unorganized Points"
      https://hhoppe.com/recon.pdf (Section 3.3: Consistent Orientation)
    - Open3D orient_normals_consistent_tangent_plane:
      https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Normal-estimation
    - Open3D PR #6198 (lambda_penalty, cos_alpha_tol for complex geometry):
      https://github.com/isl-org/Open3D/pull/6198

    """
    # Object: always single object
    obj.orient_normals_consistent_tangent_plane(k=NORMAL_K, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)

    if labels is None:
        # Global MST on whole scene
        scn.orient_normals_consistent_tangent_plane(k=NORMAL_K, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)
        if DEBUG:
            print(f"[Orient] MST global: obj={len(obj.points)}, scn={len(scn.points)}")
    else:
        # Per-cluster MST
        scn_points = np.asarray(scn.points)
        scn_normals = np.asarray(scn.normals)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        if DEBUG:
            print(f"[Orient] MST per-cluster: {len(unique_labels)} clusters")

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            # passig small clusters no necessary
            if cluster_size < 100:
                continue

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(scn_points[cluster_mask])
            cluster_pcd.normals = o3d.utility.Vector3dVector(scn_normals[cluster_mask])

            k_for_cluster = min(NORMAL_K, cluster_size - 1)
            cluster_pcd.orient_normals_consistent_tangent_plane(
                k=k_for_cluster,
                lambda_penalty=MST_LAMBDA_PENALTY,
                cos_alpha_tol=MST_COS_ALPHA_TOL
            )

            scn_normals[cluster_mask] = np.asarray(cluster_pcd.normals)

        scn.normals = o3d.utility.Vector3dVector(scn_normals)

def orient_normals_camera_consensus(obj, scn, labels):
    """
    Option 3: Camera Consensus (Facing Origin).

    Assuming camera is at (0,0,0) in scene space (or at least looking from -Z to +Z).

    1. MST for local smoothness.
    2. Orient towards [0,0,0].
    """
    camera = [0.0, 0.0, 0.0]

    # Object
    obj.orient_normals_consistent_tangent_plane(k=NORMAL_K, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)
    obj.orient_normals_towards_camera_location(camera_location=camera)

    # Scene Clusters
    scn_points = np.asarray(scn.points)
    scn_normals = np.asarray(scn.normals)
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        if cluster_size < 100: continue

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(scn_points[cluster_mask])
        cluster_pcd.normals = o3d.utility.Vector3dVector(scn_normals[cluster_mask])

        k_val = min(NORMAL_K, cluster_size - 1)
        cluster_pcd.orient_normals_consistent_tangent_plane(k=k_val, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)
        cluster_pcd.orient_normals_towards_camera_location(camera_location=camera)

        scn_normals[cluster_mask] = np.asarray(cluster_pcd.normals)

    scn.normals = o3d.utility.Vector3dVector(scn_normals)

    if DEBUG:
        print(f"[Orient] Camera consensus (Facing {camera}) applied.")

def select_best_cluster_fpfh(obj, scn, labels, cluster_sizes):
    """
    Select best matching cluster using FPFH + RANSAC fitness score.

    Based on PCL Template Alignment approach:
    - https://pointclouds.org/documentation/tutorials/template_alignment.html

    Args:
        obj: Object point cloud (template)
        scn: Scene point cloud (all clusters)
        labels: DBSCAN cluster labels
        cluster_sizes: List of point counts per cluster

    Returns:
        tuple: (selected_pcd, cluster_idx, fitness_score)
               Returns (scn, -1, 0.0) if no valid cluster found
    """
    obj_size = len(obj.points)
    pts_all = np.asarray(scn.points)

    # Filter candidates by size (0.2x to 5x of object)
    candidates = [(i, size) for i, size in enumerate(cluster_sizes)
                  if obj_size * 0.2 <= size <= obj_size * 5]

    if DEBUG:
        print(f"\n[DEBUG] CLUSTER SELECTION (FPFH-based)")
        print(f"        Object: {obj_size} pts, Candidates: {len(candidates)}")

    if not candidates:
        if DEBUG:
            print(f"        No candidates in size range!")
        return scn, -1, 0.0

    # Prepare object FPFH (compute once)
    obj_temp = copy.deepcopy(obj)
    radius_n = VOXEL_SIZE * NORMAL_RADIUS_FACTOR
    radius_f = VOXEL_SIZE * FPFH_RADIUS_FACTOR

    obj_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=NORMAL_MAX_NN))
    obj_temp.orient_normals_consistent_tangent_plane(k=NORMAL_K, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)
    obj_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        obj_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_f, max_nn=FPFH_MAX_NN))

    # Score each candidate
    best_idx, best_fitness, best_pcd = -1, -1, None
    dist_thresh = VOXEL_SIZE * RANSAC_DISTANCE_FACTOR

    for idx, size in candidates:
        # Extract cluster
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(pts_all[labels == idx])

        # Compute FPFH
        cluster_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=NORMAL_MAX_NN))
        cluster_pcd.orient_normals_consistent_tangent_plane(k=NORMAL_K, lambda_penalty=MST_LAMBDA_PENALTY, cos_alpha_tol=MST_COS_ALPHA_TOL)
        cluster_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            cluster_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_f, max_nn=FPFH_MAX_NN))

        # RANSAC for fitness score
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            obj_temp, cluster_pcd, obj_fpfh, cluster_fpfh,
            mutual_filter=True,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(RANSAC_EDGE_LENGTH),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
        )

        if DEBUG:
            print(f"        Cluster {idx}: {size} pts → fitness={result.fitness:.4f}")

        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_idx = idx
            best_pcd = cluster_pcd

    if DEBUG and best_idx >= 0:
        print(f"        >>> Selected: cluster {best_idx} (fitness={best_fitness:.4f})")

    return (best_pcd, best_idx, best_fitness) if best_pcd else (scn, -1, 0.0)


def compute_shape_features(obj, scn):
    """
    IMPORTANT: Requires consistent normal orientation for ransac so
    If normals are inconsistent (some pointing inward, some outward),
    the same surface will produce different FPFH signatures,
    causing RANSAC feature matching to fail.

    Reference:
    - https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Vertex-normal-estimation
    - https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.compute_fpfh_feature.html#open3d-pipelines-registration-compute-fpfh-feature
    
    The reference of VOXEL_SIZE * FPFH_RADIUS_FACTOR:
    - https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature
    """
    radius = VOXEL_SIZE * FPFH_RADIUS_FACTOR
    obj_features = o3d.pipelines.registration.compute_fpfh_feature(
        obj, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=FPFH_MAX_NN))
    scn_features = o3d.pipelines.registration.compute_fpfh_feature(
        scn, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=FPFH_MAX_NN))
    return obj_features, scn_features


# =============================================================================
# RANSAC (Open3D Built-in with Correspondence Checkers)
# =============================================================================

def create_kdtree(scn):
    """
    Reference: ex2_pose_est_global_solution.py:42-44
    """
    tree = o3d.geometry.KDTreeFlann(scn)
    return tree


def execute_global_registration(obj, scn, obj_fpfh, scn_fpfh):
    """
    RANSAC-based global registration with multiple correspondence checkers.

    Checkers:
    - EdgeLength: Catches 180° flips (geometric consistency)
    - Distance: Filters too-distant correspondences
    - Normal: Validates normal direction consistency (NEW)

    Reference:
    - https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    """
    distance_threshold = VOXEL_SIZE * RANSAC_DISTANCE_FACTOR

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        obj, scn, obj_fpfh, scn_fpfh,
        mutual_filter=False,  # Set to False to get more points
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(RANSAC_EDGE_LENGTH),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_MAX_ITERATION, RANSAC_CONFIDENCE)
    )

    if DEBUG:
        print(f"        RANSAC fitness: {result.fitness:.4f}, inlier_rmse: {result.inlier_rmse:.6f}")

    return result.transformation


def execute_fast_global_registration(obj, scn, obj_fpfh, scn_fpfh):
    """
    Fast Global Registration (FGR) - alternative to RANSAC.

    Faster and more robust to noise than RANSAC.
    Uses graduated non-convexity for optimization.

    Reference:
    - https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Fast-global-registration
    """
    distance_threshold = VOXEL_SIZE * 0.5  # Tighter for FGR

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        obj, scn, obj_fpfh, scn_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )

    if DEBUG:
        print(f"        FGR fitness: {result.fitness:.4f}, inlier_rmse: {result.inlier_rmse:.6f}")

    return result.transformation


def estimate_transformation(obj, scn, corr):
    """
    Reference: ex2_pose_est_global_solution.py:55-59
    Uses Kabsch algorithm (Slide 10-12)
    """
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T = est.compute_transformation(obj, scn, corr)
    return T


def apply_pose(obj, T):
    """
    Reference: ex2_pose_est_global_solution.py:46-48
    """
    obj.transform(T)
    return obj


# =============================================================================
# ICP
# =============================================================================

def find_closest_points(obj_aligned, tree, thressq):
    """
    Reference:
    - ex1_pose_est_local_solution.py:23-32
    - https://www.open3d.org/docs/release/python_api/open3d.utility.Vector2iVector.html
    """
    corr_list = []
    for j in range(len(obj_aligned.points)):
        k, idx, dist = tree.search_knn_vector_3d(obj_aligned.points[j], 1)
        if dist[0] < thressq:
            corr_list.append([j, idx[0]])

    if len(corr_list) == 0:
        return o3d.utility.Vector2iVector()

    return o3d.utility.Vector2iVector(np.array(corr_list, dtype=np.int32))


def execute_icp_point_to_point(obj, scn, initial_pose, threshold, max_iteration=2000):
    """
    Open3D ICP (Point-to-Point)

    Reference:
    - https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    """
    result = o3d.pipelines.registration.registration_icp(
        obj, scn, threshold, initial_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    if DEBUG:
        print(f"ICP fitness: {result.fitness}, inlier_rmse: {result.inlier_rmse}, correspondences: {len(result.correspondence_set)}")

    return result.transformation


def execute_icp_point_to_plane(obj, scn, initial_pose, threshold, max_iteration=2000):
    """
    Open3D ICP (Point-to-Plane)

    Reference:
    - https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    """
    result = o3d.pipelines.registration.registration_icp(
        obj, scn, threshold, initial_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    if DEBUG:
        print(f"ICP (P2Plane) fitness: {result.fitness}, inlier_rmse: {result.inlier_rmse}, correspondences: {len(result.correspondence_set)}")

    return result.transformation


def execute_icp(obj, scn, initial_pose, it, thressq):
    """
    Manual ICP implementation.
    Reference: ex1_pose_est_local_solution.py:64-78
    """
    tree = create_kdtree(scn)

    pose = initial_pose
    obj_aligned = o3d.geometry.PointCloud(obj)
    obj_aligned.transform(initial_pose)

    for i in range(it):
        corr = find_closest_points(obj_aligned, tree, thressq)

        if len(corr) < 3:
            print("Not enough correspondences found. Stopping ICP.")
            break

        T = estimate_transformation(obj_aligned, scn, corr)
        obj_aligned = apply_pose(obj_aligned, T)
        pose = T @ pose

    return pose

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def do_pose_estimation(scene_pointcloud, object_pointcloud):
    """
    Main pose estimation: RANSAC (global) + ICP (local refinement).

    Used the structure from class slides and from reference tutorial:

    reference:
        https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Global-registration
    """
    print("\nIf its stuck use: \npkill -9 python")

    obj = copy.deepcopy(object_pointcloud)
    scn = copy.deepcopy(scene_pointcloud)

    # ==========================================================================
    # STEP 0: Voxel Downsample
    # ==========================================================================
    obj_raw_n = len(obj.points)
    scn_raw_n = len(scn.points)

    obj = obj.voxel_down_sample(VOXEL_SIZE)
    scn = scn.voxel_down_sample(VOXEL_SIZE)

    if DEBUG:
        print(f"\n[DEBUG] 0. DOWNSAMPLE (voxel={VOXEL_SIZE})")
        print(f"        Object: {obj_raw_n} -> {len(obj.points)} pts")
        print(f"        Scene: {scn_raw_n} -> {len(scn.points)} pts")

    # ==========================================================================
    # STEP 1: Table Segmentation
    # ==========================================================================
    scn_before_seg = len(scn.points)
    scn = segment_table(scn)

    if DEBUG:
        print(f"\n[DEBUG] 1. TABLE SEGMENTATION")
        print(f"        Scene: {scn_before_seg} -> {len(scn.points)} pts")

    # ==========================================================================
    # STEP 2: Statistical Outlier Removal
    # ==========================================================================
    obj_before = len(obj.points)
    scn_before = len(scn.points)

    obj, _ = obj.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    scn, _ = scn.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)

    if DEBUG:
        print(f"\n[DEBUG] 2. OUTLIER REMOVAL (nb={OUTLIER_NB_NEIGHBORS}, std={OUTLIER_STD_RATIO})")
        print(f"        Object: {obj_before} -> {len(obj.points)} pts")
        print(f"        Scene: {scn_before} -> {len(scn.points)} pts")


    # ==========================================================================
    # STEP 3: DBSCAN Clustering + Cluster Selection
    # ==========================================================================
    dbscan_pcd, labels, stats = dbscan_with_colors(scn, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS)
    info = [
        f"DBSCAN: eps={DBSCAN_EPS}, min_points={DBSCAN_MIN_POINTS}",
        f"Clusters: {stats['n_clusters']}, Noise: {stats['n_noise']}"
    ]
    for i, size in enumerate(stats['cluster_sizes']):
        info.append(f"  Cluster {i}: {size} pts")
    show_point_clouds([dbscan_pcd], "3a. DBSCAN CLUSTERING", info, debug=DEBUG)

    scn = dbscan_pcd

    # ==========================================================================
    # STEP 4a: Normal Estimation (before orientation)
    # ==========================================================================
    estimate_normals(obj, scn)

    obj_before, obj_in_before, obj_out_before = color_by_normal_direction(obj)
    scn_before, scn_in_before, scn_out_before = color_by_normal_direction(scn)

    show_point_clouds([obj_before, scn_before], "4a. NORMALS BEFORE ORIENTATION", [
        f"Object - Inward(red): {obj_in_before}, Outward(blue): {obj_out_before}",
        f"Scene  - Inward(red): {scn_in_before}, Outward(blue): {scn_out_before}",
        "Random orientation - needs camera-based fix"
    ], debug=DEBUG)

    # ==========================================================================
    # STEP 4b: Normal Orientation
    # ==========================================================================

    # Option 1: MST 
    # orient_normals_mst(obj, scn)  # Global MST
    orient_normals_mst(obj, scn, labels)  # Per-cluster MST

    # Option 2: Direction-based
    # orient_normals_direction(obj, scn)

    obj_after, obj_in_after, obj_out_after = color_by_normal_direction(obj)
    scn_after, scn_in_after, scn_out_after = color_by_normal_direction(scn)

    show_point_clouds([obj_after, scn_after], "4b. NORMALS AFTER ORIENTATION", [
        f"Object - Inward(red): {obj_in_after}, Outward(blue): {obj_out_after}",
        f"Scene  - Inward(red): {scn_in_after}, Outward(blue): {scn_out_after}"
    ], debug=DEBUG)

    if DEBUG:
        print(f"\n[DEBUG] 4b. NORMAL ORIENTATION")
        print(f"        Object: {obj_in_before}/{obj_out_before} -> {obj_in_after}/{obj_out_after}")
        print(f"        Scene:  {scn_in_before}/{scn_out_before} -> {scn_in_after}/{scn_out_after}")

    # ==========================================================================
    # STEP 5a: FPFH Features
    # ==========================================================================
    obj_features, scn_features = compute_shape_features(obj, scn)

    obj_features_np = np.asarray(obj_features.data).T
    scn_features_np = np.asarray(scn_features.data).T

    radius_fpfh = VOXEL_SIZE * FPFH_RADIUS_FACTOR
    if DEBUG:
        print(f"\n[DEBUG] 5a. FPFH FEATURES (radius={radius_fpfh})")
        print(f"        Shape: obj={obj_features_np.shape}, scn={scn_features_np.shape}")

    # ==========================================================================
    # STEP 5b: Correspondences
    # ==========================================================================
    line_set, obj_corr_vis, scn_corr_vis, corr_stats = create_correspondence_lines(
        obj, scn, obj_features, scn_features
    )
    show_point_clouds([obj_corr_vis, scn_corr_vis, line_set], f"5b. CORRESPONDENCES ({corr_stats['shown']})", [
        f"Total: {corr_stats['total']}, Shown: {corr_stats['shown']}",
        f"Distance - min: {corr_stats['dist_min']}, max: {corr_stats['dist_max']}"
    ], debug=DEBUG)

    # ==========================================================================
    # STEP 6: RANSAC
    # ==========================================================================
    pose_ransac = execute_global_registration(obj, scn, obj_features, scn_features)

    obj_ransac = copy.deepcopy(obj)
    obj_ransac.transform(pose_ransac)
    ransac_dist = VOXEL_SIZE * RANSAC_DISTANCE_FACTOR
    show_point_clouds([obj_ransac, scn], "6. RANSAC", [
        f"Distance: {ransac_dist}, EdgeLength: {RANSAC_EDGE_LENGTH}"
    ], debug=DEBUG)

    # ==========================================================================
    # STEP 7: ICP Refinement
    # ==========================================================================
    icp_threshold = VOXEL_SIZE * ICP_DISTANCE_FACTOR

    # Use Point-to-Plane ICP
    # pose_final = execute_icp_point_to_plane(obj, scn, pose_ransac, icp_threshold, max_iteration=ICP_MAX_ITERATION)

    # Alternative: Point-to-Point ICP
    pose_final = execute_icp_point_to_point(obj, scn, pose_ransac, icp_threshold, max_iteration=ICP_MAX_ITERATION)

    obj_icp = copy.deepcopy(obj)
    obj_icp.transform(pose_final)
    show_point_clouds([obj_icp, scn], "7. ICP", [
        f"ICP, Threshold: {icp_threshold}"
    ], debug=DEBUG)

    return pose_final
