"""
Full Robot + Camera Integration with Hybrid P2P.

Combines vision-based pose estimation with robot manipulation:
- I0: Randomize duck position
- I1: Pose estimation (RANSAC + ICP)
- I2: Compute dynamic grasp frames from estimated pose
- I3: Execute pick-and-place with hybrid P2P

Hybrid P2P Architecture:
- DYNAMIC frames (0-1): Approach + Grasp from pose estimation
- STATIC frames (2+): Transport + Drop from config.yaml

Reference:
- Vision pipeline: vision_scripts/do_pe.py
- Robot control: scripts/manipulation.py

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import mujoco as mj
import open3d as o3d
import numpy as np
import math
import time
import random
import copy

from spatialmath import SE3, SO3
from spatialmath.base import trnorm  # Normalizes 4x4 matrix rotation
from scipy.spatial.transform import Rotation

from cam import get_pointcloud, get_camera_pose_cv
from utils import load_config, hande_ctrl_qpos, ur_set_qpos, Q_HOME, get_gripper_config
from utils.logger import ProjectLogger

# Integration-specific imports
from vision_scripts.do_pe import do_pose_estimation
from vision_scripts.helpers import computeError
from scripts.ik_simple import plan_to_frame
from scripts.manipulation import return_to_home


# =============================================================================
# CONSTANTS
# =============================================================================

APPROACH_HEIGHT = 0.02  # 2cm above duck for approach
GRASP_PENETRATION = -0.01  # 2cm inside duck center (gripper goes deeper)


# =============================================================================
# ROBOT RESET
# =============================================================================

def reset_robot_to_home(d, m, viewer=None):
    """
    Reset robot to home position with gripper open.

    Uses mj_forward() for instant positioning without physics simulation.
    Call this at the start of each integration run.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        viewer: Optional viewer to sync after reset
    """
    config = load_config()
    gripper_cfg = get_gripper_config(config, "integration_project")

    # Set robot joints to home position
    ur_set_qpos(d, Q_HOME)

    # Open gripper
    hande_ctrl_qpos(d, gripper_cfg['open_value'])

    # Update kinematics (no physics)
    mj.mj_forward(m, d)

    # Sync viewer if provided
    if viewer is not None:
        viewer.sync()


# =============================================================================
# DUCK RANDOMIZATION (from cv_demo.py)
# =============================================================================

def _r2q(rot):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w] (scipy convention)."""
    r = Rotation.from_matrix(rot)
    return r.as_quat()


def randomize_duck_position(d, m, viewer=None, settle_time=1.0, verbose=True):
    """
    Randomize duck position and let it settle naturally with physics.

    Workspace bounds (from cv_demo.py):
    - X: 0.45 to 0.75 meters
    - Y: -0.55 to 0.55 meters
    - Z: 0.025 meters (on table)
    - Rotation: random 0-359 degrees around Z

    Args:
        d: MuJoCo data
        m: MuJoCo model
        viewer: Optional viewer for visual feedback during settling
        settle_time: Time in seconds to let physics settle (default 1.0s)
        verbose: Print position info

    Returns:
        SE3: Ground truth duck pose in world frame (after settling)
    """
    # Random position on table (slightly above to let it fall)
    rand_x = random.uniform(0.45, 0.75)
    rand_y = random.uniform(-0.55, 0.55)
    z = 0.05  # Start slightly above table to let it fall and settle

    # Set position
    d.joint('duck').qpos[0:3] = [rand_x, rand_y, z]

    # Random rotation around Z
    rand_rot_deg = random.randint(0, 359)
    rot = SO3.Eul(rand_rot_deg, 90, 90, unit="deg").R
    d.joint('duck').qpos[3:] = _r2q(rot)

    # Zero initial velocity
    d.joint('duck').qvel[:] = 0

    if verbose:
        print(f"  Duck spawned: pos=[{rand_x:.3f}, {rand_y:.3f}, {z:.3f}], rot={rand_rot_deg}°")
        print(f"  Letting physics settle for {settle_time}s...")

    # Let physics run so duck settles naturally on table
    settle_steps = int(settle_time / m.opt.timestep)
    for i in range(settle_steps):
        mj.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

    # NOW freeze duck after it has settled
    d.joint('duck').qvel[:] = 0
    mj.mj_forward(m, d)

    # Get duck pose as SE3 (after settling)
    duck_pos = d.body('duck').xpos.copy()
    duck_rot = d.body('duck').xmat.reshape(3, 3).copy()
    duck_rot = trnorm(duck_rot)
    duck_se3 = SE3.Rt(duck_rot, duck_pos)

    if verbose:
        print(f"  Duck settled at: pos=[{duck_pos[0]:.3f}, {duck_pos[1]:.3f}, {duck_pos[2]:.3f}]")

    return duck_se3


def freeze_duck(d, m):
    """
    Freeze duck in current position.
    Saves position and zeros velocity so physics won't move it.

    Call this after randomization to keep duck stationary until grasped.
    """
    # Save current position
    saved_qpos = d.joint('duck').qpos.copy()

    # Zero velocity (6 components: 3 linear + 3 angular for free joint)
    d.joint('duck').qvel[:] = 0

    # Restore position (in case physics moved it slightly)
    d.joint('duck').qpos[:] = saved_qpos

    # Update kinematics without physics
    mj.mj_forward(m, d)


# =============================================================================
# POSE ESTIMATION WRAPPER
# =============================================================================

def run_pose_estimation(d, m, object_mesh_path, camera_name="cam1", verbose=True):
    """
    Run full pose estimation pipeline.

    Steps:
    1. Capture point cloud from camera
    2. Load object mesh and sample to point cloud
    3. Run RANSAC + ICP pose estimation
    4. Transform result to world frame

    Args:
        d: MuJoCo data
        m: MuJoCo model
        object_mesh_path: Path to object STL/PLY file
        camera_name: Camera name in MuJoCo
        verbose: Print debug info

    Returns:
        SE3: Estimated object pose in world frame, or None if failed
    """
    # Create renderer and capture scene point cloud
    renderer = mj.Renderer(m, height=480, width=640)
    scene_pcd_path = "scene_temp.pcd"
    get_pointcloud(m, d, renderer, scene_pcd_path, camera_name=camera_name)

    # Load point clouds
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    # Load object mesh and sample points
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    object_pcd = object_mesh.sample_points_poisson_disk(10000)

    if verbose:
        print(f"  Scene points: {len(scene_pcd.points)}")
        print(f"  Object points: {len(object_pcd.points)}")

    # Run pose estimation (returns 4x4 transform in CAMERA frame)
    try:
        pose_camera = do_pose_estimation(scene_pcd, object_pcd)
    except Exception as e:
        print(f"  [ERROR] Pose estimation failed: {e}")
        return None

    # ==========================================================================
    # COORDINATE TRANSFORM: Camera Frame → World Frame
    # ==========================================================================
    # From cv_demo.py: gt = cam_se3.inv() * duck_se3
    # This gives: T_camera_duck = T_camera_world × T_world_duck
    #
    # To get world pose from camera pose (inverse):
    # T_world_duck = T_world_camera × T_camera_duck
    #              = cam_se3 × pose_camera
    #
    # ERROR FIX: SE3(pose_camera) fails with "bad argument to constructor"
    # because ICP result may have numerical errors causing rotation matrix
    # to be slightly non-orthonormal. trnorm() normalizes the 4x4 matrix.
    # ==========================================================================
    cam_se3 = get_camera_pose_cv(m, d, camera_name=camera_name)
    object_pose_world = cam_se3 * SE3(trnorm(pose_camera))

    if verbose:
        print(f"  Estimated pose (world): {object_pose_world.t}")

    return object_pose_world


# =============================================================================
# GRASP FRAME COMPUTATION (SE3 poses for RRT)
# =============================================================================

def compute_grasp_frames(duck_pose_world):
    """
    Compute approach and grasp SE3 frames from pose estimation result.

    Grasp Strategy:
    - Top-down approach (gripper pointing down)
    - Use position from pose estimation
    - Extract Z-rotation (yaw) from estimated pose for gripper alignment
    - This allows gripper to align with duck's thin axis for better grasp

    Args:
        duck_pose_world: SE3 estimated duck pose in world frame

    Returns:
        tuple: (approach_frame, grasp_frame) as SE3 poses
    """
    logger = ProjectLogger.get_instance()

    # Extract position from estimated pose
    duck_position = duck_pose_world.t

    # Extract Z-rotation (yaw) from estimated pose
    # This helps align gripper with duck's orientation (thin vs thick axis)
    # Convert rotation matrix to Euler angles and extract Z rotation
    r = Rotation.from_matrix(duck_pose_world.R)
    euler = r.as_euler('xyz', degrees=False)
    yaw = euler[2]  # Z-rotation (yaw)

    logger.debug(f"[GRASP] Duck yaw: {np.degrees(yaw):.1f}°")

    # Gripper orientation:
    # 1. Rx(π) - point gripper downward (top-down approach)
    # 2. Rz(yaw) - align gripper with duck's orientation
    # Order: first rotate around Z (align with duck), then rotate around X (point down)
    gripper_orientation = SE3.Rz(yaw) * SE3.Rx(math.pi)

    # Grasp frame: duck position + small offset into object + aligned orientation
    grasp_pos = duck_position + np.array([0, 0, GRASP_PENETRATION])
    grasp_frame = SE3.Rt(R=gripper_orientation.R, t=grasp_pos, check=False)

    # Approach frame: same orientation, higher position
    approach_pos = duck_position + np.array([0, 0, APPROACH_HEIGHT])
    approach_frame = SE3.Rt(R=gripper_orientation.R, t=approach_pos, check=False)

    logger.debug(f"[GRASP] Duck position: {duck_position}")
    logger.debug(f"[GRASP] Approach pos: {approach_frame.t}")
    logger.debug(f"[GRASP] Grasp pos: {grasp_frame.t}")

    return approach_frame, grasp_frame


# =============================================================================
# P2P-BASED PICK (Dynamic - from pose estimation, NO collision check)
# =============================================================================

def pick_with_p2p(robot, approach_frame, grasp_frame, d, m, viewer, execute_fn,
                  log_list=None, boundary_list=None):
    """
    Pick object using direct P2P (no collision checking).

    Since we approach from above with fixed orientation, collision checking
    is not needed. This is simpler and faster than RRT.

    Flow:
    1. IK: compute approach joint config
    2. IK: compute grasp joint config
    3. P2P: current → approach
    4. P2P: approach → grasp
    5. Close gripper

    Args:
        robot: UR5robot instance
        approach_frame: SE3 approach pose (from pose estimation)
        grasp_frame: SE3 grasp pose (from pose estimation)
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        log_list, boundary_list: Optional logging

    Returns:
        bool: True if successful
    """
    logger = ProjectLogger.get_instance()
    config = load_config()
    gripper_cfg = get_gripper_config(config, "integration_project")

    # Get timing from integration_project config
    project_config = config.get('integration_project', {})
    timing = project_config.get('timing', {}).get('p2p', {})
    segment_time = timing.get('segment_time', 500)

    logger.info("[P2P-PICK] Starting P2P-based pick sequence (no collision check)")

    # Step 0: Ensure gripper is fully open before approaching
    logger.info("[P2P-PICK] Opening gripper fully...")
    robot.set_gripper(gripper_cfg['open_value'], t=gripper_cfg['open_time'])
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    # Step 1: Compute IK for approach frame (no collision check)
    logger.info("[P2P-PICK] Computing IK for approach frame...")
    start_q = robot.get_current_q()

    approach_ik = robot.robot_ur5.ik_LM(Tep=approach_frame, q0=start_q)
    if not approach_ik[1]:
        logger.error("[P2P-PICK] IK failed for approach frame!")
        return False
    approach_q = approach_ik[0]
    logger.info(f"[P2P-PICK] Approach IK: J1={np.degrees(approach_q[0]):.1f}°")

    # Step 2: Compute IK for grasp frame
    logger.info("[P2P-PICK] Computing IK for grasp frame...")
    grasp_ik = robot.robot_ur5.ik_LM(Tep=grasp_frame, q0=approach_q)
    if not grasp_ik[1]:
        logger.error("[P2P-PICK] IK failed for grasp frame!")
        return False
    grasp_q = grasp_ik[0]
    logger.info(f"[P2P-PICK] Grasp IK: J1={np.degrees(grasp_q[0]):.1f}°")

    # Step 3: P2P to approach
    logger.info("[P2P-PICK] Moving to approach position...")
    robot.move_j(start_q=start_q, end_q=approach_q, t=segment_time)
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    # Step 4: P2P to grasp (descent)
    logger.info("[P2P-PICK] Descending to grasp position...")
    robot.move_j(start_q=approach_q, end_q=grasp_q, t=segment_time)
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    # Step 5: Close gripper
    logger.info("[P2P-PICK] Closing gripper...")
    robot.set_gripper(gripper_cfg['close_value'], t=gripper_cfg['close_time'])
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    logger.info("[P2P-PICK] Pick sequence complete!")
    return True


# =============================================================================
# RRT-BASED PICK (Dynamic - from pose estimation) - BACKUP
# =============================================================================

def pick_with_rrt(robot, approach_frame, grasp_frame, d, m, viewer, execute_fn,
                  log_list=None, boundary_list=None):
    """
    Pick object using RRT path planning to estimated grasp position.

    Flow:
    1. RRT: current → approach frame
    2. RRT: approach → grasp frame
    3. Close gripper
    4. RRT: grasp → lift frame

    Args:
        robot: UR5robot instance
        approach_frame: SE3 approach pose (from pose estimation)
        grasp_frame: SE3 grasp pose (from pose estimation)
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        log_list, boundary_list: Optional logging

    Returns:
        bool: True if successful
    """
    logger = ProjectLogger.get_instance()
    config = load_config()
    gripper_cfg = get_gripper_config(config, "integration_project")

    # Get timing from integration_project config
    project_config = config.get('integration_project', {})
    timing = project_config.get('timing', {}).get('rrt', {})
    path_time = timing.get('path_time', 300)
    descent_time = timing.get('descent_time', 500)
    lift_time = timing.get('lift_time', 500)

    logger.info("[RRT-PICK] Starting RRT-based pick sequence")

    # Step 1: RRT to approach frame
    logger.info("[RRT-PICK] Planning path to approach frame...")
    start_q = robot.get_current_q()

    approach_path = plan_to_frame(
        d=d, m=m,
        start_q=start_q,
        target_frame=approach_frame,
        robot=robot.robot_ur5,
        planner_type="rrt",
        target_object=None,  # Check all collisions
        held_object=None,
        ik_mode="simple"
    )

    if not approach_path:
        logger.error("[RRT-PICK] Failed to plan path to approach frame!")
        return False

    # Execute approach path
    robot.add_planned_path(approach_path)
    robot.visualize_trajectory(viewer, use_queue=False)
    robot.move_j_via(points=approach_path, t=path_time)
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
    logger.info(f"[RRT-PICK] Approach complete ({len(approach_path)} waypoints)")

    # Step 2: Descend to grasp frame (P2P - no collision check, just go down)
    logger.info("[RRT-PICK] Descending to grasp frame (P2P - no collision check)...")
    current_q = robot.get_current_q()

    # Direct IK for grasp position (no collision checking)
    grasp_ik = robot.robot_ur5.ik_LM(Tep=grasp_frame, q0=current_q)
    if not grasp_ik[1]:
        logger.error("[RRT-PICK] IK failed for grasp frame!")
        return False
    grasp_q = grasp_ik[0]
    logger.info(f"[RRT-PICK] Grasp IK success: J1={np.degrees(grasp_q[0]):.1f}°")

    # P2P descent (direct interpolation, no collision checking)
    robot.move_j(start_q=current_q, end_q=grasp_q, t=descent_time)
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
    logger.info("[RRT-PICK] Descent complete")

    # Step 3: Close gripper
    logger.info("[RRT-PICK] Closing gripper...")
    robot.set_gripper(gripper_cfg['close_value'], t=gripper_cfg['close_time'])
    _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    # NOTE: No lift step here!
    # Lift is handled by P2P - first static frame in config.yaml is the lift position
    # This keeps RRT focused on dynamic grasp, P2P handles transport

    logger.info("[RRT-PICK] Pick sequence complete! (P2P will handle lift)")
    return True


# =============================================================================
# P2P-BASED PLACE (Static - from config.yaml)
# =============================================================================

def place_with_p2p(robot, static_frames, gripper_open_after, d, m, viewer, execute_fn,
                   log_list=None, boundary_list=None):
    """
    Place object using P2P static waypoints from config.yaml.

    Flow:
    1. P2P: through transport waypoints
    2. P2P: to drop position
    3. Open gripper

    Args:
        robot: UR5robot instance
        static_frames: List of frame dicts from config.yaml (transport + drop)
        gripper_open_after: Frame index to open gripper
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        log_list, boundary_list: Optional logging

    Returns:
        bool: True if successful
    """
    logger = ProjectLogger.get_instance()
    config = load_config()
    gripper_cfg = get_gripper_config(config, "integration_project")

    # Get timing from integration_project config
    project_config = config.get('integration_project', {})
    timing = project_config.get('timing', {}).get('p2p', {})
    segment_time = timing.get('segment_time', 500)

    logger.info(f"[P2P-PLACE] Executing {len(static_frames)} static frames")
    logger.info(f"[P2P-PLACE] Gripper open after frame: {gripper_open_after}")

    q_current = robot.get_current_q()

    for frame_idx, frame_def in enumerate(static_frames):
        waypoint = np.array(frame_def['q'])
        logger.debug(f"[P2P-PLACE] Frame {frame_idx}: J1={np.degrees(waypoint[0]):.1f}°")

        # Move to waypoint
        robot.move_j(start_q=q_current, end_q=waypoint, t=segment_time)
        _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        q_current = waypoint

        # Gripper control
        if frame_idx == gripper_open_after:
            logger.info(f"[P2P-PLACE] Gripper OPEN after frame {frame_idx}")
            robot.set_gripper(gripper_cfg['open_value'], t=gripper_cfg['release_time'])
            _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    logger.info("[P2P-PLACE] Place sequence complete!")
    return True


# =============================================================================
# HYBRID P2P EXECUTION
# =============================================================================

def execute_hybrid_p2p(robot, dynamic_frames, static_frames,
                       gripper_close_after, gripper_open_after,
                       d, m, viewer, execute_fn,
                       log_list=None, boundary_list=None):
    """
    Execute hybrid P2P with dynamic + static frames.

    Frame sequence:
    - 0: Dynamic approach (from pose estimation)
    - 1: Dynamic grasp (from pose estimation)
    - 2+: Static transport + drop (from config.yaml)

    Args:
        robot: UR5robot instance
        dynamic_frames: [approach_q, grasp_q] from pose estimation
        static_frames: List of frame dicts from config.yaml
        gripper_close_after: Frame index to close gripper (1 for after grasp)
        gripper_open_after: Frame index to open gripper
        d, m: MuJoCo data/model
        viewer: MuJoCo viewer
        execute_fn: Movement execution function
        log_list, boundary_list: Optional logging

    Returns:
        bool: True if successful
    """
    logger = ProjectLogger.get_instance()
    config = load_config()
    gripper_cfg = get_gripper_config(config, "integration_project")

    # Get timing from integration_project config
    project_config = config.get('integration_project', {})
    timing = project_config.get('timing', {}).get('p2p', {})
    segment_time = timing.get('segment_time', 500)

    # Combine frames: dynamic (np.array) + static (dicts with 'q')
    all_frames = []

    # Add dynamic frames (already np.arrays)
    for i, q in enumerate(dynamic_frames):
        all_frames.append(('dynamic', i, q))

    # Add static frames (extract 'q' from dicts)
    for i, frame_def in enumerate(static_frames):
        q = np.array(frame_def['q'])
        all_frames.append(('static', i, q))

    logger.info(f"[HYBRID-P2P] Executing {len(all_frames)} frames")
    logger.info(f"  Dynamic: {len(dynamic_frames)}, Static: {len(static_frames)}")
    logger.info(f"  Gripper close after: {gripper_close_after}, open after: {gripper_open_after}")

    q_current = robot.get_current_q()

    for frame_idx, (frame_type, sub_idx, waypoint) in enumerate(all_frames):
        logger.debug(f"[HYBRID-P2P] Frame {frame_idx} ({frame_type} #{sub_idx})")

        # Move to waypoint
        robot.move_j(start_q=q_current, end_q=waypoint, t=segment_time)
        _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)
        q_current = waypoint

        # Gripper control
        if frame_idx == gripper_close_after:
            logger.info(f"[HYBRID-P2P] Gripper CLOSE after frame {frame_idx}")
            robot.set_gripper(gripper_cfg['close_value'], t=gripper_cfg['close_time'])
            _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

        if frame_idx == gripper_open_after:
            logger.info(f"[HYBRID-P2P] Gripper OPEN after frame {frame_idx}")
            robot.set_gripper(gripper_cfg['open_value'], t=gripper_cfg['release_time'])
            _execute_movement_wrapper(robot, d, m, viewer, execute_fn, log_list, boundary_list)

    logger.info("[HYBRID-P2P] Sequence complete!")
    return True


def _execute_movement_wrapper(robot, d, m, viewer, execute_fn,
                               log_list=None, boundary_list=None):
    """Helper to visualize, execute and log movement."""
    robot.visualize_trajectory(viewer, use_queue=True, sample_rate=5)
    execute_fn(robot, robot.queue.copy(), d, m, viewer)

    if log_list is not None:
        log_list.extend([q.copy() for q, _ in robot.queue])
        if boundary_list is not None:
            boundary_list.append(len(log_list))

    robot.queue.clear()


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def run_integration(d, m, viewer, robot, execute_fn, config, num_runs=3):
    """
    Main integration loop - pose estimation + robot manipulation.

    Flow per run:
    1. I0: Randomize duck position
    2. I1: Capture point cloud and run pose estimation
    3. I2: Compute dynamic approach + grasp frames
    4. I3: Execute hybrid P2P (dynamic + static frames)
    5. Return to home

    Press 'q' during execution to skip current run and reset to next.
    Press Ctrl+C to abort all runs.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        viewer: MuJoCo viewer
        robot: UR5robot instance
        execute_fn: Movement execution function
        config: Loaded config dict
        num_runs: Number of integration runs (default 3)

    Returns:
        list: Results for each run
    """
    logger = ProjectLogger.get_instance()

    # Get duck configuration from integration_project config
    project_config = config.get('integration_project', {})
    p2p_config = project_config.get('p2p_frames', {}).get('duck', {})
    static_frames = p2p_config.get('frames', [])

    # Object mesh path for pose estimation (using highest detail mesh)
    object_mesh_path = "assets/duck/duck_3.obj"

    results = []

    print("\n" + "="*60)
    print("Press 'q' + Enter during execution to skip to next run")
    print("Press Ctrl+C to abort all runs")
    print("="*60)

    for run_id in range(num_runs):
        print(f"\n{'='*60}")
        print(f"INTEGRATION RUN {run_id + 1}/{num_runs}")
        print(f"{'='*60}")

        run_start = time.time()
        run_result = {
            'run_id': run_id + 1,
            'success': False,
            'pose_error': None,
            'duration': None,
            'skipped': False
        }

        try:
            # =================================================================
            # RESET: Ensure robot starts from HOME position
            # =================================================================
            print("\n[RESET] Setting robot to home position...")
            reset_robot_to_home(d, m, viewer)
            print(f"  Robot at home: {np.degrees(Q_HOME)[:3]}... (first 3 joints in deg)")

            # =================================================================
            # I0: Randomize duck position (let physics settle, then freeze)
            # =================================================================
            print("\n[I0] Randomizing duck position...")
            gt_pose = randomize_duck_position(d, m, viewer=viewer, settle_time=2.0, verbose=True)
            # Duck is now frozen after settling naturally on the table

            # =================================================================
            # I1: Pose estimation
            # =================================================================
            print("\n[I1] Running pose estimation...")
            estimated_pose = run_pose_estimation(d, m, object_mesh_path)

            if estimated_pose is None:
                print("  [FAILED] Pose estimation failed - skipping run")
                results.append(run_result)
                continue

            # Compute error
            error_angle, error_pos = computeError(gt_pose.A, estimated_pose.A)
            run_result['pose_error'] = {'angle_deg': error_angle, 'pos_mm': error_pos}
            print(f"  Error: {error_angle:.2f}°, {error_pos:.2f}mm")

            # =================================================================
            # I2: Compute grasp frames (SE3 poses for RRT)
            # =================================================================
            print("\n[I2] Computing grasp frames from estimated pose...")
            approach_frame, grasp_frame = compute_grasp_frames(estimated_pose)
            print(f"  Approach: {approach_frame.t}")
            print(f"  Grasp: {grasp_frame.t}")

            # =================================================================
            # I3a: RRT-based PICK (dynamic - from pose estimation)
            # =================================================================
            print("\n[I3a] RRT Pick - approach + grasp...")
            obj_trajectory = []
            obj_boundaries = [0]

            pick_success = pick_with_rrt(
                robot, approach_frame, grasp_frame,
                d, m, viewer, execute_fn,
                obj_trajectory, obj_boundaries
            )

            if not pick_success:
                print("  [FAILED] RRT pick failed - skipping run")
                results.append(run_result)
                continue

            # =================================================================
            # I3b: P2P-based PLACE (static - from config.yaml)
            # =================================================================
            print("\n[I3b] P2P Place - transport + drop...")

            # gripper_open_after is relative to static frames (last frame = drop)
            gripper_open_at = len(static_frames) - 1  # Open after last frame (drop position)

            place_success = place_with_p2p(
                robot, static_frames, gripper_open_at,
                d, m, viewer, execute_fn,
                obj_trajectory, obj_boundaries
            )

            if not place_success:
                print("  [FAILED] P2P place failed")
                results.append(run_result)
                continue

            # =================================================================
            # Return to home
            # =================================================================
            print("\n[I4] Returning to home position...")
            return_to_home(robot, "p2p", d, m, viewer, execute_fn,
                           obj_trajectory, obj_boundaries)

            run_result['success'] = True
            run_result['duration'] = time.time() - run_start
            results.append(run_result)

            print(f"\n[DONE] Run {run_id + 1} complete in {run_result['duration']:.1f}s")
            print(f"       Pose error: {error_angle:.2f}°, {error_pos:.2f}mm")

        except KeyboardInterrupt:
            # User pressed Ctrl+C - ask if they want to skip or abort
            print("\n" + "="*60)
            print("[INTERRUPTED] Press 'q' + Enter to skip to next run, Ctrl+C again to abort")
            print("="*60)
            try:
                user_input = input("Your choice: ").strip().lower()
                if user_input == 'q':
                    print(f"[SKIP] Skipping run {run_id + 1}, resetting robot...")
                    run_result['skipped'] = True
                    results.append(run_result)
                    # Reset robot to home before next run
                    reset_robot_to_home(d, m, viewer)
                    continue
                else:
                    print("[ABORT] Aborting all runs...")
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                print("\n[ABORT] Aborting integration...")
                break

        except Exception as e:
            # Any other error - log and skip to next run
            print(f"\n[ERROR] Run {run_id + 1} failed with error: {e}")
            print("[SKIP] Resetting robot and continuing to next run...")
            run_result['skipped'] = True
            run_result['error'] = str(e)
            results.append(run_result)
            # Reset robot to home before next run
            try:
                reset_robot_to_home(d, m, viewer)
            except Exception:
                pass  # Best effort reset
            continue

    # Summary
    print(f"\n{'='*60}")
    print("INTEGRATION SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['success'])
    skipped = sum(1 for r in results if r.get('skipped', False))
    failed = len(results) - successful - skipped
    print(f"Successful: {successful}/{num_runs}  |  Skipped: {skipped}  |  Failed: {failed}")

    for r in results:
        if r['success']:
            status = "✓"
        elif r.get('skipped', False):
            status = "⊘"  # Skipped symbol
        else:
            status = "✗"

        if r['pose_error']:
            print(f"  Run {r['run_id']}: {status} - {r['pose_error']['angle_deg']:.2f}°, {r['pose_error']['pos_mm']:.2f}mm")
        elif r.get('skipped', False):
            print(f"  Run {r['run_id']}: {status} - SKIPPED by user")
        else:
            print(f"  Run {r['run_id']}: {status} - No pose data")

    return results
