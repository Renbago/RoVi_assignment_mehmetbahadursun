"""
P2P IK Planning Module - Runtime IK computation for Cartesian frame definitions.

IMPORTANT NOTE:
Bu modül sadece Cartesian frame tanımları için gereklidir!
Eğer config.yaml'da tüm frame'ler type: "joint" ise (şu anki durum),
bu modül KULLANILMAZ - manipulation.py direkt q değerlerini okur.

Cartesian frame desteği gerekirse bu modülü kullanın:
- absolute: World frame position + RPY
- object_approach, object_grasp, object_lift: GRASP_CONFIG kullanır
- drop_approach, drop_place: Drop point frame'leri

Architecture:
- P2PFramePlanner: Main class for frame-to-IK conversion
- Sequential IK seeding prevents J1 wrap-around

Reference:
- roboticstoolbox IK: https://petercorke.github.io/robotics-toolbox-python/
- UR5 IK Solutions: https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import spatialmath as sm
from typing import Optional, List, Dict, Any

from utils.logger import ProjectLogger
from utils.mujoco_utils import (
    load_config, get_mjobj_frame, is_q_valid,
    GRASP_CONFIG, DEFAULT_GRASP_CONFIG
)


# Approach height for pick/place operations
APPROACH_HEIGHT = 0.15


class P2PFramePlanner:
    """
    Computes IK for P2P frame sequences with sequential seeding.

    Key Features:
    - Converts frame definitions (from config.yaml) to SE3 poses
    - Sequential IK seeding (uses q[i-1] as seed for q[i])
    - J1 sweep fallback on IK failure
    - Collision validation for each configuration

    Usage:
        planner = P2PFramePlanner(d, m, robot.robot_ur5)
        frames = [planner.frame_to_se3(f, obj_name) for f in frame_defs]
        waypoints = planner.compute_ik_sequence(frames, start_q)

    Reference:
    - https://petercorke.github.io/robotics-toolbox-python/
    """

    def __init__(self, d, m, robot_rtb, config: Optional[Dict] = None):
        """
        Initialize P2PFramePlanner.

        Args:
            d: MuJoCo data
            m: MuJoCo model
            robot_rtb: roboticstoolbox robot (for IK/FK)
            config: Optional config dict (loads from file if None)
        """
        self.d = d
        self.m = m
        self.robot = robot_rtb
        self.config = config or load_config()
        self.logger = ProjectLogger.get_instance()

    def frame_to_se3(self, frame_def: Dict[str, Any], obj_name: str):
        """
        Convert frame definition dict to SE3 pose OR joint configuration.

        Supported frame types:
        - absolute: World frame position + RPY orientation
        - object_approach: Above object using GRASP_CONFIG
        - object_grasp: At object using GRASP_CONFIG
        - object_lift: Lifted position after grasp
        - drop_approach: Above drop point
        - drop_place: At drop point
        - joint: Direct joint configuration (q values) - NO IK needed!

        Args:
            frame_def: Frame definition dict from config
            obj_name: Object name for object-relative frames

        Returns:
            SE3 pose OR np.ndarray (for 'joint' type)
        """
        frame_type = frame_def.get('type', 'absolute')

        if frame_type == 'joint':
            # Direct joint angles - return as numpy array, not SE3
            return np.array(frame_def['q'])
        elif frame_type == 'absolute':
            return self._make_absolute_frame(frame_def)
        elif frame_type == 'object_approach':
            return self._make_object_approach(frame_def, obj_name)
        elif frame_type == 'object_grasp':
            return self._make_object_grasp(frame_def, obj_name)
        elif frame_type == 'object_lift':
            return self._make_object_lift(frame_def, obj_name)
        elif frame_type == 'drop_approach':
            return self._make_drop_approach(frame_def, obj_name)
        elif frame_type == 'drop_place':
            return self._make_drop_place(frame_def, obj_name)
        else:
            self.logger.error(f"Unknown frame type: {frame_type}")
            raise ValueError(f"Unknown frame type: {frame_type}")

    def compute_ik_sequence(self, frames: List,
                            start_q: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Compute IK for frame sequence with SEQUENTIAL SEEDING.

        CRITICAL: Each IK uses previous solution as seed.
        This prevents J1 wrap-around by ensuring IK converges to nearby solutions.

        Supports mixed frames:
        - SE3 poses: IK computed with sequential seeding
        - np.ndarray (joint type): Used directly, no IK

        Args:
            frames: List of SE3 target poses OR np.ndarray (joint configs)
            start_q: Starting joint configuration (used as first seed)

        Returns:
            List of joint configurations, or None if IK fails
        """
        if not frames:
            self.logger.error("[P2P-IK] No frames provided!")
            return None

        waypoints = []
        current_seed = np.array(start_q)

        self.logger.info(f"[P2P-IK] Computing IK for {len(frames)} frames")

        for i, frame in enumerate(frames):
            # Check if frame is already a joint configuration (np.ndarray)
            if isinstance(frame, np.ndarray):
                # Direct joint angles - no IK needed
                q_result = frame
                self.logger.debug(f"[P2P-IK] Frame {i}: Direct joint (no IK)")
            else:
                # SE3 frame - compute IK
                q_result = self._solve_ik_with_fallback(
                    target_frame=frame,
                    seed_q=current_seed,
                    frame_index=i
                )

                if q_result is None:
                    self.logger.error(f"[P2P-IK] IK failed at frame {i}!")
                    return None

            waypoints.append(q_result)
            current_seed = q_result  # Sequential seeding - KEY!

            # Log J1 for debugging
            j1_deg = np.degrees(q_result[0])
            self.logger.debug(f"[P2P-IK] Frame {i}: J1 = {j1_deg:.1f}°")

        # Verify J1 continuity
        self._verify_j1_continuity(waypoints)

        return waypoints

    def _solve_ik_with_fallback(self, target_frame: sm.SE3,
                                 seed_q: np.ndarray,
                                 frame_index: int) -> Optional[np.ndarray]:
        """
        Solve IK with fallback strategy.

        Strategy:
        1. Direct IK with seed
        2. J1 sweep around seed (±45°, 8 samples)
        3. Return None if all fail

        Args:
            target_frame: SE3 target pose
            seed_q: Seed joint configuration
            frame_index: Frame index for logging

        Returns:
            Joint configuration or None
        """
        # Primary: Direct IK with seed
        result = self.robot.ik_LM(Tep=target_frame, q0=seed_q)
        if result[1]:  # IK converged
            q_solution = result[0]
            if is_q_valid(self.d, self.m, q_solution):
                return q_solution
            else:
                self.logger.debug(f"[P2P-IK] Frame {frame_index}: Primary IK collision")

        # Fallback: J1 sweep around seed
        self.logger.debug(f"[P2P-IK] Frame {frame_index}: Primary IK failed, trying J1 sweep")

        for theta_offset in np.linspace(-np.pi/4, np.pi/4, 8):
            q_init = seed_q.copy()
            q_init[0] += theta_offset  # Sweep J1

            result = self.robot.ik_LM(Tep=target_frame, q0=q_init)
            if result[1]:
                q_solution = result[0]
                if is_q_valid(self.d, self.m, q_solution):
                    self.logger.debug(f"[P2P-IK] Frame {frame_index}: "
                                     f"J1 sweep success at offset {np.degrees(theta_offset):.1f}°")
                    return q_solution

        self.logger.error(f"[P2P-IK] Frame {frame_index}: All IK attempts failed!")
        return None

    def _verify_j1_continuity(self, waypoints: List[np.ndarray]) -> None:
        """
        Verify J1 continuity - warn if large jumps detected.

        Args:
            waypoints: List of joint configurations
        """
        max_j1_jump = 0.0
        max_jump_pair = (0, 1)

        for i in range(len(waypoints) - 1):
            j1_diff = abs(waypoints[i+1][0] - waypoints[i][0])
            # Handle wrap-around check
            j1_diff = min(j1_diff, 2*np.pi - j1_diff)

            if j1_diff > max_j1_jump:
                max_j1_jump = j1_diff
                max_jump_pair = (i, i+1)

        max_j1_jump_deg = np.degrees(max_j1_jump)

        if max_j1_jump_deg > 90:
            self.logger.warning(f"[P2P-IK] Large J1 jump: {max_j1_jump_deg:.1f}° "
                               f"between waypoints {max_jump_pair[0]} and {max_jump_pair[1]}")
        else:
            self.logger.info(f"[P2P-IK] J1 continuity OK (max jump: {max_j1_jump_deg:.1f}°)")

    # =========================================================================
    # Frame Type Implementations
    # =========================================================================

    def _make_absolute_frame(self, frame_def: Dict) -> sm.SE3:
        """
        Create SE3 from absolute position and RPY orientation.

        Args:
            frame_def: Dict with 'position' [x,y,z] and 'orientation' [R,P,Y] in degrees

        Returns:
            SE3 pose
        """
        pos = np.array(frame_def['position'])
        rpy_deg = frame_def.get('orientation', [180, 0, 0])
        rpy_rad = np.radians(rpy_deg)

        # Create rotation from RPY
        R = sm.SE3.RPY(rpy_rad, order='xyz').R
        return sm.SE3.Rt(R=R, t=pos, check=False)

    def _make_object_approach(self, frame_def: Dict, obj_name: str) -> sm.SE3:
        """
        Create approach frame above object using GRASP_CONFIG.

        Args:
            frame_def: Frame definition
            obj_name: Object name

        Returns:
            SE3 approach frame
        """
        grasp_frame = self._make_object_grasp(frame_def, obj_name)
        params = self._get_grasp_params(obj_name)

        if params['side_grasp']:
            # Side grasp: approach along gripper Z axis
            approach_frame = grasp_frame * sm.SE3.Tz(-APPROACH_HEIGHT)
        else:
            # Top grasp: approach from above (world Z)
            approach_pos = grasp_frame.t + np.array([0, 0, APPROACH_HEIGHT])
            approach_frame = sm.SE3.Rt(R=grasp_frame.R, t=approach_pos, check=False)

        return approach_frame

    def _make_object_grasp(self, frame_def: Dict, obj_name: str) -> sm.SE3:
        """
        Create grasp frame at object using GRASP_CONFIG.

        Args:
            frame_def: Frame definition
            obj_name: Object name

        Returns:
            SE3 grasp frame
        """
        # Get object frame from MuJoCo
        target_obj = frame_def.get('object', obj_name)
        obj_frame = get_mjobj_frame(self.m, self.d, target_obj)

        # Apply grasp configuration
        params = self._get_grasp_params(target_obj)

        grasp_pose = obj_frame * sm.SE3.Tz(params['grasp_tz'])
        grasp_frame = (grasp_pose *
                       sm.SE3.Rz(params['grasp_rz']) *
                       sm.SE3.Rx(-params['grasp_rx']) *
                       sm.SE3.Tz(params['grasp_td']))

        return grasp_frame

    def _make_object_lift(self, frame_def: Dict, obj_name: str) -> sm.SE3:
        """
        Create lift frame after grasping.

        Args:
            frame_def: Frame definition
            obj_name: Object name

        Returns:
            SE3 lift frame
        """
        grasp_frame = self._make_object_grasp(frame_def, obj_name)
        params = self._get_grasp_params(obj_name)

        if params['side_grasp']:
            # Side grasp: lift vertically in world frame
            lift_pos = grasp_frame.t + np.array([0, 0, APPROACH_HEIGHT])
            lift_frame = sm.SE3.Rt(R=grasp_frame.R, t=lift_pos, check=False)
        else:
            # Top grasp: same as approach
            lift_pos = grasp_frame.t + np.array([0, 0, APPROACH_HEIGHT])
            lift_frame = sm.SE3.Rt(R=grasp_frame.R, t=lift_pos, check=False)

        return lift_frame

    def _make_drop_approach(self, frame_def: Dict, obj_name: str) -> sm.SE3:
        """
        Create approach frame above drop point.

        Args:
            frame_def: Frame definition
            obj_name: Object name

        Returns:
            SE3 drop approach frame
        """
        drop_frame = self._make_drop_place(frame_def, obj_name)

        # Always approach from above for drop
        approach_height = frame_def.get('approach_height', APPROACH_HEIGHT)
        approach_pos = drop_frame.t + np.array([0, 0, approach_height])

        return sm.SE3.Rt(R=drop_frame.R, t=approach_pos, check=False)

    def _make_drop_place(self, frame_def: Dict, obj_name: str) -> sm.SE3:
        """
        Create drop/place frame at drop point.

        Args:
            frame_def: Frame definition
            obj_name: Object name

        Returns:
            SE3 drop frame
        """
        # Get drop point frame from MuJoCo
        drop_point_name = frame_def.get('drop_point', f'drop_point_{obj_name}')
        drop_base_frame = get_mjobj_frame(self.m, self.d, drop_point_name)

        # Apply same grasp configuration as pick (for consistent gripper orientation)
        params = self._get_grasp_params(obj_name)

        drop_pose = drop_base_frame * sm.SE3.Tz(params['grasp_tz'])
        drop_frame = (drop_pose *
                      sm.SE3.Rz(params['grasp_rz']) *
                      sm.SE3.Rx(-params['grasp_rx']) *
                      sm.SE3.Tz(params['grasp_td']))

        return drop_frame

    def _get_grasp_params(self, obj_name: str) -> Dict[str, Any]:
        """
        Get grasp parameters for object from GRASP_CONFIG.

        Args:
            obj_name: Object name

        Returns:
            Dict with grasp parameters
        """
        grasp_config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)

        return {
            'side_grasp': grasp_config.get('side_grasp', False),
            'grasp_rx': grasp_config.get('rx', np.pi),
            'grasp_rz': grasp_config.get('rz', 0),
            'grasp_tz': grasp_config.get('tz', 0),
            'grasp_td': grasp_config.get('td', 0),
        }


def compute_fk_positions(robot_rtb, waypoints: List[np.ndarray]) -> List[Dict]:
    """
    Utility function: Compute FK for joint waypoints.

    Useful for converting existing joint waypoints to Cartesian positions.

    Args:
        robot_rtb: roboticstoolbox robot
        waypoints: List of joint configurations

    Returns:
        List of dicts with 'position' and 'rpy_deg'
    """
    results = []

    for i, q in enumerate(waypoints):
        T = robot_rtb.fkine(q)
        pos = T.t
        rpy = T.rpy(order='xyz', unit='deg')

        results.append({
            'waypoint_idx': i,
            'position': [float(pos[0]), float(pos[1]), float(pos[2])],
            'rpy_deg': [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        })

    return results
