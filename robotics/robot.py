"""
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import mujoco as mj
import spatialmath as sm
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from spatialmath.base import trinterp, trnorm
from scripts.trapezoidal_profile import TrapezoidalProfile
from utils.mujoco_utils import (
    ur_ctrl_qpos,
    ur_set_qpos,
    hande_ctrl_qpos,
    get_mjobj_frame,
    ur_get_qpos,
    is_q_valid,
    Q_HOME,
)


class UR5robot():
    # Class-level constants
    TOOL_LENGTH = 0.15
    NUM_JOINTS = 6
    TIME_STEP = 0.002
    Q_HOME = Q_HOME  # Import from utils (single source of truth)

    def __init__(self, data: mj.MjData, model: mj.MjModel):
        self.name = "UR5"
        # UR5e DH parameters using RTB
        self.robot_ur5 = rtb.DHRobot([
                rtb.RevoluteDH(d=0.1625, alpha=np.pi / 2.0, qlim=(-np.pi, np.pi)),              # J1
                rtb.RevoluteDH(a=-0.425, qlim=(-np.pi, np.pi)),                                 # J2
                rtb.RevoluteDH(a=-0.3922, qlim=(-np.pi, np.pi)),                                # J3
                rtb.RevoluteDH(d=0.1333, alpha=np.pi / 2.0, qlim=(-np.pi, np.pi)),              # J4
                rtb.RevoluteDH(d=0.0997, alpha=-np.pi / 2.0, qlim=(-np.pi, np.pi)),             # J5
                rtb.RevoluteDH(d=0.0996 + UR5robot.TOOL_LENGTH, qlim=(-np.pi, np.pi)),          # J6
                ], name="UR5", base=sm.SE3.Rz(-np.pi))
        self.d = data 
        self.m = model
        self.queue = []
        self.gripper_value = 0

        # Visualization params
        self.planned_trajectories = []

    def get_current_tcp(self):
        q0 = self.get_current_q()
        tcp_frame = self.robot_ur5.fkine(q0)
        return tcp_frame

    def get_current_q(self):
        # Define the joint names (adjust based on your UR model)
        UR_JOINT_NAMES = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # Get joint IDs
        joint_ids = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, name) for name in UR_JOINT_NAMES]
        # Get qpos indices for the joints
        # (since qpos is a flat array, we need to know where each joint's value is stored)
        qpos_indices = []
        for jid in joint_ids:
            # Each joint's position is stored at data.qpos[model.jnt_qposadr[jid]]
            qpos_indices.append(self.m.jnt_qposadr[jid])
        current_q = self.d.qpos[qpos_indices]
        return current_q
    
    def set_gripper(self, value, t=100):
        # Get last position from queue, or current position if queue is empty
        if len(self.queue) > 0:
            last_q = self.queue[-1][0]
        else:
            last_q = self.get_current_q()
        for _ in range(t):
            self.queue.append((last_q, value))
        self.gripper_value = value

    def move_l(self, T0: sm.SE3, T1: sm.SE3, q0, total_time: int, time_step=0.002) -> list[sm.SE3]:
        steps = int(total_time / time_step)
        qinit = q0
        for cpose in rtb.ctraj(T0=T0, T1=T1, t=steps): # .s to get position information
            q_step = self.robot_ur5.ik_LM(Tep=cpose, q0=qinit)
            if q_step[1]:
                self.queue.append((q_step[0], None))
                qinit = q_step[0]

    def move_j(self, start_q, end_q, t=100):
        """
        Move joints using Trapezoidal Velocity Profile.
        t: number of steps (duration = t * TIME_STEP)
        """
        dt = UR5robot.TIME_STEP
        t_f = t * dt
        
        # Use Unified Trapezoidal Profile (handles arrays natively)
        profile = TrapezoidalProfile(q_i=start_q, q_f=end_q, t_f=t_f)
        _, positions, _, _ = profile.sample_trajectory(dt=dt)
        
        for p in positions:
            self.queue.append((p, self.gripper_value))

    def move_j_via(self, points, t=100):
        # a list of q poses -> a joint space interpolaton
        for i in range(1, len(points)):
            self.move_j(points[i-1], points[i], t=t)

    """
    Visualization functions:
    """

    def add_planned_path(self, trajectory):
        """Store planned path"""
        self.planned_trajectories.extend(trajectory)
    
    def get_ee_positions(self, trajectory=None, use_queue=False, sample_rate=10):
        """
        Compute end-effector positions.
        trajectory: optional list of q (joint) arrays. If provided, calculate EE for this list.
        use_queue=True: use all interpolated queue positions (denser) from ROBOT memory
        sample_rate: only use every Nth point (for performance)
        """
        ee_positions = []
        
        # 1. Use EXTERNAL Trajectory if provided
        if trajectory is not None:
             for i, q in enumerate(trajectory):
                if i % sample_rate == 0:
                     T = self.robot_ur5.fkine(q)
                     ee_positions.append(T.t.copy())
                     
        # 2. Use INTERNAL Queue (Dense)
        elif use_queue and len(self.queue) > 0:
            # Use dense queue (all interpolated steps)
            for i, (q, _) in enumerate(self.queue):
                if i % sample_rate == 0:  # Sample every Nth point
                    T = self.robot_ur5.fkine(q)
                    ee_positions.append(T.t.copy())
                    
        # 3. Use INTERNAL Planned Trajectories (Sparse)
        else:
            # Use sparse planned trajectories (RRT waypoints)
            for q in self.planned_trajectories:
                T = self.robot_ur5.fkine(q)
                ee_positions.append(T.t.copy())
        
        return ee_positions
    
    def visualize_trajectory(self, viewer, use_queue=True, sample_rate=10):
        """
        Visualize trajectory in viewer with spheres and connecting lines.
        use_queue=True: show all interpolated steps (dense)
        sample_rate: show every Nth point (for performance)
        """
        ee_positions = self.get_ee_positions(use_queue=use_queue, sample_rate=sample_rate)
        
        if len(ee_positions) == 0:
            print("No trajectory to visualize")
            # Fallback: check if we have planned sparse waypoints even if queue is empty or not requested
            if len(self.planned_trajectories) == 0:
                return

        scene = viewer.user_scn
        
        # 1. VISUALIZE DENSE TRACE (Small Spheres)
        if use_queue and len(ee_positions) > 0:
            num_points = len(ee_positions)
            for idx, pos in enumerate(ee_positions):
                t = idx / max(num_points - 1, 1)
                color = [t, 0.2, 1.0 - t, 0.5]  # Blue -> Red, semi-transparent
                
                mj.mjv_initGeom(scene.geoms[scene.ngeom], type=mj.mjtGeom.mjGEOM_SPHERE,
                               size=[0.005, 0, 0], pos=pos, mat=np.eye(3).flatten(),
                               rgba=np.array(color, dtype=np.float32))
                scene.ngeom += 1

        # 2. VISUALIZE PLANNED WAYPOINTS (Large Spheres - "Main Stopping Points")
        # These are the key nodes from RRT/PRM
        if len(self.planned_trajectories) > 0:
            for i, q_wp in enumerate(self.planned_trajectories):
                # Calculate Forward Kinematics for this waypoint
                T = self.robot_ur5.fkine(q_wp)
                pos = T.t
                
                # Colors: Green for Start, Red for Goal, Yellow for Intermediate
                if i == 0:
                    color = [0, 1, 0, 1] # Start: Green
                    size = 0.03
                elif i == len(self.planned_trajectories) - 1:
                    color = [1, 0, 0, 1] # Goal: Red
                    size = 0.03
                else:
                    color = [1, 1, 0, 1] # Intermediate: Yellow
                    size = 0.02
                
                mj.mjv_initGeom(scene.geoms[scene.ngeom], type=mj.mjtGeom.mjGEOM_SPHERE,
                               size=[size, 0, 0], pos=pos, mat=np.eye(3).flatten(),
                               rgba=np.array(color, dtype=np.float32))
                scene.ngeom += 1
        
        print(f"Visualization updated: {len(ee_positions)} trace points, {len(self.planned_trajectories)} waypoints")

    def save_trajectory(self, filename="trajectory.npy"):
        """Save queue trajectory to file for later replay"""
        import json
        data = {
            "queue": [(q.tolist(), g) for q, g in self.queue],
            "planned": [q.tolist() for q in self.planned_trajectories]
        }
        np.save(filename, data, allow_pickle=True)
        print(f"Trajectory saved to {filename}")
    
    def clear_trajectories(self, viewer=None):
        """Clear stored trajectories and optionally clear viewer visualization"""
        self.planned_trajectories = []
        if viewer is not None:
            viewer.user_scn.ngeom = 0