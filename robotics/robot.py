"""
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import mujoco as mj
import spatialmath as sm
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

import sys
import os

# the helpers path directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from helpers.trapezoidal_profile import TrapezoidalProfile

from spatialmath.base import trinterp, trnorm


def ur_ctrl_qpos(data, q_desired):
     assert len(q_desired) == 6, "Expected 6 joint positions for UR robot"
     for i in range(len(q_desired)):
            data.ctrl[i] = q_desired[i]  # Assumes actuators are position-controlled
    
def ur_set_qpos(data, q_desired):
     """
     Set the desired joint position for the ur robot arm
     """
     assert len(q_desired) == 6, "Expected 6 joint positions for UR robot"
     for i in range(len(q_desired)):
            # Forcing the joint values to be our desired values
            data.qpos[i] = q_desired[i]
            # Remember to also set the control values
            # otherwise the robot will just move back to the original position
            data.ctrl[i] = q_desired[i] 

def hande_ctrl_qpos(data, gripper_value:int=0):
    data.ctrl[6] = gripper_value

def get_mjobj_frame(model, data, obj_name):
    """
    Get the frame of a specific object in the MuJoCo simulation.
    """
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError(f"Object '{obj_name}' not found")
    # Get the object's position and orientation
    obj_pos = data.xpos[obj_id]
    obj_rot = data.xmat[obj_id]
    return _make_tf(R=obj_rot.reshape(3,3), t=obj_pos)

def _make_tf(R, t):
    """
        Combine translation and orientation
    """
    # TODO: add checks for dimensions
    return sm.SE3.Rt(R=R, t=t, check=False)

def ur_get_qpos(data, model):
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
    joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in UR_JOINT_NAMES]
    # Get qpos indices for the joints
    # (since qpos is a flat array, we need to know where each joint's value is stored)
    qpos_indices = []
    for jid in joint_ids:
        # Each joint's position is stored at data.qpos[model.jnt_qposadr[jid]]
        qpos_indices.append(model.jnt_qposadr[jid])
    q_values = data.qpos[qpos_indices]
    return q_values

def is_q_valid(d, m, q):
    
    UR_JOINT_NAMES = [
        "shoulder_collision",
        "upper_arm_link_1_collision",
        "upper_arm_link_2_collision",
        "forearm_link_1_collision",
        "forearm_link_2_collision",
        "wrist_1_joint_collision",
        "wrist_2_link_1_collision",
        "wrist_2_link_2_collision",
        "eef_geom",
        "base_mount_collision",
        "base_collision",
        "right_driver_collision",
        "right_coupler_collision",
        "right_spring_link_collision",
        "right_follower_collision",
        "left_driver_collision",
        "left_coupler_collision",
        "left_spring_link_collision",
        "left_follower_collision",
    ]
    q0 = ur_get_qpos(d,m)
    # Set robot q
    ur_set_qpos(d, q)
    mj.mj_forward(m, d)
    
    # Check if there is any collisions
    # print("testing q: ", q)
    if d.ncon > 0:
        # print(f"Collisions detected: {d.ncon}")
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1_name = m.geom(contact.geom1).name
            geom2_name = m.geom(contact.geom2).name            
            # Filter out collision that isn't about the robot
            if geom1_name in UR_JOINT_NAMES or geom2_name in UR_JOINT_NAMES:
                # the problem is the robot does see the boxes 
                # which we grasped as a collision, so we need to cancel it
                # sorry not the robot RRT so its cannot create a path 
                # that is why we need to cancel the collision
                graspables = ["box", "cylinder", "t_block_pt1", "t_block_pt2"]
                if (geom1_name in graspables or geom2_name in graspables):
                    continue

                # Return pose back to original pose
                ur_set_qpos(d, q0)
                mj.mj_forward(m, d)
                return False
    # Return pose back to original pose
    ur_set_qpos(d, q0)
    mj.mj_forward(m, d)
    return True

class UR5robot():
    def __init__(self, data: mj.MjData, model:mj.MjModel,):
        self.name = "UR5"
        self.TOOL_LENGTH = 0.15
        # TODO: Implement DH robot for an UR5e using RTB
        self.robot_ur5 = rtb.DHRobot([
                rtb.RevoluteDH(d=0.1625, alpha=np.pi / 2.0, qlim=(-np.pi, np.pi)),              # J1
                rtb.RevoluteDH(a=-0.425, qlim=(-np.pi, np.pi)),                                 # J2
                rtb.RevoluteDH(a=-0.3922, qlim=(-np.pi, np.pi)),                                # J3
                rtb.RevoluteDH(d=0.1333, alpha=np.pi / 2.0, qlim=(-np.pi, np.pi)),              # J4
                rtb.RevoluteDH(d=0.0997, alpha=-np.pi / 2.0, qlim=(-np.pi, np.pi)),             # J5
                rtb.RevoluteDH(d=0.0996 + self.TOOL_LENGTH, qlim=(-np.pi, np.pi)),                   # J6
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
        t: number of steps (duration = t * 0.002)
        """
        dt = 0.002
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
    
    def clear_trajectories(self):
        """Clear stored trajectories"""
        self.planned_trajectories = []