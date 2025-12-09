import spatialmath as sm
import mujoco as mj
import roboticstoolbox as rtb
import numpy as np
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
                # print(f"  Contact {i}: {geom1_name} <-> {geom2_name}", "Robot in collision!")
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
        last_q = self.queue[-1][0]
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
        for step in rtb.jtraj(q0=start_q, qf=end_q, t=t).s: # .s to get position information
            self.queue.append((step, self.gripper_value))

    def move_j_via(self, points, t=100):
        # a list of q poses -> a joint space interpolaton
        for i in range(1, len(points)):
            self.move_j(points[i-1], points[i], t=t)

    