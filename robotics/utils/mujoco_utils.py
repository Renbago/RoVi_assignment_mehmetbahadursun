"""
Utility functions for mujoco and configuration loading.

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import mujoco as mj
import spatialmath as sm
import yaml
import numpy as np
import os

# ============================================================
# Shared Robot Constants
# ============================================================
Q_HOME = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])

# Robot collision geometry names
UR_COLLISION_GEOMS = [
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
    # Gripper pads (contact surfaces)
    "right_pad",
    "left_pad",
]

# Object name to geom name mapping
OBJECT_GEOMS = {
    "box": ["box"],
    "cylinder": ["cylinder"],
    "t_block": ["t_block_pt1", "t_block_pt2"],
}

# Grasp offsets: Object center position relative to gripper frame (local coordinates)
# When holding an object, its center is NOT at the gripper position
# These offsets define where the object center is in the gripper's local frame
GRASP_OFFSETS = {
    "box": np.array([0, 0, -0.03]),       # Top-down grasp: center 3cm below gripper (Z-)
    "cylinder": np.array([0, 0, -0.05]),  # Side grasp: center 5cm in gripper's pointing direction (local Z-)
    "t_block": np.array([0, 0, -0.02]),   # Top-down grasp: center 2cm below gripper
}

# ============================================================
# Config Loading Functions
# ============================================================
def load_config(path=None):
    """
    Load configuration from YAML file.
    Default path is config.yaml in the robotics directory.
    """
    if path is None:
        # Get the robotics directory (parent of utils)
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        robotics_dir = os.path.dirname(utils_dir)
        path = os.path.join(robotics_dir, "config.yaml")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_p2p_waypoints(config, obj_name):
    """
    Get P2P waypoints for a specific object as numpy arrays.
    """
    waypoints = config.get('p2p_waypoints', {}).get(obj_name, [])
    return [np.array(wp) for wp in waypoints]


def get_objects_to_move(config):
    """
    Get list of objects to move from config.
    """
    return config.get('scene', {}).get('objects_to_move', [])


def get_model_path(config):
    """
    Get model path from config.
    """
    return config.get('scene', {}).get('model_path', 'scene_obstacles_lecture_6.xml')


def get_default_planner(config):
    """Get default planner type from config."""
    return config.get('planner', {}).get('default_type', 'rrt')


def get_ik_mode(config):
    """Get IK mode from config: 'simple' or 'goal_region'."""
    return config.get('planner', {}).get('ik_mode', 'simple')

# ============================================================
# mujoco Functions
# ============================================================
def ur_ctrl_qpos(data, q_desired):
    """
    Control commands
    """
    assert len(q_desired) == 6, "6 joint positions"
    for i in range(len(q_desired)):
        data.ctrl[i] = q_desired[i]


def ur_set_qpos(data, q_desired):
    """
    Starting joint positions
    """
    assert len(q_desired) == 6, "6 joint positions"
    for i in range(len(q_desired)):
        data.qpos[i] = q_desired[i]
        data.ctrl[i] = q_desired[i]


def hande_ctrl_qpos(data, gripper_value: int = 0):
    """
    Gripper position.
    """
    data.ctrl[6] = gripper_value


def ur_get_qpos(data, model):
    """
    Current joint positions.
    """
    UR_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in UR_JOINT_NAMES]
    qpos_indices = []
    for jid in joint_ids:
        qpos_indices.append(model.jnt_qposadr[jid])
    q_values = data.qpos[qpos_indices]
    return q_values


def _make_tf(R, t):
    """
    Creating TF matrix
    """
    return sm.SE3.Rt(R=R, t=t, check=False)


def get_mjobj_frame(model, data, obj_name):
    """
    Get the frame of the desired object.
    """
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
    if obj_id == -1:
        raise ValueError(f"Object '{obj_name}' not found")
    obj_pos = data.xpos[obj_id]
    obj_rot = data.xmat[obj_id]
    return _make_tf(R=obj_rot.reshape(3, 3), t=obj_pos)


def is_q_valid(d, m, q, target_object=None):
    """
    Check if joint configuration is collision-free.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        q: Joint configuration to check
        target_object: Name of object being grasped (collisions with this object
                      are ignored to allow gripper contact). If None, all
                      graspable object collisions are checked.

    IMPORTANT: Only ignore collisions with the SPECIFIC target object.
    Previously this ignored ALL graspables, causing robot to push other
    objects out of the way during path execution.
    """
    # Save current pose
    q0 = ur_get_qpos(d, m)

    # Set robot to test pose
    ur_set_qpos(d, q)
    mj.mj_forward(m, d)

    # Check collisions
    if d.ncon > 0:
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1_name = m.geom(contact.geom1).name
            geom2_name = m.geom(contact.geom2).name

            if geom1_name in UR_COLLISION_GEOMS or geom2_name in UR_COLLISION_GEOMS:
                # Only ignore collisions with the SPECIFIC target object
                if target_object is not None:
                    allowed_geoms = OBJECT_GEOMS.get(target_object, [])
                    if geom1_name in allowed_geoms or geom2_name in allowed_geoms:
                        continue

                # Collision detected - restore and return False
                ur_set_qpos(d, q0)
                mj.mj_forward(m, d)
                return False

    # No collision - restore and return True
    ur_set_qpos(d, q0)
    mj.mj_forward(m, d)
    return True


def is_q_valid_with_held_object(d, m, q, robot_rtb, held_object, target_object=None):
    """
    Check if joint configuration is collision-free INCLUDING the held object.

    When the robot is holding an object and planning a path (e.g., to drop location),
    we need to check if the HELD OBJECT would collide with obstacles along the path.

    MuJoCo's mj_forward computes collision at a single instant based on qpos.
    Problem: When we set robot joints, the held object doesn't move with the gripper.
    Solution: Move the held object to gripper position before calling mj_forward.

    Args:
        d: MuJoCo data
        m: MuJoCo model
        q: Joint configuration to test
        robot_rtb: roboticstoolbox robot (for FK to compute gripper position)
        held_object: Name of object being held ("box", "cylinder", "t_block")
        target_object: Object we're approaching (collisions ignored, e.g., drop_point)

    Returns:
        True if configuration is collision-free, False otherwise
    """
    from spatialmath import UnitQuaternion

    # 1. Save current state
    q0 = ur_get_qpos(d, m)

    # 2. Get held object's qpos address and save original position
    held_obj_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, held_object)
    if held_obj_id == -1:
        # Object not found, fall back to normal check
        return is_q_valid(d, m, q, target_object=target_object)

    held_obj_jnt_id = m.body_jntadr[held_obj_id]
    held_obj_qpos_adr = m.jnt_qposadr[held_obj_jnt_id]
    held_obj_qpos_original = d.qpos[held_obj_qpos_adr:held_obj_qpos_adr + 7].copy()

    # 3. Set robot to test configuration
    ur_set_qpos(d, q)

    # 4. Compute gripper frame using FK
    gripper_frame = robot_rtb.fkine(q)

    # 4b. Apply grasp offset - object center is NOT at gripper position
    # Offset is in gripper's local frame, need to transform to world frame
    local_offset = GRASP_OFFSETS.get(held_object, np.zeros(3))
    world_offset = gripper_frame.R @ local_offset
    obj_pos = gripper_frame.t + world_offset

    # Object orientation matches gripper orientation
    gripper_quat = UnitQuaternion(gripper_frame.R).A  # [w, x, y, z]

    # Move object to computed position (with offset)
    d.qpos[held_obj_qpos_adr:held_obj_qpos_adr + 3] = obj_pos
    d.qpos[held_obj_qpos_adr + 3:held_obj_qpos_adr + 7] = gripper_quat

    # 5. Compute forward dynamics (collision detection)
    mj.mj_forward(m, d)

    # 6. Check collisions
    collision = False
    held_geoms = OBJECT_GEOMS.get(held_object, [])
    target_geoms = OBJECT_GEOMS.get(target_object, []) if target_object else []

    if d.ncon > 0:
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1_name = m.geom(contact.geom1).name
            geom2_name = m.geom(contact.geom2).name

            robot_involved = geom1_name in UR_COLLISION_GEOMS or geom2_name in UR_COLLISION_GEOMS
            held_involved = geom1_name in held_geoms or geom2_name in held_geoms

            # Skip gripper-to-held-object contact (expected when holding)
            gripper_held_contact = (
                (geom1_name in held_geoms and geom2_name in UR_COLLISION_GEOMS) or
                (geom2_name in held_geoms and geom1_name in UR_COLLISION_GEOMS)
            )
            if gripper_held_contact:
                continue

            # Skip contact with target object (we're approaching it)
            if target_geoms:
                target_contact = geom1_name in target_geoms or geom2_name in target_geoms
                if target_contact:
                    continue

            # If robot or held object is in collision with something else
            if robot_involved or held_involved:
                # Debug: uncomment to see collisions
                # print(f"  [COLLISION] {geom1_name} <-> {geom2_name}")
                collision = True
                break

    # 7. Restore original state
    ur_set_qpos(d, q0)
    d.qpos[held_obj_qpos_adr:held_obj_qpos_adr + 7] = held_obj_qpos_original
    mj.mj_forward(m, d)

    return not collision


