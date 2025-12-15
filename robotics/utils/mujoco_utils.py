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
from utils.logger import ProjectLogger
from spatialmath import SO3

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

# Gripper geometry names (subset of UR_COLLISION_GEOMS)
# These are the ONLY robot parts that should touch held objects
GRIPPER_GEOMS = [
    "right_driver_collision",
    "right_coupler_collision",
    "right_spring_link_collision",
    "right_follower_collision",
    "left_driver_collision",
    "left_coupler_collision",
    "left_spring_link_collision",
    "left_follower_collision",
    "right_pad",
    "left_pad",
    "eef_geom",  # End effector
]

# Gripper self-collision pairs to IGNORE (these are expected/normal contacts)
# When gripper is closed, pads touch each other - this is NOT a real collision
IGNORED_COLLISION_PAIRS = {
    ("right_pad", "left_pad"),
    ("left_pad", "right_pad"),
}

# Stores the REAL gripper-to-object transform measured after physics grasp.
# Key: object name, Value: SE3 transform from gripper to object center
RUNTIME_GRASP_OFFSET = {}

"""
    Grasp Configuration:

    rx: rotation around X axis (π = top-down, π/2 = side grasp)
    rz: rotation around Z axis (for side grasp orientation)
    
    tz: offset from object center where gripper aims (used for both grasp and collision checking)
    td: How far gripper goes into object along approach direction (gripper local +Z)

    side_grasp: True = horizontal approach, False = vertical approach
    IMPORTANT: tz is now used for BOTH grasp positioning and collision checking.

    example:
    tz=0.03: Gripper aims 3cm above cylinder center
    td=0.02: Gripper goes 2cm deeper into cylinder for better grip

"""

GRASP_CONFIG = {
    "box": {"rx": np.pi, "tz": 0.03, "td": 0.03, "side_grasp": False},
    "cylinder": {"rx": np.pi / 2, "rz": -np.pi / 2, "tz": 0.0, "td": 0.02, "side_grasp": True},
    "t_block": {"rx": np.pi,"rz": -np.pi / 2, "tz": 0.02, "td": 0.03, "side_grasp": False},
}

DEFAULT_GRASP_CONFIG = {"rx": np.pi, "rz": 0, "tz": 0, "td": 0, "side_grasp": False}

def get_grasp_offset(obj_name):
    """
    Use tz directly - it tells us where object center is relative to gripper TCP
    tz = 0 means gripper at center, tz > 0 means gripper above center
    """
    config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)

    tz = config.get("tz", 0)

    # Object center is tz distance in +Z direction from gripper TCP
    return np.array([0, 0, tz])

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

            # # Skip gripper self-collision (pads touching when closed)
            # if (geom1_name, geom2_name) in IGNORED_COLLISION_PAIRS:
            #     continue

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


def compute_real_grasp_offset(d, m, robot_rtb, obj_name):
    """
    We are calling after grasp, the thing is we will be giving this
    informatino to rtb for calculation the obstacle collision while 
    IK is starting the path planning. so the obstacle will be calculated
    as a new joint with area of the object. and will be not collide with
    other objects check the get_grasp_transform function.

    Reference: https://petercorke.github.io/robotics-toolbox-python/arm_erobot.html#tool
    """
    logger = ProjectLogger.get_instance()

    # Grippers current position
    q = ur_get_qpos(d, m)
    gripper_frame = robot_rtb.fkine(q)

    # Object's current real position 
    obj_frame = get_mjobj_frame(m, d, obj_name)

    # We are finding the new TF with new obj_frame 
    real_offset = gripper_frame.inv() * obj_frame

    # Store the new runtime offset for calculation 
    RUNTIME_GRASP_OFFSET[obj_name] = real_offset

    # Log the grasp offset
    logger.log_grasp_offset(obj_name, real_offset.t, "runtime")
    logger.debug(f"  Gripper: [{gripper_frame.t[0]}, {gripper_frame.t[1]}, {gripper_frame.t[2]}]")
    logger.debug(f"  Object:  [{obj_frame.t[0]}, {obj_frame.t[1]}, {obj_frame.t[2]}]")

    return real_offset


def clear_grasp_offset(obj_name):
    """
    Call after release. Clears stored grasp offset.
    """

    if obj_name in RUNTIME_GRASP_OFFSET:
        del RUNTIME_GRASP_OFFSET[obj_name]
        ProjectLogger.get_instance().debug(f"Grasp offset cleared for {obj_name}")


def get_grasp_transform(obj_name):
    """
    Calculate the transform from gripper to object frame.
    The purpose is calculating for collision control.
    Like checking i have the gripper knowledge but where is the obstacle ?

    Math explanation:
    - Grasp frame is computed as: 
        Grasp = Obj * Tz(tz) * Rz(rz) * Rx(-rx) * Tz(td)
    - To get Object from Grasp: 
        Obj = Grasp * (Tz(td) * Rx * Rz * Tz(tz))^-1
    - The inverse is: 
        Tz(-td) * Rx(rx) * Rz(-rz) * Tz(-tz)

    This transform can be set as robot.tool in RTB so that fkine()
    returns the OBJECT CENTER pose directly instead of gripper TCP.

    Important thing the td must be included, without it FK will compute wrong
    object position during held_object collision check (2cm error for cylinder).

    From author:
        "This function is kinda cool and necessary for collision control :P"

    Reference: https://petercorke.github.io/robotics-toolbox-python/arm_erobot.html#tool
    """
    config = GRASP_CONFIG.get(obj_name, DEFAULT_GRASP_CONFIG)
    rx = config.get("rx", np.pi)
    rz = config.get("rz", 0)
    tz = config.get("tz", 0)
    td = config.get("td", 0)

    T_rel = sm.SE3.Tz(-td) * sm.SE3.Rx(rx) * sm.SE3.Rz(-rz) * sm.SE3.Tz(-tz)
    return T_rel


def is_q_valid_with_held_object(d, m, q, robot_rtb, held_object, target_object=None):
    """
    Check if joint configuration is collision-free INCLUDING the held object.

    Uses Roboticstoolbox 'tool' property to calculate object pose automatically.
    This correctly computes where the held object would be in world coordinates
    based on the grasp transform (how the object was picked up).

    Args:
        d: MuJoCo data
        m: MuJoCo model
        q: Joint configuration to test
        robot_rtb: roboticstoolbox robot (for FK to compute object position)
        held_object: Name of object being held ("box", "cylinder", "t_block")
        target_object: Object we're approaching (collisions ignored)

    Returns:
        True if configuration is collision-free, False otherwise
    """
    from spatialmath import UnitQuaternion

    # 1. Save current state and robot tool
    q0 = ur_get_qpos(d, m)
    original_tool = robot_rtb.tool

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

    # 4. Calculate where object would be at this robot config
    gripper_frame = robot_rtb.fkine(q)

    if held_object in RUNTIME_GRASP_OFFSET:
        # Use measured real offset
        obj_frame = gripper_frame * RUNTIME_GRASP_OFFSET[held_object]
    else:
        # Use static config (fallback)
        obj_frame = gripper_frame * get_grasp_transform(held_object)

    # Extract position and orientation
    obj_pos = obj_frame.t

    # Convert rotation to quaternion - use SO3 for robustness
    obj_quat = UnitQuaternion(SO3(obj_frame.R, check=False)).A  # [w, x, y, z]


    # Move object to computed position
    # Free joint için qpos layout:
    # [0:3] = pozisyon (x, y, z)
    # [3:7] = quaternion (w, x, y, z)
    # Reference: https://mujoco.readthedocs.io/en/stable/modeling.html#floating-objects
    d.qpos[held_obj_qpos_adr:held_obj_qpos_adr + 3] = obj_pos
    d.qpos[held_obj_qpos_adr + 3:held_obj_qpos_adr + 7] = obj_quat

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

            # # Skip gripper self-collision (pads touching when closed)
            # if (geom1_name, geom2_name) in IGNORED_COLLISION_PAIRS:
            #     continue

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
                collision = True
                break

    # 7. Restore original state
    ur_set_qpos(d, q0)
    d.qpos[held_obj_qpos_adr:held_obj_qpos_adr + 7] = held_obj_qpos_original
    mj.mj_forward(m, d)

    return not collision


