import mujoco as mj
from spatialmath import SE3, SO3
from spatialmath.base import trnorm
from scipy.spatial.transform import Rotation
import math
import random
import mujoco

from cam import *


def r2q(rot):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]
    using scipy's Rotation class
    """
    r = Rotation.from_matrix(rot)
    return r.as_quat()  # Returns [x, y, z, w]

def program(d, m):
    # Computer vision
    # _width = 640*3,
    # _height = 480*3,
    # Initialize OpenGL context
    # mj.GLContext(max_width=640, max_height=480)
    # Create renderer

    camera_name = "cam1"

    # Random duck position
    rand_x = random.uniform(0.45,0.75)
    rand_y = random.uniform(-0.55,0.55)
    z = 0.025
    d.joint('duck').qpos [0:3]= [rand_x, rand_y, z+0.01]
    rand_rot = random.randint(0,359)
    rot = SO3.Eul(rand_rot, 90, 90,unit="deg").R
    d.joint('duck').qpos[3:] = r2q(rot)

    duck_pos = d.body('duck').xpos
    duck_rot = d.body('duck').xmat.reshape(3, 3)
    duck_rot = trnorm(duck_rot)
    duck_se3 = SE3.Rt(duck_rot, duck_pos)

    mujoco.mj_step(m, d)

    cam_se3_2 = get_camera_pose_cv(m, d, camera_name=camera_name)

    gt = cam_se3_2.inv() * duck_se3

    id = 0

    with open(f"gt_{id:04}.txt", 'w') as f:
        for i in range(4):
            for j in range(4):
                f.write(f"{gt.A[i,j]} ")
            f.write("\n")
    
    renderer = mj.Renderer(m, height=480, width=640)
    get_pointcloud(m, d, renderer, f"point_cloud_{id:04}.pcd", camera_name=camera_name)
    show_pointcloud(f"point_cloud_{id:04}.pcd")
