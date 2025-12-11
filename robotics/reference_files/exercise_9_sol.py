
import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random

from ompl import base as ob
from ompl import geometric as og

from robot import *

import numpy as np
import roboticstoolbox as rtb
from scipy.optimize import minimize
from spatialmath import SE3, SO3
import mujoco

def random_box_pos(m, d):
    selected_pos = []
    for i in range(1, 7):
        STOP = False
        while not STOP:
            rand_x = random.uniform(0.45+0.25,0.75-0.25)
            rand_y = random.uniform(-0.55+0.25,0.55-0.25)
            if len(selected_pos) == 0:
                STOP = True
                break

            SHOULD_STOP = True   
            for pos in selected_pos:
                dist = np.linalg.norm(np.array([rand_x, rand_y]) - np.array(pos))
                if dist <= 0.15:
                    SHOULD_STOP = False
                    break
            
            if SHOULD_STOP:
                STOP = True

        z = 0.025
        d.joint(f'box{i}').qpos [0:3]= [rand_x, rand_y, z+0.01]
        print(f'box{i}: ', rand_x, rand_y)
        selected_pos.append([rand_x, rand_y])

    mujoco.mj_step(m, d)

    



def program(d, m):

    # Randomize box positions
    random_box_pos(m, d)

    # Define our robot object
    robot = UR5robot(data=d, model=m)

    drop_point_box1_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_box2") * sm.SE3.Rx(-np.pi)  # Get body frame

    pickzone_init_guess = [0.264, -1.17, 1.18, -1.58, -1.57, -1.31]
    robot.move_j(start_q=robot.get_current_q(), end_q=pickzone_init_guess, t=100)

    EXE_TIME = 500 #ms

    for box_name, y_offset, z_offset  in [("box1", -0.08, 0), ("box2", 0, 0), ("box3", 0.08, 0), ("box4", -0.04, -0.05), ("box5", 0.04, -0.05), ("box6", 0, -0.1)]:
        # # Define grasping frames for object: box
        box_frame = get_mjobj_frame(model=m, data=d, obj_name=box_name) * sm.SE3.Rx(-np.pi)  # Get body frame

        end_q = robot.robot_ur5.ik_NR(Tep=box_frame * sm.SE3.Tz(-0.15), q0=robot.queue[-1][0], slimit=1000)[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=end_q, t=EXE_TIME)

        end_q = robot.robot_ur5.ik_NR(Tep=box_frame, q0=robot.queue[-1][0], slimit=1000)[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=end_q, t=EXE_TIME)

        robot.set_gripper(255) # Close gripper

        end_q = robot.robot_ur5.ik_NR(Tep=box_frame * sm.SE3.Tz(-0.15), q0=robot.queue[-1][0], slimit=1000)[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=end_q, t=EXE_TIME)

        # # A good position: 
        dropzone_init_guess = [2.92, -1.35, 1.42, -1.64, -1.57, 1.35]
        robot.move_j(start_q=robot.queue[-1][0], end_q=dropzone_init_guess, t=int(EXE_TIME * 1.5))

        end_q = robot.robot_ur5.ik_NR(Tep=drop_point_box1_frame * sm.SE3.Tz(0.25+z_offset) * sm.SE3.Ty(y_offset), q0=robot.queue[-1][0], slimit=1000)[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=end_q, t=EXE_TIME)

        robot.set_gripper(0) # open gripper

        # Go up again
        robot.move_j(start_q=robot.queue[-1][0], end_q=dropzone_init_guess, t=EXE_TIME)

        # Reset
        robot.move_j(start_q=robot.queue[-1][0], end_q=pickzone_init_guess, t=EXE_TIME)

    return robot.queue

    
    

       
         
