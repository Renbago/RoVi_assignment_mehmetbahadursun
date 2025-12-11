"""
Simple tool to manually teach the robot waypoints (P2P).
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import mujoco
import mujoco.viewer
import numpy as np
import os
import sys

def main():
    """
    Main purpose of the code is for just saving the
    points and collecting the positions for p2p task
    if u cannot close with ctrl+c 
    just use:

    ```bash
    pkill -9 -f python
    ```
    """
    model_path = "scene_obstacles_lecture_6.xml"

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    print("Double click and ctrl + right to drag.")

    m.opt.gravity = np.array([0, 0, 0])

    # set ur5 start position
    start_pose = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    d.qpos[:6] = np.array(start_pose)
    d.ctrl[:6] = np.array(start_pose)

    # disable motors
    if m.nu > 0:
        m.actuator_gainprm[:, 0] = 0  # stiffness = 0
        m.actuator_biasprm[:, 1] = 0  # bias = 0
    
    mujoco.mj_forward(m, d)
    mujoco.viewer.launch(m, d)

if __name__ == "__main__":
    main()
