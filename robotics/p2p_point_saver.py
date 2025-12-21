"""
Main purpose of the code is for just saving the
points and collecting the positions for p2p task
if u cannot close with ctrl+c 
just use:

```bash
pkill -9 -f python
```

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import mujoco
import mujoco.viewer
import numpy as np
from robot import UR5robot
from utils.mujoco_utils import get_mjobj_frame
from utils import load_config, get_objects_to_move, get_model_path
from scripts.manipulation import _init_grasp_params, _compute_grasp_frame
import spatialmath as sm


def main():
    # Load configuration
    config = load_config()
    model_path = get_model_path(config)
    objects_to_move = get_objects_to_move(config)

    # Let user select which object to record waypoints for
    print("Available objects:")
    for i, obj in enumerate(objects_to_move, 1):
        print(f"  {i}. {obj}")

    obj_choice = input(f"Select object (1-{len(objects_to_move)}) or name [{objects_to_move[0]}]: ").strip()

    if obj_choice.isdigit() and 1 <= int(obj_choice) <= len(objects_to_move):
        obj_name = objects_to_move[int(obj_choice) - 1]
    elif obj_choice in objects_to_move:
        obj_name = obj_choice
    else:
        obj_name = objects_to_move[0]

    print(f"\nRecording waypoints for: {obj_name}")
    print("Drag the robot to desired positions and use 'Copy state' and copy those at .txt folder")
    print("pkill -9 -f python")
    print("\nThen paste those values to config.yaml under p2p_frames.\n")

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    m.opt.gravity = np.array([0, 0, 0])

    # Disable motors for free dragging
    if m.nu > 0:
        m.actuator_gainprm[:, 0] = 0
        m.actuator_biasprm[:, 1] = 0

    mujoco.mj_forward(m, d)

    # Start at grasp position for selected object -
    robot = UR5robot(d, m)
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name=obj_name)
    params = _init_grasp_params(obj_name)
    grasp_frame = _compute_grasp_frame(obj_frame, params)
    q_grasp = robot.robot_ur5.ik_LM(Tep=grasp_frame, q0=UR5robot.Q_HOME)[0]

    # Start at grasp position
    d.qpos[:6] = q_grasp
    d.ctrl[:6] = q_grasp
    d.ctrl[6] = 255  # gripper closing
    mujoco.mj_forward(m, d)

    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
