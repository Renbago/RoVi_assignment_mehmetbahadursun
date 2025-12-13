import numpy as np
import spatialmath as sm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from robot import *


# Function to check if any joint position is below floor
def is_valid_configuration(robot, q):
    # Calculate forward kinematics for all joints
    Ts = robot.fkine_all(q)    
    threshold = 0.1 # Padding to account for joint sizes
    # Check each joint position's z-coordinate
    for T in Ts[1:]: # Omit base
        if T.t[2] < threshold:  # z-coordinate is below floor
            return False
    return True

def program(d, m):
    # Define our robot object
    robot = UR5robot(data=d, model=m)

    results_data = []
    vis_traj = []
    # Find the frame of an object:
    for cylinder_num in range(1, 11):
        cylinder_grasp_pose = get_mjobj_frame(model=m, data=d, obj_name=f"cylinder{cylinder_num}") * sm.SE3.Tz(0.05) # Get body frame
        success_count = 0 
        for theta in np.linspace(0, 2*np.pi, 100):
            print(f"Trying: cylinder{cylinder_num} with theta {theta}")

            for _ in range(0, 5): # Tries to get a valid q-pose
                res = robot.robot_ur5.ik_LM(cylinder_grasp_pose * sm.SE3.Rz(theta) * sm.SE3.Rx(-np.pi/2), q0=ur_get_qpos(d, m))
                if res[1]:
                    q_desired = res[0]
                    
                    # Filter solutions
                    if is_valid_configuration(robot.robot_ur5, q_desired):
                        # for _ in range(10):
                        vis_traj.append((q_desired, None)) # qpose and gripper value
                        success_count += 1
                        break
        results_data.append(( f"cylinder{cylinder_num}", success_count))
    
    print("results_data: ", results_data)


    values = []
    for name, value in results_data:
        values.append(value)
    values = np.array(values).reshape(2,5)
    print("values: ", values)
    # plt.imshow(values, cmap='hot', interpolation='nearest')
    ax = sns.heatmap(values, linewidth=0.5)
    plt.title("Reachability Analysis")
    plt.xlabel("cylinders")
    plt.ylabel("rows")
    plt.savefig("ReachabilityAnalysis.png")

    # The calculated trajectory from the robot is sent to the controller
    trajectory = vis_traj
    return trajectory