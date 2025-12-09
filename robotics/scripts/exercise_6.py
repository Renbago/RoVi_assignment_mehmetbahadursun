
import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random

from ompl import base as ob
from ompl import geometric as og

from robot import *


class StateValidator:
    # This is mujoco specific, so I have implemented this for you
    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint
    
    def __call__(self, state):
        # print("isStateValid - state: ", state)
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose) 

def plan(d, m, start_q, goal_q):
    num_joint = 6
    
    space = ob.RealVectorStateSpace(num_joint) # Create a joint-space vector instead of a 2D space as in the example
    # Create joint bounds
    bounds = ob.RealVectorBounds(num_joint)
    bounds.setLow(-3.14)
    bounds.setHigh(3.14)
    space.setBounds(bounds)
    
    # Create SimpleSetup
    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, num_joint)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    
    # Set start and goal states
    start = ob.State(space)
    goal = ob.State(space)
    
    # Set specific joint values for start and goal
    for i in range(num_joint):
        start[i] = start_q[i] # initial joint angles
        goal[i] = goal_q[i]  # goal joint angles (~90 degrees)
    
    ss.setStartAndGoalStates(start, goal)

    # ===== EXPLICITLY SET PLANNER =====

    usr_input = input("rrt/prm: ")

    # PRM
    if usr_input.lower() == "prm":
        planner = og.PRM(ss.getSpaceInformation())
    # RRT
    else:
        planner = og.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner)
    
    # Solve the problem
    solved = ss.solve(10.0)
    
    if solved:
        # Get the solution path
        path = ss.getSolutionPath()
        print("Found a Solution!")
        # Print basic information about the path
        print(f"Path length: {path.length()}")
        print(f"Number of states: {path.getStateCount()}")

        solution_trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q_pose = [state[i] for i in range(space.getDimension())]
            print(f"State {i}: {q_pose}")
            solution_trajectory.append(np.array(q_pose))

        return solution_trajectory


def program(d, m):
    # Define our robot object
    robot = UR5robot(data=d, model=m)
    
    all_trajectories = []  # Store all planned paths for visualization
    
    start_q = robot.get_current_q()
    # Define grasping frames for object: box
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_box") # Get body frame
    obj_frame = obj_frame * sm.SE3.Rx(-np.pi)
    goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q)[0]
    sol_traj = plan(d=d, m=m, start_q=start_q, goal_q=goal_q)
    all_trajectories.extend(sol_traj)

    robot.move_j_via(points=sol_traj, t=500)   

    start_q = goal_q
    # Define grasping frames for object: box
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name="pickup_point_box") # Get body frame
    obj_frame = obj_frame * sm.SE3.Rx(-np.pi)
    goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q)[0]
    sol_traj = plan(d=d, m=m, start_q=start_q, goal_q=goal_q)
    all_trajectories.extend(sol_traj)

    robot.move_j_via(points=sol_traj, t=500)   

    return robot.queue, robot, all_trajectories

    
    

       
         
