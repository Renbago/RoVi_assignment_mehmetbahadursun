
import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random

from ompl import base as ob
from ompl import geometric as og

from robot import *


class StateValidator:
    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint
    
    def __call__(self, state):
        # print("isStateValid - state: ", state)
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose)

def rrt_plan(d, m, start_q, goal_q):
    num_joint = 6
    # Create your q-space (choose one of the above)
    space = ob.RealVectorStateSpace(num_joint)
    bounds = ob.RealVectorBounds(num_joint)
    bounds.setLow(-3.14)
    bounds.setHigh(3.14)
    space.setBounds(bounds)
    
    # Create SimpleSetup
    ss = og.SimpleSetup(space)
    # Create a partial function with bound arguments
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
    planner = og.RRT(ss.getSpaceInformation())
    ss.setPlanner(planner)
    
    # Solve the problem
    solved = ss.solve(10.0)
    
    if solved:
        # ss.simplifySolution() # OMPL's inbuilt Path pruning algorithm
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
    

def path_prune_simple(d, m, trajectory):
    pruned = [trajectory[0]]
    current_idx = 0
    
    while current_idx < len(trajectory) - 1:
        next_idx = len(trajectory) - 1
        
        # We look for the furthest reachable point from the current_idx q-pose
        while next_idx > current_idx + 1:
            if check_valid_linear(d, m, trajectory[current_idx], trajectory[next_idx]):
                break
            next_idx -= 1
        
        pruned.append(trajectory[next_idx])
        current_idx = next_idx
    
    return np.array(pruned)

def check_valid_linear(d, m, a, b):
    # We check if it's possible to reach b from a (linear in joint space) without any collisions
    for qpose in np.linspace(a, b, 100):
        if not is_q_valid(d=d, m=m, q=qpose):
            return False
    return True





def program(d, m):

    # Simple program using RRT for collision free planning of a pick and place of the box
    # Use your path pruning algorithm to reduce redundant q-poses

    # Define our robot object
    robot = UR5robot(data=d, model=m)

    start_q = robot.get_current_q()
    # # Define grasping frames for object: box
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name="box")
    obj_frame = obj_frame * sm.SE3.Rx(-np.pi)

    goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q)[0]
    sol_traj = rrt_plan(d=d, m=m, start_q=start_q, goal_q=goal_q)

    robot.move_j_via(points=sol_traj, t=500)   

    robot.set_gripper(255)

    start_q = goal_q
    # # Define grasping frames for object: box
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name=f"drop_point_box")
    obj_frame = obj_frame * sm.SE3.Rx(-np.pi)

    goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q)[0]
    sol_traj = rrt_plan(d=d, m=m, start_q=start_q, goal_q=goal_q)
    print("Before: ", len(sol_traj))
    sol_traj = path_prune_simple(d, m, sol_traj)
    print("After: ", len(sol_traj))

    robot.move_j_via(points=sol_traj, t=500)   
    robot.set_gripper(0)

    return robot.queue

    
    

       
         
