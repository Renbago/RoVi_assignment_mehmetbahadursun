"""
This file is for planners so everything can be added here
dynamicly change the planner type 

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import numpy as np
from robot import is_q_valid
from ompl import base as ob
from ompl import geometric as og
import spatialmath as sm

class StateValidator:
    # This is mujoco specific, so I have implemented this for you
    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint
        self.all_trajectories = []
    
    def __call__(self, state):
        # We assume 6 joints as per standard UR5
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose) 

def generate_p2p_path(robot, start_q, target_frame):
    """
    #TODO: use p2p_point_saver.py and create 6 points
    then write in here or else u can just connect the data's ? maybe
    """

    # The starting pose
    current_pose = robot.robot_ur5.fkine(start_q)
    start_q = np.array(start_q)
    target_frame = np.array(target_frame)
    # TODO: path will be at least 6 points
    path = None
    print(f"[P2P] generated {len(path)} waypoints with 6-point requirement")
    return path

def plan(d, m, start_q, goal_q, planner_type="rrt", robot=None, target_frame=None):
    """
    Unified planner function.
    For now only supports but i want to add more:
    - 'rrt', 'prm', 'p2p':
    """
    
    # P2P
    if planner_type.lower() == "p2p":
        print("do not supported right now")
        return generate_p2p_path(robot, start_q, target_frame)
        # if robot is None or target_frame is None:
        #     print("P2P requires 'robot' and 'target_frame' arguments.")
        #     return None

        # return generate_p2p_path(robot, start_q, target_frame)

    # RRT / PRM
    num_joint = 6
    
    space = ob.RealVectorStateSpace(num_joint)
    bounds = ob.RealVectorBounds(num_joint)
    bounds.setLow(-3.14) # TODO: change with directly math.pi its kinda :P
    bounds.setHigh(3.14)
    space.setBounds(bounds)
    
    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, num_joint)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    
    start = ob.State(space)
    goal = ob.State(space)
    
    for i in range(num_joint):
        start[i] = start_q[i]
        goal[i] = goal_q[i]
    
    ss.setStartAndGoalStates(start, goal)

    if planner_type.lower() == "prm":
        planner = og.PRM(ss.getSpaceInformation())
    else:
        planner = og.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner)
    
    solved = ss.solve(5.0) # TODO: the 5 seconds might be not fit check it
    
    if solved:
        path = ss.getSolutionPath()
        # TODO: for fair comparison i gave it 6 but its not required
        # So also add the without interpolate we can check the difference also
        path.interpolate(6) 
        
        # TODO: create a logger or use same logic with the ros2
        # self.logger_info, logger_debug, logger_error do it tomorrow its very bad
        # debugging way also write in the file always the test results for 
        # latex output later
        print(f"Debug: path type: {type(path)}")
        print(f"Debug: path state count: {path.getStateCount()}")
        if path.getStateCount() > 0:
            st = path.getState(0)

            print(f"Debug: sample state[0]: {[st[i] for i in range(6)]}")
            
        print(f"solution ({planner_type})! length: {path.length()}, states: {path.getStateCount()}")
        
        solution_trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q_pose = [state[i] for i in range(space.getDimension())]
            solution_trajectory.append(np.array(q_pose))

        return solution_trajectory
    else:
        print(f"{planner_type} failed to find a path.")
        return None
