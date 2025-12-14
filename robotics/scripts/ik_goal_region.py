"""
OMPL GoalSampleableRegion implementation for collision problem,
so we are creating multiple inverse kinematic solutions.

The problem is in exercise 3/6/7
is its only select the first solution,
so its not the optimal one.

with GoalSampleableRegion we solves these:
1. Pre-compute ALL valid IK solutions (collision-free)
2. RRT samples from goals, picks EASIEST to reach
3. Implicit path optimization (shorter paths preferred)
4. Centralized, reusable implementation

IK Sampling Strategy - Why J1 Sweep with 16 Samples?
=====================================================

1. UR5 has up to 8 unique IK solutions:
   Source: https://alexanderelias.com/ur5-ik/

2. Numerical IK (ik_LM) heavily depends on initial guess q0:
   Source: https://petercorke.github.io/robotics-toolbox-python/IK/ik.html

3. Why sweep J1 (base joint)?
   - J1 has the widest motion range (-pi to +pi)
   - Different J1 values lead to completely different arm configurations
   - Sweeping J1 explores different "elbow up/down", "shoulder left/right" configs

4. Why 16 samples?
   - Minimum needed: 8 (max number of UR5 IK solutions)
   - We use 16 = 2x safety margin (22.5 degree intervals)
   - Ensures we don't miss valid solutions due to numerical convergence issues

OMPL References:
- https://ompl.kavrakilab.org/classompl_1_1base_1_1GoalSampleableRegion.html
- https://ompl.kavrakilab.org/goalRepresentation.html

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
import random
from ompl import base as ob
from utils.mujoco_utils import is_q_valid, is_q_valid_with_held_object, load_config


class IKGoalRegion(ob.GoalSampleableRegion):
    """
    Usage referance:
        goal = IKGoalRegion(si, robot, target_frame, d, m)
        ss.setGoal(goal)  # Instead of ss.setStartAndGoalStates()
    """

    def __init__(self, si, robot, target_frame, d, m, n_samples=None, target_object=None, held_object=None):
        """
        Args:
            si: OMPL SpaceInformation
            robot: roboticstoolbox robot (for IK computation)
            target_frame: SE3 target pose
            d: MuJoCo data
            m: MuJoCo model
            n_samples: Number of IK attempts with different J1 seeds (default from config)
            target_object: Name of target object (collisions with it are ignored)
            held_object: Name of object being held during transport (for collision check)
        """
        super(IKGoalRegion, self).__init__(si)
        self.setThreshold(0.01)  # Goal distance threshold in radians
        self.robot = robot
        self.target_frame = target_frame
        self.d = d
        self.m = m
        self.target_object = target_object
        self.held_object = held_object
        self.valid_solutions = []

        # Load n_samples from config if not provided
        if n_samples is None:
            config = load_config()
            n_samples = config.get('ik_sampling', {}).get('n_samples', 16)

        self._compute_ik_solutions(n_samples)

    def _compute_ik_solutions(self, n_samples):
        """
        Checking it with J1 which is base frame
        # TODO: Baha be more spesific explanation in here for why did u choose J1,
        """
        print(f"[IKGoalRegion] Computing IK solutions for target...")
        print(f"  Target position: {self.target_frame.t}")
        if self.held_object:
            print(f"  Held object: {self.held_object}")

        for i in range(n_samples):

            theta = -np.pi + (2 * np.pi * i / n_samples)
            q_init = np.zeros(6)
            q_init[0] = theta

            result = self.robot.ik_LM(Tep=self.target_frame, q0=q_init)

            if result[1]:
                q_solution = result[0]

                # Use appropriate collision checker based on whether object is held
                if self.held_object:
                    is_valid = is_q_valid_with_held_object(
                        self.d, self.m, q_solution,
                        robot_rtb=self.robot,
                        held_object=self.held_object,
                        target_object=self.target_object
                    )
                else:
                    is_valid = is_q_valid(self.d, self.m, q_solution, target_object=self.target_object)

                if is_valid:
                    is_unique = True
                    for existing in self.valid_solutions:
                        if np.linalg.norm(q_solution - existing) < 0.1:
                            is_unique = False
                            break

                    if is_unique:
                        self.valid_solutions.append(q_solution)
                        print(f"  [+] Valid solution: J1={np.degrees(q_solution[0]):.1f}deg")

        print(f"[IKGoalRegion] Found {len(self.valid_solutions)} valid solutions")

    def distanceGoal(self, state):
        """
        Distance from state to nearest valid IK solution.
        """
        if not self.valid_solutions:
            return float('inf')

        q = np.array([state[i] for i in range(6)])
        min_dist = min(np.linalg.norm(q - sol) for sol in self.valid_solutions)
        return min_dist

    def sampleGoal(self, state):
        """
        Sample a random valid IK solution.

        IMPORTANT: Must use index assignment (state[i] = val),
        do not use direct assignment (state = ...) due to OMPL Python bindings.
        """
        if self.valid_solutions:
            sol = random.choice(self.valid_solutions)
            for i in range(6):
                state[i] = sol[i]

    def maxSampleCount(self):
        """
        Number of unique goal samples available.
        """
        return len(self.valid_solutions) if self.valid_solutions else 0

    def canSample(self):
        """
        checking valid_positions
        """
        return len(self.valid_solutions) > 0

    def couldSample(self):
        """
        Is goal sampling possible ?
        """
        return True

    def hasValidSolution(self):
        """
        Check if at least one valid solution exists
        """
        return len(self.valid_solutions) > 0

    def getBestSolution(self, start_q):
        """
        Get the solution closest to start configuration
        """
        if not self.valid_solutions:
            return None
        return min(self.valid_solutions, key=lambda s: np.linalg.norm(s - start_q))

    def getAllSolutions(self):
        """
        Return all valid solutions (for debugging/visualization)
        """
        return self.valid_solutions.copy()
