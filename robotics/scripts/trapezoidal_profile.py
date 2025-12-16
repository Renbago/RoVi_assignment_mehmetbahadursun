"""
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import numpy as np
from typing import Tuple, List, Union, Optional

class TrapezoidalProfile:
    """
        The problem is linear interpolation,
        with linear interpolation, velocity = derivative of position is constant:
            q_dot = (q_f - q_i) / (t_f - t_i) == constant value
        
        However, at t=0, velocity jumps from 0 to this constant value instantly.
        Since this transition happens in infinitely short time:
            q_ddot = delta_v/delta_t = V/0 == infinite acceleration

        so as a result, motor gears will get damage.
        
        There is a alternatives method for it but the task is about
        trapezoidal profile so lets start with that but for learning
        i will also implement other classes whenever i do have time.

        1. Acceleration phase: 0 to q_dot_c (linear increase)
        2. Constant velocity phase: q_dot_c (constant cruise)
        3. Deceleration phase: q_dot_c to 0 (linear decrease)
        
        its can be seenable from the book eq 4.8 

        From study pdf's there is a parabolic blend 
        so i think its good to mention in here:

        Parabolic blend is kinda same as a trapezoidal profile,
        The difference is:
        - Parabolic blend is position-based (linear + parabolic pieces)
        - Trapezoidal is velocity-based (acceleration + cruise + deceleration)
        

        Extra knowledges from book for my studies and later latex format: 

        In particular, consider the case of one intermediate point only, and suppose
        that trapezoidal velocity profiles are considered as motion primitives with
        the possibility to specify the initial and final point and the duration of the
        motion only; it is assumed that ˙qi = ˙qf = 0. If two segments with trapezoidal
        velocity profiles were generated, the manipulator joint would certainly reach
        the intermediate point, but it would be forced to stop there, before continuing
        the motion towards the final point. A keen alternative is to start generating
        the second segment ahead of time with respect to the end of the first segment,
        using the sum of velocities (or positions) as a reference. In this way, the joint
        is guaranteed to reach the final position; crossing of the intermediate point at
        the specified instant of time is not guaranteed, though.

        Reference sections are mentioned but
        in also under of the code sections but the informations and the references
        are those:

        the main book :Sciavicco, "Robotics: Modelling, Planning and Control", 
        Section 4.2, 4.8/ Figures: 4.3, 4.10
    """

    # Extra informations to note my self while working.

    """
        so this is must be third order or else we will be losing of the dependency on time.

        For trapezoidal profile, we assume q_dot(t_i) = q_dot(t_f) = 0.
        If we need non-zero boundary velocities, we should use cubic/quintic polynomials

        you can see the differences. I will be try to add the functions for show it.

        1. Trapezoidal Profile (LSPB):
            - Velocity is "Trapezoidal" (Linear ramp -> Constant -> Linear ramp).
            - Position is "Parabolic" (t^2) during accel/decel, and Linear (t) during cruise.
            - Pros: Fastest way to move within acceleration limits. Easy to verify limits.
            - Cons: Jerk is infinite at phase switches (sudden acceleration change).
        
        2. Cubic Polynomial (3rd Order):
            - Position is described by one equation: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
            - Smoother than Trapezoidal but doesn't have a "constant velocity" phase.
        
        3. Quintic Polynomial (5th Order):
            - q(t) = ... + a5*t^5
            - Even smoother (Finite Jerk).


        referance: https://blogs.mathworks.com/student-lounge/2019/11/06/robot-manipulator-trajectory/

        Cubic (3rd order) – Requires 4 boundary conditions: position and velocity at both ends
        Quintic (5th order) – Requires 6 boundary conditions: position, velocity, and acceleration at both ends

        as a prove:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        q_dot(t) = a1 + 2*a2*t + 3*a3*t^2
        q_ddot(t) = 2*a2 + 6*a3*t

        You can interpolate between two waypoints using polynomials of various orders. 

        For quintic (5th order), we add 2 more constraints:
        - q_ddot(t_i) = 0 (initial acceleration = 0)
        - q_ddot(t_f) = 0 (final acceleration = 0)
        This gives smoother motion but requires more computation.
        the 5th order is for jerk implementation
        the guy whose find it is Nevilla Hogan  extra information for whose reading :}

    """

    def __init__(self, 
                 q_i: Union[float, np.ndarray], 
                 q_f: Union[float, np.ndarray], 
                 t_f: float, 
                 t_i: float = 0.0,
                 q_dot_c: Optional[Union[float, np.ndarray]] = None,
                 q_ddot_c: Optional[Union[float, np.ndarray]] = None):
        """
        Args:
            q_i: Initial position q(t_i)
            q_f: Final position q(t_f)
            t_f: Final time (duration if t_i=0)
            t_i: Initial time (default: 0)
            q_dot_c: Cruise velocity (optional). If both are provided, q_dot_c takes precedence.
            because u can cerate the q_ddot_c from q_dot_c.
            q_ddot_c: Cruise acceleration (optional).
        
            Trapezoidal velocity trajectories are piecewise trajectories of constant acceleration,
            zero acceleration, and constant deceleration. This leads to a trapezoidal velocity profile,
            and a “linear segment with parabolic blend” (LSPB) or s-curve position profile.
            
        """

        # converted positions to numpy arrays so we can do multi joint profiles
        # np.atleast_1d to handle both scalar and array inputs uniformly
        # referanced: https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html
        self.q_i = np.atleast_1d(np.asanyarray(q_i, dtype=float))
        self.q_f = np.atleast_1d(np.asanyarray(q_f, dtype=float))
        
        # store original dimensionality to return appropriate type (scalar vs array)
        self.is_scalar = (np.asanyarray(q_i).ndim == 0)
        
        self.t_f = float(t_f)
        self.t_i = float(t_i)
        
        # for creating a space in equaitons 4.10/4.11
        self.delta_q = self.q_f - self.q_i
        
        """
        First create the acceleration paremeters and boundary conditions
        """ 

        # referance:  eq (4.7)
        self.acc_min_abs = 4 * np.abs(self.delta_q) / (self.t_f ** 2)

        # Max velocity (Eq 4.9 upper bound)
        self.vel_max_abs = 2 * np.abs(self.delta_q) / self.t_f
        
        # Min velocity (Eq 4.9 lower bound)
        self.vel_min_abs = np.abs(self.delta_q) / self.t_f

        # Initialize cruise parameters
        self.q_dot_c = np.zeros_like(self.delta_q)
        self.q_ddot_c = np.zeros_like(self.delta_q)
        self.t_c = np.zeros_like(self.delta_q)


        if q_dot_c is not None:
            # Case 1: if the vel provided

            user_vel = np.atleast_1d(np.asanyarray(q_dot_c, dtype=float))
            if user_vel.size == 1 and self.q_i.size > 1:
                user_vel = np.full_like(self.delta_q, user_vel[0])

            vel_mag = np.abs(user_vel)
            
            """
            i've used boolean indexing instead of for loop its more optimized
            https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
            """
            # Check Eq (4.9) constraints
            velocity_value = vel_mag < self.vel_min_abs
            if np.any(velocity_value):
                print("Warning: Velocity too low. (Eq 4.9 lower bound)")
                vel_mag[velocity_value] = self.vel_min_abs[velocity_value] * 1.001

            velocity_value = vel_mag > self.vel_max_abs
            if np.any(velocity_value):
                print("Warning: Velocity too high. (Eq 4.9 upper bound)")
                vel_mag[velocity_value] = self.vel_max_abs[velocity_value] * 0.999

            # just to make sure its higher then epsilon for not attaching random value
            # its quite good if u check that
            nonzero_dist = np.abs(self.delta_q) > 1e-9
            self.q_dot_c[nonzero_dist] = np.sign(self.delta_q[nonzero_dist]) * vel_mag[nonzero_dist]
            
            # setup
            valid_idx = np.abs(self.q_dot_c) > 1e-9
            equation = (-self.delta_q[valid_idx] + self.q_dot_c[valid_idx] * self.t_f)
            
            # Eq (4.10)
            self.t_c[valid_idx] = equation / self.q_dot_c[valid_idx]
            
            # Eq (4.11)
            self.q_ddot_c[valid_idx] = (self.q_dot_c[valid_idx] ** 2) / equation

        elif q_ddot_c is not None:
            # Case 2: if the acc provided

            user_acc = np.atleast_1d(np.asanyarray(q_ddot_c, dtype=float))
            if user_acc.size == 1 and self.q_i.size > 1:
                user_acc = np.full_like(self.delta_q, user_acc[0])
            
            acc_magnitude = np.abs(user_acc)
            
            # Eq (4.7)
            # there is no upper bound for acceleration if u increase the value
            # instead of the triangle profile it will become a rectangle 
            acc_value = acc_magnitude < self.acc_min_abs
            if np.any(acc_value):
                print("Warning: Acceleration too low (Eq 4.7)")
                acc_magnitude[acc_value] = self.acc_min_abs[acc_value] * 1.001
            
            nonzero_dist = np.abs(self.delta_q) > 1e-9
            self.q_ddot_c[nonzero_dist] = np.sign(self.delta_q[nonzero_dist]) * acc_magnitude[nonzero_dist]
            
            self._compute_t_c_from_accel()
            self.q_dot_c = self.q_ddot_c * self.t_c

        else:
            # Case 3, 
            # We select a default acceleration based on Eq (4.7).

            print("Notice: No q_dot_c or q_ddot_c provided. Auto-calculating based on t_f (Eq 4.6 & 4.7).")
            
            # the 1.1 is just initial choice 
            # just higher then a bit from the lowest acceleration
            acc_magnitude = self.acc_min_abs * 1.1 
            
            nonzero_dist = np.abs(self.delta_q) > 1e-9
            self.q_ddot_c[nonzero_dist] = np.sign(self.delta_q[nonzero_dist]) * acc_magnitude[nonzero_dist]
            
            self._compute_t_c_from_accel()
            self.q_dot_c = self.q_ddot_c * self.t_c
            
        # Checking the system
        # before run it we need to check is it working
        # in theory
        # if t_c > t_f/2, it means we don't have enough time 
        # to reach the cruise velocity and then decelerate. 
        # From book:
        """
            Usually, ¨qc is specified with the constraint that sgn ¨qc = sgn (qf − qi); hence,
            for given tf , qi and qf , the solution for tc is computed from (4.5) as (tc ≤ tf /2)
        """
        is_triangular = (self.t_c >= self.t_f / 2)
        # so we are checking here if we passed half of the time we are forcing
        # to be triangular with making the same time for acc and dec
        if np.any(is_triangular):
            self.t_c[is_triangular] = self.t_f / 2
            # recalculate peak velocity matching this new t_c
            # v_peak = a * (tf/2)
            self.q_dot_c[is_triangular] = self.q_ddot_c[is_triangular] * self.t_c[is_triangular]


    def _compute_t_c_from_accel(self):
        """
        t_c = time of cruise

        Eq (4.6)
        Usually,q_ddot_c is specified with the constraint that sgn q_ddot_c = sgn (q_f − q_i); hence,
        for given tf , q_i and q_f , the solution for tc is computed from (4.5) as (tc ≤ tf /2)
        
        """
        # Avoid division by zero
        # like rn we have a 6 DoF robot so we need to avoid division by zero.
        # lets say the 3rd joint is moving but the first jint is stable so if we try to
        # divide it its give error so we cancel de divide and invalid calculation
        # errors in here
        with np.errstate(divide='ignore', invalid='ignore'):
            term = ((self.t_f**2)*self.q_ddot_c - 4*(self.delta_q)) / self.q_ddot_c
            
            # term must be higher then 0 bcs we get sqrt of it
            # so if its negative we assign it 0 and that eans
            # the robot cannot reach the final position in the given time

            sqrt_val = np.sqrt(np.maximum(term, 0))
            
            self.t_c = 0.5 * self.t_f - 0.5 * sqrt_val
            
            # Where accel is 0 (no motion), t_c remains 0
            self.t_c[np.isnan(self.t_c)] = 0.0

    def get_position(self, t: float) -> Union[float, np.ndarray]:
        """
        Get position(s) at time t.
        """
        tau = t - self.t_i
        
        # Initialize with boundaries
        q = np.full_like(self.q_i, self.q_i) # Default to q_i
        
        if tau <= 0:
            result = self.q_i
        elif tau >= self.t_f:
            result = self.q_f
        else:
            # Masks for different phases
            # Phase 1: Acceleration (0 < tau <= t_c)
            mask_acc = (tau > 0) & (tau <= self.t_c)
            
            # Phase 2: Constant Velocity (t_c < tau <= t_f - t_c)
            mask_cruise = (tau > self.t_c) & (tau <= self.t_f - self.t_c)
            
            # Phase 3: Deceleration (t_f - t_c < tau < t_f)
            mask_dec = (tau > self.t_f - self.t_c) & (tau < self.t_f)

            # Apply logic per phase (vectorized)
            # Accel: q = qi + 0.5 * a * t^2
            if np.any(mask_acc):
                q[mask_acc] = self.q_i[mask_acc] + 0.5 * self.q_ddot_c[mask_acc] * (tau ** 2)
            
            # Cruise: q = qi + a*tc*(t - tc/2)
            if np.any(mask_cruise):
                q[mask_cruise] = self.q_i[mask_cruise] + self.q_ddot_c[mask_cruise] * self.t_c[mask_cruise] * (tau - self.t_c[mask_cruise] / 2)
            
            # Decel: q = qf - 0.5 * a * (tf - t)^2
            if np.any(mask_dec):
                tau_rem = self.t_f - tau
                q[mask_dec] = self.q_f[mask_dec] - 0.5 * self.q_ddot_c[mask_dec] * (tau_rem ** 2)
                
            result = q

        return result[0] if self.is_scalar else result

    def get_velocity(self, t: float) -> Union[float, np.ndarray]:
        """Get velocity at time t."""
        tau = t - self.t_i
        v = np.zeros_like(self.q_i)
        
        if tau <= 0 or tau >= self.t_f:
            return 0.0 if self.is_scalar else v
            
        mask_acc = (tau <= self.t_c)
        mask_cruise = (tau > self.t_c) & (tau <= self.t_f - self.t_c)
        mask_dec = (tau > self.t_f - self.t_c)
        
        # Accel: v = a * t
        if np.any(mask_acc):
            v[mask_acc] = self.q_ddot_c[mask_acc] * tau
            
        # Cruise: v = v_c
        if np.any(mask_cruise):
            v[mask_cruise] = self.q_dot_c[mask_cruise]
            
        # Decel: v = a * (tf - t)
        if np.any(mask_dec):
            tau_rem = self.t_f - tau
            v[mask_dec] = self.q_ddot_c[mask_dec] * tau_rem
            
        return v[0] if self.is_scalar else v

    def get_acceleration(self, t: float) -> Union[float, np.ndarray]:
        """Get acceleration at time t."""
        tau = t - self.t_i
        a = np.zeros_like(self.q_i)
        
        if tau <= 0 or tau >= self.t_f:
            return 0.0 if self.is_scalar else a
            
        mask_acc = (tau <= self.t_c)
        mask_dec = (tau > self.t_f - self.t_c)
        # Cruise phase acceleration is naturally 0
        
        if np.any(mask_acc):
            a[mask_acc] = self.q_ddot_c[mask_acc]
            
        if np.any(mask_dec):
            a[mask_dec] = -self.q_ddot_c[mask_dec]
            
        return a[0] if self.is_scalar else a

    def sample_trajectory(self, dt: float = 0.002):
        """
        Samples the trajectory at fixed time steps.
        
        Returns:
            times: [N] array of times
            positions: [N x D] (or [N] if scalar) positions
            velocities: [N x D] (or [N] if scalar) velocities
            accelerations: [N x D] (or [N] if scalar) accelerations
        """
        times = np.arange(self.t_i, self.t_i + self.t_f + dt, dt)
        
        positions = []
        velocities = []
        accelerations = []
        
        # This loop could be vectorized further but is usually fast enough for plotting
        for t in times:
            positions.append(self.get_position(t))
            velocities.append(self.get_velocity(t))
            accelerations.append(self.get_acceleration(t))
            
        return times, np.array(positions), np.array(velocities), np.array(accelerations)


# --- Helper Function for P2P Via Points ---

def generate_via_point_trajectory(via_points: List[np.ndarray], 
                                   segment_times: List[float],
                                   dt: float = 0.002) -> List[np.ndarray]:
    """
    Generate trajectory through multiple via points (P2P segments).
    Stops at each via point (velocity=0).
    
    Args:
        via_points: List of joint configurations (N points)
        segment_times: List of durations for each segment (N-1 times)
        dt: Time step
    """
    assert len(segment_times) == len(via_points) - 1, \
        "Need N-1 segment times for N via points"
    
    full_trajectory = []
    current_time = 0.0
    
    for i in range(len(via_points) - 1):
        # Create profile for this segment
        # We can pass arrays directly now!
        profile = TrapezoidalProfile(
            q_i=via_points[i],
            q_f=via_points[i+1],
            t_f=segment_times[i],
            t_i=current_time  # Start where previous left off? 
                              # Actually sample_trajectory handles absolute time?
                              # Yes, if we set t_i.
        )
        
        # Sample
        _, pos, _, _ = profile.sample_trajectory(dt)
        
        # Append to full list (avoid duplicating boundary points if needed, but for simple list it's fine)
        # Convert numpy array of positions to list of arrays
        for p in pos:
             full_trajectory.append(p)
             
        current_time += segment_times[i]
        
    return full_trajectory
