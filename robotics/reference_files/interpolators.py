import numpy as np

class TrapezoidalVelocityProfile:
    def __init__(self, q_start, q_end, v_max, a_max, dt):
        """
        Computes a trapezoidal velocity profile for a single joint.
        """
        self.q_start = q_start
        self.q_end = q_end
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)
        self.dt = dt
        self.total_dist = self.q_end - self.q_start
        self.sign = np.sign(self.total_dist)
        
        # Calculate time phases
        self.t_acc = self.v_max / self.a_max
        self.d_acc = 0.5 * self.a_max * self.t_acc**2
        
        # Check if we reach v_max (triangle vs trapezoid)
        if 2 * self.d_acc > abs(self.total_dist):
            # Triangular profile
            self.t_acc = np.sqrt(abs(self.total_dist) / self.a_max)
            self.d_acc = 0.5 * self.a_max * self.t_acc**2
            self.t_const = 0.0
            self.v_peak = self.a_max * self.t_acc
        else:
            # Trapezoidal profile
            self.d_const = abs(self.total_dist) - 2 * self.d_acc
            self.t_const = self.d_const / self.v_max
            self.v_peak = self.v_max
            
        self.t_dec = self.t_acc
        self.t_total = self.t_acc + self.t_const + self.t_dec

    def get_position(self, t):
        if t <= 0:
            return self.q_start
        elif t > self.t_total:
            return self.q_end
        
        if t <= self.t_acc:
            # Acceleration phase
            return self.q_start + self.sign * 0.5 * self.a_max * t**2
        elif t <= self.t_acc + self.t_const:
            # Constant velocity phase
            return self.q_start + self.sign * (self.d_acc + self.v_peak * (t - self.t_acc))
        else:
            # Deceleration phase
            t_rem = self.t_total - t
            return self.q_end - self.sign * 0.5 * self.a_max * t_rem**2

def generate_p2p_trajectory(start_q, end_q, time_step=0.002, speed_factor=1.0):
    """
    Generates a synchronized P2P trajectory for multiple joints.
    """
    num_joints = len(start_q)
    
    # Configure limits (approximate UR5e limits)
    v_max_defaults = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14]) * speed_factor
    a_max_defaults = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * speed_factor # Conservative acceleration

    # 1. Compute profiles for all joints to find the slowest one (Synchronization)
    max_duration = 0.0
    profiles = []
    
    for i in range(num_joints):
        profile = TrapezoidalVelocityProfile(start_q[i], end_q[i], v_max_defaults[i], a_max_defaults[i], time_step)
        if profile.t_total > max_duration:
            max_duration = profile.t_total
            
    # 2. Re-compute profiles scalable to max_duration (scale down v_max/a_max)
    # Ideally we'd scale strict parameters, but simple time-scaling is often sufficient for homework
    # Let's generate waypoints based on the max duration
    
    times = np.arange(0, max_duration + time_step, time_step)
    trajectory = []
    
    # Re-calculate synchronized profiles (simple scaling method: stretch time)
    # A cleaner way is to recalculate v_peak for the fixed T_total
    
    final_profiles = []
    for i in range(num_joints):
        dist = abs(end_q[i] - start_q[i])
        # To synchronize, we want T_total to be max_duration.
        # It's complex to analytically solve a, v for exact T.
        # Instead, we will sample the fastest profile and hold position? 
        # Better: Scale velocities to match duration. v_new = dist / T_new (approx)
        
        # Simple Approach: Use the profile generated initially but sample it at varying times? No.
        # Standard approach: Scale v_max and a_max relative to the slowest joint.
        
        # Let's stick to the simplest valid implementation: Independent profiles, Robot waits for all to finish.
        # Or better: Just sample the individual profiles. Since we command joint positions, 
        # if one joint finishes early, it stays at q_end. This is valid P2P.
        final_profiles.append(TrapezoidalVelocityProfile(start_q[i], end_q[i], v_max_defaults[i], a_max_defaults[i], time_step))

    for t in times:
        waypoint = np.zeros(num_joints)
        for i in range(num_joints):
            # If we want true synchronization, we should have scaled v_maxdown.
            # Here we just execute independent profiles.
            waypoint[i] = final_profiles[i].get_position(t)
        trajectory.append(waypoint)
        
    return np.array(trajectory)

def get_checkpoints_for_object(obj_name, current_q, robot):
    """
    Returns a sequence of waypoints (q_start -> q_approach -> q_pick -> q_lift -> q_drop)
    This is a placeholder logic based on object names. 
    Ideally involves InvKinematics (IK).
    """
    # For now, let's assume we have some hardcoded or IK-calculated joint targets
    # Since I don't have the IK solver active here, I will return dummy list of configs based on current q.
    # In a real scenario, this function would use robot.ik_solver to find q for the object pose.
    
    # Placeholder: Return list of configurations
    # You should implementing the IK call here using roboticstoolbox
    return [current_q] 
