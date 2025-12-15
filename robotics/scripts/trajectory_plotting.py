"""
Plotting utilities for trajectory analysis and visualization.
Generates plots for position, velocity, and acceleration profiles.
"""

"""
Helper functions for plotting joint trajectories and via-points.
Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import os

# Base output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def _get_output_path(filename: str) -> str:
    """
    Helper to ensure file saves to outputs/ directory if just name provided
    """
    
    if filename and not os.path.dirname(filename):
        return os.path.join(OUTPUT_DIR, filename)
    return filename


def plot_trapezoidal_profile(profile, save_path: Optional[str] = None, show: bool = True):
    """
    Plot position, velocity, acceleration for a TrapezoidalProfile.
    Assumes single joint profile for now.
    """
    times, positions, velocities, accelerations = profile.sample_trajectory()
    
    # Extract scalar values for labels if input is array
    def to_scalar(val):
        return val.item() if hasattr(val, 'item') and val.size == 1 else val
        
    q_i = to_scalar(profile.q_i)
    q_f = to_scalar(profile.q_f)
    t_c = to_scalar(profile.t_c)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Position
    axes[0].plot(times, positions, 'b-', linewidth=2)
    axes[0].set_ylabel('Position [rad]')
    axes[0].set_title('Trapezoidal Velocity Profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=q_i, color='g', linestyle='--', alpha=0.5, label=f'$q_i$={q_i:.3f}')
    axes[0].axhline(y=q_f, color='r', linestyle='--', alpha=0.5, label=f'$q_f$={q_f:.3f}')
    axes[0].legend(loc='right')
    
    # Velocity
    axes[1].plot(times, velocities, 'r-', linewidth=2)
    axes[1].set_ylabel('Velocity [rad/s]')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=t_c, color='g', linestyle=':', alpha=0.7, label=f'$t_c$={t_c:.3f}s')
    axes[1].axvline(x=profile.t_f - t_c, color='g', linestyle=':', alpha=0.7)
    axes[1].legend()
    
    # Acceleration
    axes[2].plot(times, accelerations, 'g-', linewidth=2)
    axes[2].set_ylabel('Acceleration [rad/s²]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {final_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_multi_joint_trajectory(trajectory: List[np.ndarray], 
                                 t_f: float,
                                 joint_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None,
                                 show: bool = True):
    """
    Plot multi-joint trajectory.
    
    """
    trajectory = np.array(trajectory)
    n_points, n_joints = trajectory.shape
    times = np.linspace(0, t_f, n_points)
    
    if joint_names is None:
        joint_names = [f'Joint {i+1}' for i in range(n_joints)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_joints))
    
    for i in range(n_joints):
        ax.plot(times, trajectory[:, i], color=colors[i], label=joint_names[i], linewidth=1.5)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Joint Position [rad]')
    ax.set_title('Multi-Joint Trajectory')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {final_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_trajectory_derivatives(trajectory: List[np.ndarray],
                                 dt: float = 0.002,
                                 joint_idx: int = 0,
                                 title_prefix: str = "",
                                 save_path: Optional[str] = None,
                                 show: bool = True):
    """
    Plot Position, Velocity, Acceleration, and Jerk for a trajectory.

    Args:
        trajectory: List of joint configurations
        dt: Time step between samples
        joint_idx: Which joint to plot (default 0)
        title_prefix: Prefix for plot title (e.g., "box", "cylinder")
        save_path: Optional path to save figure
        show: Whether to display plot
    """
    trajectory = np.array(trajectory)
    n_points = len(trajectory)

    # Extract single joint data
    if trajectory.ndim == 2:
        pos = trajectory[:, joint_idx]
    else:
        pos = trajectory

    times = np.arange(n_points) * dt

    # Compute derivatives
    vel = np.diff(pos) / dt
    acc = np.diff(vel) / dt
    jerk = np.diff(acc) / dt

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    title = f'{title_prefix} Trajectory Profile (Joint {joint_idx + 1})' if title_prefix else f'Trajectory Profile (Joint {joint_idx + 1})'
    fig.suptitle(title, fontsize=14)

    axs[0].plot(times, pos, 'b-', linewidth=1.5)
    axs[0].set_ylabel('Position [rad]')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(times[:-1], vel, 'r-', linewidth=1.5)
    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(times[:-2], acc, 'g-', linewidth=1.5)
    axs[2].set_ylabel('Acceleration [rad/s²]')
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(times[:-3], jerk, 'm-', linewidth=1.5)
    axs[3].set_ylabel('Jerk [rad/s³]')
    axs[3].set_xlabel('Time [s]')
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {final_path}")

    if show:
        plt.show()

    return fig


def plot_all_joints_derivatives(trajectory: List[np.ndarray],
                                 dt: float = 0.002,
                                 title_prefix: str = "",
                                 save_path: Optional[str] = None,
                                 show: bool = True):
    """
    Plot Position, Velocity, Acceleration, and Jerk for ALL joints.
    """
    trajectory = np.array(trajectory)
    n_points, n_joints = trajectory.shape
    times = np.arange(n_points) * dt

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    title = f'{title_prefix} - All Joints' if title_prefix else 'Trajectory (All Joints)'
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 1, n_joints))
    labels = [f'J{i+1}' for i in range(n_joints)]

    for j in range(n_joints):
        pos = trajectory[:, j]
        vel = np.diff(pos) / dt
        acc = np.diff(vel) / dt
        jerk = np.diff(acc) / dt

        axs[0].plot(times, pos, color=colors[j], label=labels[j], linewidth=1)
        axs[1].plot(times[:-1], vel, color=colors[j], linewidth=1)
        axs[2].plot(times[:-2], acc, color=colors[j], linewidth=1)
        axs[3].plot(times[:-3], jerk, color=colors[j], linewidth=1)

    axs[0].set_ylabel('Position [rad]')
    axs[0].legend(loc='upper right', ncol=n_joints)
    axs[1].set_ylabel('Velocity [rad/s]')
    axs[2].set_ylabel('Acceleration [rad/s²]')
    axs[3].set_ylabel('Jerk [rad/s³]')
    axs[3].set_xlabel('Time [s]')

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {final_path}")

    if show:
        plt.show()

    return fig


def plot_combined_trajectory(trajectories_dict: dict,
                              dt: float = 0.002,
                              planner_type: str = "rrt",
                              save_path: Optional[str] = None,
                              show: bool = True):
    """
    Plot combined trajectory for multiple objects in one figure.

    Args:
        trajectories_dict: {"box": trajectory, "cylinder": trajectory, ...}
        dt: Time step
        planner_type: Planner name for title
        save_path: Path to save
        show: Display plot
    """
    n_objects = len(trajectories_dict)
    if n_objects == 0:
        return None

    fig, axs = plt.subplots(4, n_objects, figsize=(6*n_objects, 12), squeeze=False)
    fig.suptitle(f'{planner_type.upper()} - Combined Trajectory Analysis', fontsize=16)

    colors = plt.cm.viridis(np.linspace(0, 1, 6))  # 6 joints
    labels = [f'J{i+1}' for i in range(6)]

    for col, (obj_name, trajectory) in enumerate(trajectories_dict.items()):
        if not trajectory:
            continue

        trajectory = np.array(trajectory)
        n_points, n_joints = trajectory.shape
        times = np.arange(n_points) * dt

        for j in range(n_joints):
            pos = trajectory[:, j]
            vel = np.diff(pos) / dt
            acc = np.diff(vel) / dt
            jerk = np.diff(acc) / dt

            axs[0, col].plot(times, pos, color=colors[j], label=labels[j] if col == 0 else "", linewidth=1)
            axs[1, col].plot(times[:-1], vel, color=colors[j], linewidth=1)
            axs[2, col].plot(times[:-2], acc, color=colors[j], linewidth=1)
            axs[3, col].plot(times[:-3], jerk, color=colors[j], linewidth=1)

        axs[0, col].set_title(obj_name.upper(), fontsize=12, fontweight='bold')

        for row in range(4):
            axs[row, col].grid(True, alpha=0.3)

    # Y-axis labels (only left column)
    axs[0, 0].set_ylabel('Position [rad]')
    axs[1, 0].set_ylabel('Velocity [rad/s]')
    axs[2, 0].set_ylabel('Acceleration [rad/s²]')
    axs[3, 0].set_ylabel('Jerk [rad/s³]')

    # X-axis label (bottom row)
    for col in range(n_objects):
        axs[3, col].set_xlabel('Time [s]')

    # Legend
    axs[0, 0].legend(loc='upper right', ncol=3, fontsize=8)

    plt.tight_layout()

    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {final_path}")

    if show:
        plt.show()

    return fig


def plot_via_points_comparison(via_points: List[np.ndarray],
                               trajectory: List[np.ndarray],
                               segment_times: List[float],
                               save_path: Optional[str] = None,
                               show: bool = True):
    """
    Plot trajectory showing via points marked.
    
    Args:
        via_points: Original via points
        trajectory: Interpolated trajectory
        segment_times: Time for each segment
    """
    trajectory = np.array(trajectory)
    via_points = np.array(via_points)
    n_joints = trajectory.shape[1]
    
    # Calculate time for each via point
    via_times = [0]
    for t in segment_times:
        via_times.append(via_times[-1] + t)
    
    total_time = sum(segment_times)
    times = np.linspace(0, total_time, len(trajectory))
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2*n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    for i in range(n_joints):
        axes[i].plot(times, trajectory[:, i], color=colors[i], linewidth=1.5)
        axes[i].scatter(via_times, via_points[:, i], color='red', s=50, zorder=5, marker='o')
        axes[i].set_ylabel(f'J{i+1} [rad]')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title(f'Trajectory with {len(via_points)} Via Points')
    
    plt.tight_layout()
    
    if save_path:
        final_path = _get_output_path(save_path)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {final_path}")
    
    if show:
        plt.show()
    
    return fig
