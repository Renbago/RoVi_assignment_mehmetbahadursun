"""
Generate Report Figures - RRT vs P2P Comparison Plots

This script reads NPY trajectory files and generates visualization figures
for the robotics assignment report. Produces:
- Joint space comparison (overlay + side-by-side)
- Forward Kinematics (end-effector) comparison
- Combined multi-object plots

Usage:
    python generate_report_figures.py [--objects box cylinder t_block] [--show]

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk

Reference:
- roboticstoolbox FK: https://petercorke.github.io/robotics-toolbox-python/
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.trajectory_plotting import (
    plot_comparison_overlay,
    plot_comparison_sidebyside,
    plot_fk_comparison,
    plot_forward_kinematics,
    plot_all_joints_derivatives,
    plot_combined_trajectory
)
from utils.logger import ProjectLogger


# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
REPORT_DIR = os.path.join(OUTPUTS_DIR, 'report_figures')
JSON_DATA_DIR = os.path.join(OUTPUTS_DIR, 'trajectory_data')

OBJECTS = ['box', 'cylinder', 't_block']
DT = 0.002  # Time step for derivative calculations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_report_dir():
    """Create report figures directory if not exists."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        print(f"Created directory: {REPORT_DIR}")


def load_trajectory(obj_name: str, planner: str) -> np.ndarray:
    """
    Load trajectory from NPY file.

    Args:
        obj_name: Object name (box, cylinder, t_block)
        planner: Planner type (rrt, p2p)

    Returns:
        Trajectory array or None if not found
    """
    filename = f"{planner}_{obj_name}_trajectory.npy"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    if os.path.exists(filepath):
        traj = np.load(filepath)
        print(f"Loaded: {filename} ({len(traj)} points)")
        return traj
    else:
        print(f"NOT FOUND: {filename}")
        return None


def get_robot_rtb():
    """
    Get roboticstoolbox robot for FK calculations.

    Returns:
        roboticstoolbox UR5 robot or None
    """
    try:
        import roboticstoolbox as rtb
        robot = rtb.models.UR5()
        return robot
    except ImportError:
        print("WARNING: roboticstoolbox not available, FK plots will be skipped")
        return None


def load_trajectories_from_json(session_file: str = None) -> dict:
    """
    Load trajectories from JSON session file.

    Args:
        session_file: Specific session file (if None, uses latest)

    Returns:
        Dict with structure: {'planner_type': str, 'trajectories': {obj_name: np.array}}

    Reference:
    - JSON format saved by TrajectoryDataLogger
    """
    import json
    from glob import glob

    if session_file is None:
        # Find latest session file
        pattern = os.path.join(JSON_DATA_DIR, 'session_*.json')
        files = sorted(glob(pattern), reverse=True)
        if not files:
            print("No JSON session files found")
            return None
        session_file = files[0]
        print(f"Using latest session: {os.path.basename(session_file)}")

    if not os.path.exists(session_file):
        print(f"Session file not found: {session_file}")
        return None

    with open(session_file, 'r') as f:
        data = json.load(f)

    result = {
        'session_id': data.get('session_id', ''),
        'planner_type': data.get('planner_type', ''),
        'start_time': data.get('start_time', ''),
        'end_time': data.get('end_time', ''),
        'trajectories': {}
    }

    for obj_name, record in data.get('trajectories', {}).items():
        traj = np.array(record.get('trajectory', []))
        result['trajectories'][obj_name] = traj
        print(f"Loaded from JSON: {obj_name} ({len(traj)} points)")

    return result


def list_json_sessions() -> list:
    """
    List all available JSON session files.

    Returns:
        List of (filepath, session_id, planner_type, timestamp)
    """
    import json
    from glob import glob

    pattern = os.path.join(JSON_DATA_DIR, 'session_*.json')
    files = sorted(glob(pattern), reverse=True)

    sessions = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            sessions.append({
                'filepath': filepath,
                'session_id': data.get('session_id', ''),
                'planner_type': data.get('planner_type', ''),
                'start_time': data.get('start_time', ''),
                'objects': data.get('objects_processed', [])
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return sessions


# =============================================================================
# MAIN GENERATION FUNCTIONS
# =============================================================================

def generate_single_object_figures(obj_name: str, robot_rtb, show: bool = False):
    """
    Generate all figures for a single object.

    Args:
        obj_name: Object name
        robot_rtb: roboticstoolbox robot
        show: Display plots interactively
    """
    logger = ProjectLogger.get_instance()
    logger.info(f"Generating figures for: {obj_name.upper()}")

    # Load trajectories
    rrt_traj = load_trajectory(obj_name, 'rrt')
    p2p_traj = load_trajectory(obj_name, 'p2p')

    if rrt_traj is None and p2p_traj is None:
        logger.warning(f"No trajectories found for {obj_name}, skipping")
        return

    # --- Individual Planner Plots ---
    if rrt_traj is not None:
        plot_all_joints_derivatives(
            rrt_traj, dt=DT,
            title_prefix=f"{obj_name.upper()} RRT",
            save_path=os.path.join(REPORT_DIR, f"{obj_name}_rrt_joints.png"),
            show=show
        )

    if p2p_traj is not None:
        plot_all_joints_derivatives(
            p2p_traj, dt=DT,
            title_prefix=f"{obj_name.upper()} P2P",
            save_path=os.path.join(REPORT_DIR, f"{obj_name}_p2p_joints.png"),
            show=show
        )

    # --- Comparison Plots (require both) ---
    if rrt_traj is not None and p2p_traj is not None:
        # Overlay comparison
        plot_comparison_overlay(
            rrt_traj, p2p_traj, dt=DT,
            obj_name=obj_name,
            save_path=os.path.join(REPORT_DIR, f"{obj_name}_comparison_overlay.png"),
            show=show
        )

        # Side-by-side comparison
        plot_comparison_sidebyside(
            rrt_traj, p2p_traj, dt=DT,
            obj_name=obj_name,
            save_path=os.path.join(REPORT_DIR, f"{obj_name}_comparison_sidebyside.png"),
            show=show
        )

        # FK comparison (if robot available)
        if robot_rtb is not None:
            plot_fk_comparison(
                rrt_traj, p2p_traj, robot_rtb, dt=DT,
                obj_name=obj_name,
                save_path=os.path.join(REPORT_DIR, f"{obj_name}_fk_comparison.png"),
                show=show
            )

    # --- Individual FK Plots ---
    if robot_rtb is not None:
        if rrt_traj is not None:
            plot_forward_kinematics(
                rrt_traj, robot_rtb, dt=DT,
                title=f"{obj_name.upper()} RRT",
                save_path=os.path.join(REPORT_DIR, f"{obj_name}_rrt_fk.png"),
                show=show
            )

        if p2p_traj is not None:
            plot_forward_kinematics(
                p2p_traj, robot_rtb, dt=DT,
                title=f"{obj_name.upper()} P2P",
                save_path=os.path.join(REPORT_DIR, f"{obj_name}_p2p_fk.png"),
                show=show
            )

    logger.info(f"Completed figures for: {obj_name.upper()}")


def generate_combined_figures(objects: list, show: bool = False):
    """
    Generate combined multi-object comparison figures.

    Args:
        objects: List of object names
        show: Display plots interactively
    """
    logger = ProjectLogger.get_instance()
    logger.info("Generating combined multi-object figures")

    # Collect all trajectories
    rrt_trajectories = {}
    p2p_trajectories = {}

    for obj in objects:
        rrt_traj = load_trajectory(obj, 'rrt')
        p2p_traj = load_trajectory(obj, 'p2p')

        if rrt_traj is not None:
            rrt_trajectories[obj] = rrt_traj
        if p2p_traj is not None:
            p2p_trajectories[obj] = p2p_traj

    # Combined RRT plot
    if rrt_trajectories:
        plot_combined_trajectory(
            rrt_trajectories, dt=DT,
            planner_type="RRT",
            save_path=os.path.join(REPORT_DIR, "combined_rrt_all_objects.png"),
            show=show
        )

    # Combined P2P plot
    if p2p_trajectories:
        plot_combined_trajectory(
            p2p_trajectories, dt=DT,
            planner_type="P2P",
            save_path=os.path.join(REPORT_DIR, "combined_p2p_all_objects.png"),
            show=show
        )

    logger.info("Completed combined figures")


def print_trajectory_stats(objects: list):
    """
    Print statistics for all trajectories.

    Args:
        objects: List of object names
    """
    print("\n" + "=" * 60)
    print("TRAJECTORY STATISTICS")
    print("=" * 60)

    for obj in objects:
        print(f"\n{obj.upper()}:")

        rrt_traj = load_trajectory(obj, 'rrt')
        p2p_traj = load_trajectory(obj, 'p2p')

        if rrt_traj is not None:
            duration_rrt = len(rrt_traj) * DT
            print(f"  RRT: {len(rrt_traj)} points, {duration_rrt:.2f}s duration")

        if p2p_traj is not None:
            duration_p2p = len(p2p_traj) * DT
            print(f"  P2P: {len(p2p_traj)} points, {duration_p2p:.2f}s duration")

        if rrt_traj is not None and p2p_traj is not None:
            # Compare smoothness (average jerk magnitude)
            def avg_jerk(traj):
                vel = np.diff(traj, axis=0) / DT
                acc = np.diff(vel, axis=0) / DT
                jerk = np.diff(acc, axis=0) / DT
                return np.mean(np.abs(jerk))

            rrt_jerk = avg_jerk(rrt_traj)
            p2p_jerk = avg_jerk(p2p_traj)
            print(f"  Avg Jerk - RRT: {rrt_jerk:.2f}, P2P: {p2p_jerk:.2f} rad/s³")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for report figure generation."""
    parser = argparse.ArgumentParser(
        description='Generate report figures for RRT vs P2P comparison'
    )
    parser.add_argument(
        '--objects', nargs='+', default=OBJECTS,
        help=f'Objects to process (default: {OBJECTS})'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display plots interactively'
    )
    parser.add_argument(
        '--stats-only', action='store_true',
        help='Only print statistics, no plots'
    )
    parser.add_argument(
        '--list-sessions', action='store_true',
        help='List available JSON session files'
    )
    parser.add_argument(
        '--from-json', type=str, default=None,
        metavar='SESSION_FILE',
        help='Load trajectories from specific JSON session file'
    )

    args = parser.parse_args()

    # List sessions mode
    if args.list_sessions:
        print("\n" + "=" * 60)
        print("AVAILABLE JSON SESSIONS")
        print("=" * 60)
        sessions = list_json_sessions()
        if not sessions:
            print("No JSON session files found")
        else:
            for i, s in enumerate(sessions, 1):
                print(f"\n{i}. {s['session_id']}")
                print(f"   Planner: {s['planner_type']}")
                print(f"   Time: {s['start_time']}")
                print(f"   Objects: {', '.join(s['objects'])}")
                print(f"   File: {s['filepath']}")
        return

    # Initialize
    logger = ProjectLogger.get_instance()
    logger.info("=" * 50)
    logger.info("REPORT FIGURE GENERATION")
    logger.info("=" * 50)

    ensure_report_dir()

    # Print statistics
    print_trajectory_stats(args.objects)

    if args.stats_only:
        return

    # Get robot for FK
    robot_rtb = get_robot_rtb()

    # Generate figures for each object
    for obj in args.objects:
        generate_single_object_figures(obj, robot_rtb, show=args.show)

    # Generate combined figures
    generate_combined_figures(args.objects, show=args.show)

    logger.info("=" * 50)
    logger.info(f"All figures saved to: {REPORT_DIR}")
    logger.info("=" * 50)

    print(f"\nDone! Figures saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
