"""
This module provides a single logger instance used throughout the project.
Configuration is loaded from config.yaml debug section.

Usage:
    from utils.logger import ProjectLogger

    # In class __init__:
    self.logger = ProjectLogger.get_instance()

    # In methods:
    self.logger.info("Message")
    if self.logger.debug_enabled:
        self.logger.debug("Detailed debug info")

Reference:
- Python logging: https://docs.python.org/3/library/logging.html
- Singleton pattern: https://refactoring.guru/design-patterns/singleton/python
- ANSI escape codes: https://en.wikipedia.org/wiki/ANSI_escape_code

Author: Mehmet Baha Dursun
Email: medur25@student.sdu.dk
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Optional, Any, Dict, List, Union

# ANSI color codes for console output
# Reference: https://en.wikipedia.org/wiki/ANSI_escape_code
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m',
}

# Log level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
    'ALL': logging.DEBUG,  # ALL = show everything from DEBUG up
}


class SelectiveLevelFilter(logging.Filter):
    """
    Filter that only allows specific log levels.

    Used when log_level is a list like ["INFO", "WARNING", "ERROR"].
    Only messages with levels in the list will be shown.

    Reference: https://docs.python.org/3/library/logging.html#filter-objects
    """

    def __init__(self, allowed_levels: list):
        """
        Args:
            allowed_levels: List of level names to allow (e.g., ["INFO", "ERROR"])
        """
        super().__init__()
        self.allowed_levels = {LOG_LEVELS.get(lvl.upper(), logging.INFO) for lvl in allowed_levels}

    def filter(self, record):
        """Only allow records with levels in allowed_levels."""
        return record.levelno in self.allowed_levels


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output based on log level.

    Colors:
    - DEBUG: Cyan (detailed technical info)
    - INFO: Green (normal operation)
    - WARNING: Yellow (potential issues)
    - ERROR: Red (operation failures)
    """

    LEVEL_COLORS = {
        logging.DEBUG: COLORS['CYAN'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['RED'] + COLORS['BOLD'],
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, COLORS['END'])
        formatted = super().format(record)
        colored_level = f"{color}{record.levelname}{COLORS['END']}"
        formatted = formatted.replace(record.levelname, colored_level, 1)
        return formatted


class ProjectLogger:
    """
    Singleton logger class for centralized project logging.

    Attributes:
        debug_enabled: Whether DEBUG level logging is active
        log_file_path: Path to current log file (or None)

    Example:
        logger = ProjectLogger.get_instance()
        logger.info("Planning started")
        if logger.debug_enabled:
            logger.debug("Detailed state info")
    """

    _instance: Optional['ProjectLogger'] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[Dict] = None):
        """Ensure only one instance exists (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize logger with configuration.

        Args:
            config: Optional config dict. If None, loads from config.yaml
        """
        # Prevent re-initialization
        if ProjectLogger._initialized:
            return

        # Load config
        if config is None:
            from utils.mujoco_utils import load_config
            config = load_config()

        self.config = config
        self._setup_from_config()
        ProjectLogger._initialized = True

    def _setup_from_config(self):
        """Setup logger based on config.yaml debug section."""
        debug_config = self.config.get('debug', {})

        # Get settings from config
        self.debug_enabled = debug_config.get('enabled', True)
        log_to_file = debug_config.get('log_to_file', True)
        log_level_config = debug_config.get('log_level', 'INFO')

        # Create internal logger
        self._logger = logging.getLogger('rovi_project')
        self._logger.setLevel(logging.DEBUG)  # Capture all, filter in handlers
        self._logger.handlers = []  # Clear existing handlers

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = ColoredFormatter(
            fmt='%(levelname)-8s %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        # Handle log_level options:
        # 1. "ALL" - show all levels (DEBUG and above)
        # 2. Single string like "INFO" - threshold based
        # 3. List like ["INFO", "WARNING", "ERROR"] - selective filter
        self.selective_filter = None

        if isinstance(log_level_config, list):
            # Multiple levels specified - use selective filter
            self.log_level = logging.DEBUG  # Accept all, filter below
            self.selective_filter = SelectiveLevelFilter(log_level_config)
            console_handler.addFilter(self.selective_filter)
            console_handler.setLevel(logging.DEBUG)
        elif isinstance(log_level_config, str):
            if log_level_config.upper() == 'ALL':
                # ALL = show everything from DEBUG up
                self.log_level = logging.DEBUG
            else:
                # Single level = threshold based
                self.log_level = LOG_LEVELS.get(log_level_config.upper(), logging.INFO)
            console_handler.setLevel(self.log_level)
        else:
            # Default to INFO
            self.log_level = logging.INFO
            console_handler.setLevel(self.log_level)

        self._logger.addHandler(console_handler)

        # File handler
        self.log_file_path = None
        if log_to_file:
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup file logging with timestamp."""
        try:
            # Get output directory
            output_dir = self.config.get('output', {}).get('directory', 'outputs')
            project_root = Path(__file__).parent.parent
            log_dir = project_root / output_dir / 'planning_logs'
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file_path = log_dir / f'planning_{timestamp}.log'

            file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_format = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self._logger.addHandler(file_handler)

        except Exception as e:
            self._logger.warning(f"Could not create log file: {e}")

    @classmethod
    def get_instance(cls, config: Optional[Dict] = None) -> 'ProjectLogger':
        """
        Get the singleton logger instance.

        Args:
            config: Optional config dict (only used on first call)

        Returns:
            ProjectLogger singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing purposes only)."""
        if cls._instance is not None:
            cls._instance.cleanup()
        cls._instance = None
        cls._initialized = False

    def cleanup(self):
        """
        Clean up logger resources (file handlers, etc.)

        Call this on application shutdown for graceful cleanup.
        """
        if hasattr(self, '_logger'):
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self.cleanup()

    # =========================================================================
    # LOGGING METHODS (delegate to internal logger)
    # =========================================================================

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message (only if debug_enabled)."""
        if self.debug_enabled:
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)

    # =========================================================================
    # STRUCTURED LOGGING HELPERS
    # =========================================================================

    def log_planning_start(self, planner_type: str, start_q: np.ndarray,
                           goal_info: Any, target_object: str = None,
                           held_object: str = None):
        """
        Log the start of a planning operation.

        Args:
            planner_type: "rrt", "prm", "p2p", "simple_ik", "goal_region"
            start_q: Starting joint configuration
            goal_info: Goal configuration or SE3 frame
            target_object: Object being approached (collisions ignored)
            held_object: Object being held during transport
        """
        self.info("=" * 60)
        self.info(f"PLANNING START: {planner_type.upper()}")
        self.info("-" * 60)

        if isinstance(start_q, np.ndarray):
            start_deg = np.degrees(start_q)
            self.debug(f"  Start Q (rad): [{', '.join(f'{q:.4f}' for q in start_q)}]")
            self.info(f"  Start Q (deg): [{', '.join(f'{q:.1f}' for q in start_deg)}]")

        if hasattr(goal_info, 't'):  # SE3 frame
            self.info(f"  Goal Position: [{goal_info.t[0]:.4f}, {goal_info.t[1]:.4f}, {goal_info.t[2]:.4f}]")
        elif isinstance(goal_info, np.ndarray):
            goal_deg = np.degrees(goal_info)
            self.info(f"  Goal Q (deg): [{', '.join(f'{q:.1f}' for q in goal_deg)}]")

        if target_object:
            self.info(f"  Target Object: {target_object} (collisions ignored)")
        if held_object:
            self.info(f"  Held Object: {held_object} (collision checked)")

        self.info("-" * 60)

    def log_planning_result(self, success: bool, path_length: float = None,
                            state_count: int = None, planner_type: str = "rrt"):
        """
        Log the result of a planning operation.

        Args:
            success: Whether planning succeeded
            path_length: OMPL path length (if successful)
            state_count: Number of states in path (if successful)
            planner_type: Planner used
        """
        if success:
            self.info(f"PLANNING SUCCESS: {planner_type.upper()}")
            if state_count is not None:
                self.info(f"  Path States: {state_count}")
            if path_length is not None:
                self.info(f"  Path Length: {path_length:.4f} rad")
            self.info("=" * 60)
        else:
            self.error(f"PLANNING FAILED: {planner_type.upper()}")
            self.error("  No valid path found within timeout")
            self.info("=" * 60)

    def log_ik_sampling(self, n_samples: int, converged: int,
                        collision_failed: int, valid_count: int,
                        best_distance: float = None):
        """
        Log IK sampling results.

        Args:
            n_samples: Total IK attempts
            converged: How many IK solutions converged
            collision_failed: How many failed collision check
            valid_count: Number of valid, unique solutions
            best_distance: Distance to chosen solution (if any)
        """
        self.info("IK SAMPLING RESULTS:")
        self.debug(f"  Total Samples: {n_samples}")
        self.info(f"  IK Converged: {converged}/{n_samples} ({100*converged/n_samples:.1f}%)")

        if converged > 0:
            collision_pct = 100 * collision_failed / converged
            self.info(f"  Collision Failed: {collision_failed}/{converged} ({collision_pct:.1f}%)")

        if valid_count > 0:
            self.info(f"  Valid & Unique: {valid_count}")
            if best_distance is not None:
                self.info(f"  Selected (closest): distance = {best_distance:.4f} rad")
        else:
            self.warning("  No valid IK solutions found!")

    def log_manipulation_phase(self, phase: str, obj_name: str,
                                details: Union[Dict, str] = None):
        """
        Log manipulation phase transitions.

        Args:
            phase: "approach", "grasp", "lift", "transport", "place", "release", "retreat"
            obj_name: Object being manipulated
            details: Additional details (position, frame, etc.)
        """
        phase_upper = phase.upper()
        self.info("")
        self.info(f"PHASE: {phase_upper} [{obj_name}]")

        if details:
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, np.ndarray):
                        self.info(f"    {key}: [{', '.join(f'{v:.4f}' for v in value)}]")
                    else:
                        self.info(f"    {key}: {value}")
            else:
                self.info(f"    {details}")

    def log_grasp_offset(self, obj_name: str, offset_translation: Any,
                          computed_vs_config: str = "runtime"):
        """
        Log grasp offset computation.

        Args:
            obj_name: Object name
            offset_translation: Translation component of grasp offset
            computed_vs_config: "runtime" (measured after grasp) or "config" (from YAML)
        """
        if computed_vs_config == "runtime":
            self.info(f"GRASP OFFSET (runtime measured) for {obj_name}:")
        else:
            self.info(f"GRASP OFFSET (from config) for {obj_name}:")

        if hasattr(offset_translation, '__iter__'):
            self.info(f"  Translation: [{offset_translation[0]:.4f}, {offset_translation[1]:.4f}, {offset_translation[2]:.4f}]")
        else:
            self.info(f"  Offset: {offset_translation}")

    def log_session_summary(self, objects_moved: List[str], total_time: float = None,
                            success_count: int = None, fail_count: int = None):
        """
        Log end-of-session summary.

        Args:
            objects_moved: List of object names processed
            total_time: Total execution time (seconds)
            success_count: Number of successful operations
            fail_count: Number of failed operations
        """
        self.info("")
        self.info("=" * 60)
        self.info("SESSION SUMMARY")
        self.info("=" * 60)
        self.info(f"  Objects Processed: {', '.join(objects_moved)}")

        if success_count is not None:
            self.info(f"  Successful: {success_count}")
        if fail_count is not None:
            self.info(f"  Failed: {fail_count}")
        if total_time is not None:
            self.info(f"  Total Time: {total_time:.2f} seconds")

        if self.log_file_path:
            self.info(f"  Full Log: {self.log_file_path}")

        self.info("=" * 60)


# Convenience function for backward compatibility
def get_logger() -> ProjectLogger:
    """
    Get the project logger instance.

    Returns:
        ProjectLogger singleton instance

    Note:
        This is provided for backward compatibility.
        Prefer using ProjectLogger.get_instance() directly.
    """
    return ProjectLogger.get_instance()
