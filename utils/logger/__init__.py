"""
Logging configuration for Starship Analyzer.
This module centralizes all logging configuration to ensure consistent logs across the application.
"""

# Import and re-export all necessary components
from .constants import (
    LOG_LEVELS, DEFAULT_LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, 
    PROJECT_ROOT, LOG_DIR, LOG_FILE, CURRENT_SESSION_LOG_FILE
)
from .core import (
    get_logger, set_global_log_level, _update_file_handlers, start_new_session
)
from .system_info import (
    get_cpu_model, collect_system_info, write_system_info_section, log_system_info
)
from .formatters import ColoredFormatter, COLORS

__all__ = [
    'LOG_LEVELS', 'DEFAULT_LOG_LEVEL', 'LOG_FORMAT', 'DATE_FORMAT',
    'PROJECT_ROOT', 'LOG_DIR', 'LOG_FILE', 'CURRENT_SESSION_LOG_FILE',
    'get_logger', 'set_global_log_level', 'start_new_session',
    'get_cpu_model', 'collect_system_info', 'log_system_info',
    'ColoredFormatter', 'COLORS',
]
