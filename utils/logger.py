"""
Logging configuration for Starship Analyzer.
This module centralizes all logging configuration to ensure consistent logs across the application.

This is a compatibility layer that re-exports all functionality from the logger package.
"""

# Re-export all components from the logger package
from utils.logger import (
    # Constants
    LOG_LEVELS, DEFAULT_LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
    PROJECT_ROOT, LOG_DIR, LOG_FILE, CURRENT_SESSION_LOG_FILE,
    
    # Core logging functions
    get_logger, set_global_log_level, _update_file_handlers, start_new_session,
    
    # System info functions
    get_cpu_model, collect_system_info, write_system_info_section, log_system_info,
    
    # Formatters
    ColoredFormatter, COLORS
)

# For backward compatibility, ensure all previously exposed symbols are available
__all__ = [
    'LOG_LEVELS', 'DEFAULT_LOG_LEVEL', 'LOG_FORMAT', 'DATE_FORMAT',
    'PROJECT_ROOT', 'LOG_DIR', 'LOG_FILE', 'CURRENT_SESSION_LOG_FILE',
    'get_logger', 'set_global_log_level', '_update_file_handlers', 'start_new_session',
    'get_cpu_model', 'collect_system_info', 'write_system_info_section', 'log_system_info',
    'ColoredFormatter', 'COLORS'
]
