"""
Constants and configuration values for the logging system.
"""

import os
import logging

# Define log levels with their names for easy reference
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Default configuration
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log directory and file
# Change from relative to absolute path at project root level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'starship_analyzer.log')  # Default log file, will be overridden by session-specific logs

# Create the log directory if it doesn't exist
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.access(LOG_DIR, os.W_OK):
        print(f"Warning: No write access to log directory: {LOG_DIR}")
except Exception as e:
    print(f"Error creating log directory {LOG_DIR}: {e}")

# Store all loggers to avoid creating duplicates
_loggers = {}

# Store the current session's log file path
CURRENT_SESSION_LOG_FILE = None
