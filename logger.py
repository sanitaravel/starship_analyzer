"""
Logging configuration for Starship Analyzer.
This module centralizes all logging configuration to ensure consistent logs across the application.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime
from typing import Union

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
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'starship_analyzer.log')  # Default log file, will be overridden by session-specific logs

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Store all loggers to avoid creating duplicates
_loggers = {}

# Store the current session's log file path
CURRENT_SESSION_LOG_FILE = None

def get_logger(name: str, level: int = None) -> logging.Logger:
    """
    Get a logger configured with appropriate handlers.
    
    Args:
        name: The name of the logger (typically __name__ of the module)
        level: Optional specific log level for this logger
        
    Returns:
        A configured logger instance
    """
    global _loggers
    
    # Return existing logger if already created
    if name in _loggers:
        return _loggers[name]
    
    # Create a new logger
    logger = logging.getLogger(name)
    
    # Set level (use specified level or default)
    logger.setLevel(level or DEFAULT_LOG_LEVEL)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Create handlers if logger doesn't already have them
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler - use current session log file if available, otherwise use default
        log_file = CURRENT_SESSION_LOG_FILE or LOG_FILE
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store logger for reuse
    _loggers[name] = logger
    return logger

def set_global_log_level(level: Union[int, str]) -> None:
    """
    Set the log level for all loggers.
    
    Args:
        level: The log level (either a string name or integer value)
    """
    global DEFAULT_LOG_LEVEL
    
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), DEFAULT_LOG_LEVEL)
    
    # Update the default level
    DEFAULT_LOG_LEVEL = level
    
    # Update all existing loggers
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    
    # Update root logger as well
    logging.getLogger().setLevel(level)

def _update_file_handlers(new_log_file: str) -> None:
    """
    Update all existing loggers to use a new log file.
    
    Args:
        new_log_file: Path to the new log file
    """
    for logger in _loggers.values():
        # Remove existing file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        
        # Add new file handler
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler = logging.FileHandler(new_log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)

def start_new_session() -> logging.Logger:
    """
    Start a new logging session with a new log file.
    Call this at the beginning of program execution.
    
    Returns:
        Root logger for the application
    """
    global CURRENT_SESSION_LOG_FILE
    
    # Generate a new log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_SESSION_LOG_FILE = os.path.join(LOG_DIR, f"starship_analyzer_{timestamp}.log")
    
    # Update existing loggers to use the new file
    if _loggers:
        _update_file_handlers(CURRENT_SESSION_LOG_FILE)
    
    # Create a root logger for session-wide messages
    root_logger = get_logger("starship_analyzer")
    
    # Add a session separator
    session_start = datetime.now().strftime(DATE_FORMAT)
    separator = f"\n{'='*80}\n"
    root_logger.info(f"{separator}NEW SESSION STARTED AT {session_start}{separator}")
    
    return root_logger
