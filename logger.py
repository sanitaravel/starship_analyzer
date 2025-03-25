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
LOG_FILE = os.path.join(LOG_DIR, 'starship_analyzer.log')

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Store all loggers to avoid creating duplicates
_loggers = {}

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
        
        # File handler with rotation (10MB max size, keep 5 backup files)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5
        )
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
    global DEFAULT_LOG_LEVEL  # Move the global declaration to the beginning of the function
    
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

def start_new_session() -> None:
    """
    Start a new logging session, adding a session separator to the log file.
    Call this at the beginning of program execution.
    """
    # Create a root logger for session-wide messages
    root_logger = get_logger("starship_analyzer")
    
    # Add a session separator
    session_start = datetime.now().strftime(DATE_FORMAT)
    separator = f"\n{'='*80}\n"
    root_logger.info(f"{separator}NEW SESSION STARTED AT {session_start}{separator}")
    
    return root_logger
