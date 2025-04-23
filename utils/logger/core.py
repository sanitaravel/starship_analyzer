"""
Core logging functionality for the Starship Analyzer.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime
from typing import Union

# Import constants from the constants module
from .constants import (
    LOG_LEVELS, DEFAULT_LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
    LOG_DIR, LOG_FILE, _loggers, CURRENT_SESSION_LOG_FILE
)
from .system_info import log_system_info
from .formatters import ColoredFormatter

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
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
    
    # Create handlers if logger doesn't already have them
    if not logger.handlers:
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler - use current session log file if available, otherwise use default
        log_file = CURRENT_SESSION_LOG_FILE or LOG_FILE
        try:
            # Use RotatingFileHandler to prevent excessive file sizes
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.debug(f"Logger initialized with log file: {log_file}")
        except Exception as e:
            # Log to console if file logging fails
            print(f"Warning: Could not set up file logging to {log_file}: {e}")
            logger.error(f"Failed to set up file logging: {e}")
    
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
            if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
                logger.removeHandler(handler)
        
        # Add new file handler
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        try:
            file_handler = RotatingFileHandler(
                new_log_file,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logger.level)
            logger.addHandler(file_handler)
            logger.debug(f"Switched to log file: {new_log_file}")
        except Exception as e:
            logger.error(f"Failed to update file handler: {e}")

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
    
    # Create log directory again just to be sure
    try:
        os.makedirs(os.path.dirname(CURRENT_SESSION_LOG_FILE), exist_ok=True)
    except Exception as e:
        print(f"Error ensuring log directory exists: {e}")
    
    # Update existing loggers to use the new file
    if _loggers:
        _update_file_handlers(CURRENT_SESSION_LOG_FILE)
    
    # Create a root logger for session-wide messages
    root_logger = get_logger("starship_analyzer")
    
    # Add a session separator and header
    session_start = datetime.now().strftime(DATE_FORMAT)
    separator = f"\n{'='*80}\n"
    root_logger.info(f"{separator}NEW SESSION STARTED AT {session_start}{separator}")
    
    # Verify log file creation
    if os.path.exists(CURRENT_SESSION_LOG_FILE):
        root_logger.info(f"Log file created successfully at: {CURRENT_SESSION_LOG_FILE}")
    else:
        root_logger.error(f"Failed to create log file at: {CURRENT_SESSION_LOG_FILE}")
    
    # Log system information right after session start message
    log_system_info(root_logger)
    
    return root_logger
