"""
Custom formatters for the Starship Analyzer logging system.
"""

import logging
import platform

# ANSI color codes for terminal output
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
}

class ColoredFormatter(logging.Formatter):
    """
    A formatter that adds colors to log messages based on their level.
    Colors work in most Unix terminals and Windows 10+ terminals.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: COLORS['BLUE'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['BOLD'] + COLORS['RED'],
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', use_colors=True):
        super().__init__(fmt, datefmt, style)
        # Disable colors on Windows versions that don't support ANSI
        if platform.system() == 'Windows':
            try:
                # Extract only numeric characters from the version string
                version_str = ''.join(c for c in platform.release() if c.isdigit())
                if version_str and int(version_str) < 10:
                    use_colors = False
            except (ValueError, TypeError):
                # If parsing fails (e.g., with "2022Server"), assume it's a newer Windows
                # version that supports ANSI colors
                pass
        self.use_colors = use_colors
    
    def format(self, record):
        # Get the original formatted message
        message = super().format(record)
        
        if not self.use_colors:
            return message
            
        # Add color based on log level
        level_color = self.LEVEL_COLORS.get(record.levelno, COLORS['RESET'])
        
        # Color the level name but leave the rest of the message as is
        level_start = message.find(record.levelname)
        if level_start >= 0:
            level_end = level_start + len(record.levelname)
            colored_level = level_color + record.levelname + COLORS['RESET']
            message = message[:level_start] + colored_level + message[level_end:]
            return message
        
        # If we can't find the level name in the message, color the whole message
        return level_color + message + COLORS['RESET']
