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
import platform
import subprocess
import psutil
import re

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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
        try:
            # Use RotatingFileHandler to prevent excessive file sizes
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
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

def get_cpu_model() -> str:
    """
    Get a more readable CPU model name instead of the raw processor string.
    
    Returns:
        str: A user-friendly CPU model name
    """
    cpu_model = "Unknown CPU"
    
    try:
        if platform.system() == "Windows":
            try:
                # Try using WMI on Windows
                import wmi
                w = wmi.WMI()
                for processor in w.Win32_Processor():
                    return processor.Name.strip()
            except ImportError:
                # Fall back to registry query if WMI is not available
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                    winreg.CloseKey(key)
                    return cpu_model
                except Exception:
                    pass
        
        elif platform.system() == "Linux":
            # Parse /proc/cpuinfo on Linux
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':', 1)[1].strip()
            except Exception:
                pass
        
        elif platform.system() == "Darwin":  # macOS
            try:
                # Use sysctl on macOS
                output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                return output
            except Exception:
                pass
        
        # If all else fails, use platform module and try to clean up the output
        raw_info = platform.processor()
        
        # Try to extract a cleaner model name
        if "Intel" in raw_info:
            # Extract things like "Intel(R) Core(TM) i7-10700K" from the raw string
            match = re.search(r'(Intel.*?(?:Core|Xeon|Pentium|Celeron).*?(?:\d+-\w+|\w+\d+))', raw_info)
            if match:
                return match.group(1)
        elif "AMD" in raw_info:
            # Extract AMD processor model
            match = re.search(r'(AMD.*?(?:Ryzen|Athlon|Phenom|FX).*?(?:\d+\s+\w+|\w+\s+\d+))', raw_info)
            if match:
                return match.group(1)
        
        # If we can't extract a clean name, use the frequency info to make it more useful
        cpu_info = {
            "name": raw_info,
            "cores": psutil.cpu_count(logical=False),
            "frequency": f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "Unknown"
        }
        return f"{cpu_info['name']} ({cpu_info['cores']} cores @ {cpu_info['frequency']})"
        
    except Exception as e:
        # Last fallback: just return whatever processor string we have
        return platform.processor() or "Unknown CPU"

def collect_system_info() -> dict:
    """
    Collect important system information for debugging purposes.
    
    Returns:
        A dictionary containing system information
    """
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": get_cpu_model(),  # Use our improved function
        "architecture": platform.architecture(),
        "system": platform.system(),
        "release": platform.release(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }
    
    # Memory information
    mem = psutil.virtual_memory()
    system_info.update({
        "total_memory_gb": round(mem.total / (1024**3), 2),
        "available_memory_gb": round(mem.available / (1024**3), 2),
        "used_memory_gb": round(mem.used / (1024**3), 2),
        "memory_percent": mem.percent
    })
    
    # Try to get GPU information
    try:
        import torch
        system_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            
            # Get details for each GPU
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(i) / (1024**2), 2),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved(i) / (1024**2), 2)
                })
            system_info["gpus"] = gpu_info
    except ImportError:
        system_info["cuda_available"] = "Unknown (torch not installed)"
    
    # Try to get NVIDIA driver and detailed GPU information on Windows or Linux
    if system_info.get("system") in ["Windows", "Linux"]:
        try:
            # Try nvidia-smi for detailed GPU info
            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,name,temperature.gpu,memory.total,memory.free,memory.used,utilization.gpu", 
                 "--format=csv,noheader,nounits"], 
                shell=True if system_info["system"] == "Windows" else False
            ).decode().strip()
            
            # Parse nvidia-smi output and add to system_info
            gpu_detailed_info = []
            for idx, line in enumerate(nvidia_smi_output.split('\n')):
                values = [x.strip() for x in line.split(',')]
                if len(values) >= 7:
                    gpu_detailed_info.append({
                        "driver_version": values[0],
                        "name": values[1],
                        "temperature_c": values[2],
                        "memory_total_mb": values[3],
                        "memory_free_mb": values[4],
                        "memory_used_mb": values[5],
                        "utilization_percent": values[6]
                    })
            
            if gpu_detailed_info:
                system_info["gpu_detailed_info"] = gpu_detailed_info
                system_info["nvidia_driver"] = gpu_detailed_info[0]["driver_version"]
        except (subprocess.SubprocessError, FileNotFoundError, IndexError):
            system_info["nvidia_driver"] = "Not found or not installed"
    
    # Get OpenCV version if available (important for video processing)
    try:
        import cv2
        system_info["opencv_version"] = cv2.__version__
    except ImportError:
        system_info["opencv_version"] = "Not installed"
    
    # Get NumPy version (important for array processing)
    try:
        import numpy
        system_info["numpy_version"] = numpy.__version__
    except ImportError:
        system_info["numpy_version"] = "Not installed"
    
    # Check for EasyOCR (critical component)
    try:
        import easyocr
        system_info["easyocr_version"] = getattr(easyocr, "__version__", "Unknown version")
    except ImportError:
        system_info["easyocr_version"] = "Not installed"
    
    return system_info

def write_system_info_section(log_file: str, system_info: dict) -> None:
    """
    Write system information to a separate section in the log file.
    
    Args:
        log_file: Path to the log file
        system_info: Dictionary containing system information
    """
    try:
        with open(log_file, 'a') as f:
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("SYSTEM INFORMATION\n")
            f.write("="*80 + "\n")
            
            # Write hardware information
            f.write("\nHARDWARE:\n")
            f.write("---------\n")
            hardware_keys = ["processor", "cpu_count_physical", "cpu_count_logical", 
                            "total_memory_gb", "available_memory_gb", "used_memory_gb"]
            for key in hardware_keys:
                if key in system_info:
                    f.write(f"{key.replace('_', ' ').title()}: {system_info[key]}\n")
            
            # Write GPU information if available
            if "gpus" in system_info and system_info["gpus"]:
                f.write("\nGPU:\n")
                f.write("----\n")
                for i, gpu in enumerate(system_info["gpus"]):
                    f.write(f"GPU {i+1}: {gpu['name']}\n")
                    if "memory_total_mb" in gpu:
                        f.write(f"  Memory: {gpu['memory_total_mb']} MB\n")
                    if "memory_allocated_mb" in gpu:
                        f.write(f"  Allocated: {gpu['memory_allocated_mb']} MB\n")
            elif "gpu_detailed_info" in system_info and system_info["gpu_detailed_info"]:
                f.write("\nGPU:\n")
                f.write("----\n")
                for i, gpu in enumerate(system_info["gpu_detailed_info"]):
                    f.write(f"GPU {i+1}: {gpu['name']}\n")
                    f.write(f"  Driver: {gpu['driver_version']}\n")
                    f.write(f"  Memory: {gpu['memory_total_mb']} MB (Free: {gpu['memory_free_mb']} MB)\n")
                    f.write(f"  Temperature: {gpu['temperature_c']}°C\n")
                    f.write(f"  Utilization: {gpu['utilization_percent']}%\n")
                    
            # Write software information
            f.write("\nSOFTWARE:\n")
            f.write("---------\n")
            f.write(f"OS: {system_info.get('system', 'Unknown')} {system_info.get('release', '')}\n")
            f.write(f"Platform: {system_info.get('platform', 'Unknown')}\n")
            f.write(f"Python: {system_info.get('python_version', 'Unknown')}\n")
            
            # Write other dependencies
            if "opencv_version" in system_info:
                f.write(f"OpenCV: {system_info['opencv_version']}\n")
            if "numpy_version" in system_info:
                f.write(f"NumPy: {system_info['numpy_version']}\n")
            if "easyocr_version" in system_info:
                f.write(f"EasyOCR: {system_info['easyocr_version']}\n")
            if "cuda_available" in system_info:
                f.write(f"CUDA: {'Yes - ' + system_info.get('cuda_version', 'Unknown version') if system_info['cuda_available'] else 'No'}\n")
            
            f.write("\n")
            f.write("="*80 + "\n\n")
    except Exception as e:
        # If we can't write to the log file directly, log the error through the regular logging system
        logger = get_logger("system_info")
        logger.error(f"Failed to write system information section: {str(e)}")

def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging purposes.
    
    Args:
        logger: The logger to use for logging system information
    """
    try:
        system_info = collect_system_info()
        
        # Write full system info to a separate section in the log file
        if logger.handlers:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    write_system_info_section(handler.baseFilename, system_info)
                    break
    except Exception as e:
        logger.error(f"Failed to collect system information: {str(e)}")

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
