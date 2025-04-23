import os
import sys
import logging
import ctypes
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def suppress_ffmpeg_warnings():
    """
    Attempts to suppress FFmpeg warnings by setting environment variables.
    
    This should be called at the start of your application before importing OpenCV.
    """
    # Set environment variables to reduce FFmpeg verbosity
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "0"  # For OpenCV's FFmpeg integration
    os.environ["AV_LOG_LEVEL"] = "0"           # For PyAV/FFmpeg
    
    logger.info("FFmpeg warnings suppression enabled. This may not completely eliminate all warnings.")
    logger.info("Common H.264 warnings like 'co located POCs unavailable' are harmless and won't affect functionality.")

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to temporarily suppress stdout and stderr output.
    
    Works cross-platform on both Windows and Unix-like systems.
    """
    if sys.platform == 'win32':
        # Windows implementation
        try:
            # Define C file functions
            c_stdout = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stdout')
            c_stderr = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stderr')
            
            # Save original stdout, stderr
            original_stdout = c_stdout.value
            original_stderr = c_stderr.value
            
            # Open null device
            null = ctypes.c_void_p(os.open(os.devnull, os.O_WRONLY))
            
            # Replace stdout and stderr
            c_stdout.value = null
            c_stderr.value = null
            
            yield
        except (AttributeError, ValueError, OSError):
            # If the approach fails (msvcrt not accessible), fall back to a no-op
            logger.warning("Could not suppress stdout/stderr on this Windows system")
            yield
        finally:
            # Only restore and close if we successfully set them
            if 'null' in locals() and 'original_stdout' in locals():
                # Restore stdout and stderr
                c_stdout.value = original_stdout
                c_stderr.value = original_stderr
                
                # Close null device
                os.close(null.value)
    else:
        # Unix/Linux implementation
        try:
            # Get libc
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            # Save file descriptors
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
            
            # Save a copy of the original file descriptors
            original_stdout_fd = os.dup(stdout_fd)
            original_stderr_fd = os.dup(stderr_fd)
            
            # Open the null device
            null_fd = os.open(os.devnull, os.O_WRONLY)
            
            # Duplicate null device to stdout and stderr
            os.dup2(null_fd, stdout_fd)
            os.dup2(null_fd, stderr_fd)
            
            yield
        except (AttributeError, ValueError, OSError):
            # Fall back to a no-op
            logger.warning("Could not suppress stdout/stderr on this Unix/Linux system")
            yield
        finally:
            # Only restore if we successfully saved the original descriptors
            if 'original_stdout_fd' in locals():
                # Restore original file descriptors
                os.dup2(original_stdout_fd, stdout_fd)
                os.dup2(original_stderr_fd, stderr_fd)
                
                # Close duplicated descriptors
                os.close(original_stdout_fd)
                os.close(original_stderr_fd)
                
                # Close null device
                if 'null_fd' in locals():
                    os.close(null_fd)

# Example usage:
# import cv2
# with suppress_stdout_stderr():
#     cap = cv2.VideoCapture("video.mp4")
#     ret, frame = cap.read()
#     cap.release()
