import os
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
    
    Note: This is a more aggressive approach and should be used carefully.
    """
    # Define C file functions
    c_stdout = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stdout')
    c_stderr = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stderr')
    
    # Save original stdout, stderr
    original_stdout = c_stdout.value
    original_stderr = c_stderr.value
    
    try:
        # Open null device
        null = ctypes.c_void_p(os.open(os.devnull, os.O_WRONLY))
        
        # Replace stdout and stderr
        c_stdout.value = null
        c_stderr.value = null
        
        yield
    finally:
        # Restore stdout and stderr
        c_stdout.value = original_stdout
        c_stderr.value = original_stderr
        
        # Close null device
        os.close(null.value)

# Example usage:
# import cv2
# with suppress_stdout_stderr():
#     cap = cv2.VideoCapture("video.mp4")
#     ret, frame = cap.read()
#     cap.release()
