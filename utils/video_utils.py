"""
Video utility functions for handling video files.
"""
import os
import cv2
from utils.logger import get_logger
import logging
import subprocess
from typing import Tuple, Optional, List

logger = get_logger(__name__)

def get_video_files_from_flight_recordings():
    """
    Get a list of video files from the flight_recordings folder.
    
    Returns:
        list: List of tuples (filename, path) for video files
    """
    flight_recordings_folder = os.path.join('.', 'flight_recordings')
    video_files = []
    
    if (os.path.exists(flight_recordings_folder)):
        for root, _, files in os.walk(flight_recordings_folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    relative_path = os.path.relpath(os.path.join(root, file), '.')
                    video_files.append((file, relative_path))
    
    if not video_files:
        print("No video files found in flight_recordings folder.")
    
    return video_files

def display_video_info(video_path):
    """
    Display information about the selected video.
    
    Args:
        video_path (str): Path to the video file
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration_sec = frame_count / fps if fps > 0 else 0
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        seconds = int(duration_sec % 60)
        
        # Get codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        # Display information
        print("\n----- Video Information -----")
        print(f"Resolution: {width}x{height}")
        print(f"Frame Rate: {fps:.2f} fps")
        print(f"Total Frames: {frame_count}")
        print(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Codec: {codec}")
        print("----------------------------\n")
        
        logger.debug(f"Video info displayed for {video_path}: {width}x{height}, {fps:.2f} fps, {frame_count} frames")
        
    except Exception as e:
        print(f"Error getting video information: {str(e)}")
        logger.error(f"Error displaying video info: {str(e)}")

def get_video_info(video_path: str) -> dict:
    """
    Get detailed video information using FFprobe (if available)
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Dictionary containing video information
    """
    info = {
        "path": video_path,
        "exists": os.path.exists(video_path),
        "size_mb": os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0,
    }
    
    # Try using OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            info["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()
        else:
            info["opencv_open_failed"] = True
    except Exception as e:
        logger.error(f"Error getting video info with OpenCV: {str(e)}")
        info["opencv_error"] = str(e)
    
    # Try using FFprobe if available
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
             '-show_format', '-show_streams', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            import json
            ffprobe_info = json.loads(result.stdout)
            info["ffprobe"] = ffprobe_info
        else:
            info["ffprobe_error"] = result.stderr
    except Exception as e:
        # FFprobe might not be installed, which is okay
        logger.debug(f"FFprobe not available: {str(e)}")
    
    return info


def get_video_fps(video_path: str) -> Optional[float]:
    """Return the video's frames-per-second as a float, or None if it can't be read."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception as e:
        logger.debug(f"Could not get video fps for {video_path}: {e}")
    return None

def try_alternative_decoder(video_path: str) -> bool:
    """
    Try to use an alternative decoder for problematic H.264 videos
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        bool: True if alternative decoder works, False otherwise
    """
    try:
        # Try with a different API backend
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.warning("Alternative decoder (FFMPEG) failed to open the video")
            return False
            
        # Test reading a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            logger.info("Alternative decoder successfully read a frame")
            return True
        else:
            logger.warning("Alternative decoder failed to read frames")
            return False
    except Exception as e:
        logger.error(f"Error using alternative decoder: {str(e)}")
        return False
