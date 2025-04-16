"""
Video utility functions for handling video files.
"""
import os
import cv2
from utils.logger import get_logger

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
