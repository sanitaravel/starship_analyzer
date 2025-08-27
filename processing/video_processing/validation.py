"""
Video validation and property extraction functions.
"""
import os
import cv2
from typing import Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_video(video_path: str) -> bool:
    """
    Validate that the video file exists, is an MP4, and can be properly opened and read.

    Args:
        video_path (str): The path to the video file.

    Returns:
        bool: True if video is valid, False otherwise.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found at {video_path}")
        return False
        
    # Check file extension
    _, extension = os.path.splitext(video_path)
    if extension.lower() != '.mp4':
        logger.warning(f"File {video_path} is not an MP4 file. It might not be properly supported.")
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        logger.warning(f"Video file is suspiciously small ({file_size/1024/1024:.2f}MB). It might be corrupted.")
    
    # Note about H.264 warnings
    logger.info("Note: You may see H.264 warnings like 'co located POCs unavailable' or 'mmco: unref short failure'. "
               "These are harmless and can be safely ignored.")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video file at {video_path}")
        return False
    
    # Check video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        logger.error(f"Video contains no frames")
        cap.release()
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning(f"Invalid FPS value: {fps}")
    
    # Reset position to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Check video codec
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    logger.debug(f"Video codec: {codec_str}")
    
    # Get video duration
    duration_seconds = frame_count / fps if fps > 0 else 0
    logger.debug(f"Video duration: {duration_seconds:.2f} seconds (~{duration_seconds/60:.2f} minutes)")
    
    logger.info(f"Successfully validated video at {video_path}")
    cap.release()
    return True


def get_video_properties(video_path: str, max_frames: Optional[int] = None) -> Tuple[int, float]:
    """
    Get video properties like frame count and FPS.

    Args:
        video_path (str): The path to the video file.
        max_frames (int, optional): The maximum number of frames to process.

    Returns:
        tuple: (frame_count, fps)
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
        
    return frame_count, fps
