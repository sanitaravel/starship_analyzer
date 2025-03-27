import cv2
import json
import os
import multiprocessing
import traceback
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from ocr import extract_data
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set the start method to 'spawn' to avoid CUDA re-initialization issues
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("'spawn' start method already set or couldn't be set")


def process_frame(frame_number: int, frame: np.ndarray, display_rois: bool, debug: bool, zero_time_met: bool) -> Dict:
    """
    Process a single frame and extract data.

    Args:
        frame_number (int): The frame number.
        frame (numpy.ndarray): The frame to process.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        zero_time_met (bool): Whether a frame with time 0:0:0 has been met.

    Returns:
        dict: A dictionary containing the extracted data for the frame.
    """
    try:
        superheavy_data, starship_data, time_data = extract_data(
            frame, display_rois=display_rois, debug=debug, zero_time_met=zero_time_met)
        frame_result = {
            "frame_number": frame_number,
            "superheavy": superheavy_data,
            "starship": starship_data,
            "time": time_data
        }
        return frame_result
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {str(e)}")
        # Return a minimal result to avoid breaking the pipeline
        return {
            "frame_number": frame_number,
            "superheavy": {},
            "starship": {},
            "time": None,
            "error": str(e)
        }


def process_batch(batch: List[int], video_path: str, display_rois: bool, debug: bool, zero_time_met: bool) -> List[Dict]:
    """
    Process a batch of frames and extract data.

    Args:
        batch (list): A list of frame numbers to process.
        video_path (str): The path to the video file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        zero_time_met (bool): Whether a frame with time 0:0:0 has been met.

    Returns:
        list: A list of dictionaries containing the extracted data for each frame.
    """
    try:
        # Import torch and set device in each process to avoid CUDA issues
        import torch
        if torch.cuda.is_available():
            # Use GPU 0 explicitly to avoid conflicts
            torch.cuda.set_device(0)
            # Empty CUDA cache at the start of each batch
            torch.cuda.empty_cache()
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        results = []
        for frame_number in batch:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame_result = process_frame(
                    frame_number, frame, display_rois, debug, zero_time_met)
                results.append(frame_result)
                if frame_result["time"] and frame_result["time"].get('hours') == 0 and frame_result["time"].get('minutes') == 0 and frame_result["time"].get('seconds') == 0:
                    zero_time_met = True
        cap.release()
        
        # Release GPU resources at the end of batch processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        print(traceback.format_exc())
        # Return a minimal result to avoid breaking the pipeline
        return [{"frame_number": fn, "error": str(e)} for fn in batch]


def validate_video(video_path: str) -> bool:
    """
    Validate that the video file exists and can be opened.

    Args:
        video_path (str): The path to the video file.

    Returns:
        bool: True if video is valid, False otherwise.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found at {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file at {video_path}")
        return False
    
    cap.release()
    logger.debug(f"Successfully validated video at {video_path}")
    return True


def get_video_properties(video_path: str, max_frames: Optional[int] = None) -> tuple:
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


def create_batches(frame_count: int, batch_size: int) -> List[List[int]]:
    """
    Create batches of frame numbers.

    Args:
        frame_count (int): The total number of frames.
        batch_size (int): The size of each batch.

    Returns:
        List[List[int]]: A list of batches, where each batch is a list of frame numbers.
    """
    frame_numbers = list(range(frame_count))
    return [frame_numbers[i:i + batch_size] for i in range(0, len(frame_numbers), batch_size)]


def process_video_frames(batches: List[List[int]], video_path: str, display_rois: bool, debug: bool) -> tuple:
    """
    Process all batches of frames.

    Args:
        batches (List[List[int]]): The batches of frame numbers.
        video_path (str): The path to the video file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        tuple: (results, zero_time_frame)
    """
    # Determine core count based on GPU availability
    available_cores = os.cpu_count() or 8
    try:
        import torch
        if torch.cuda.is_available():
            # If GPU-OCR is in use, use half of available cores
            num_cores = max(1, available_cores // 2)
            logger.info(f"GPU-OCR detected. Using {num_cores} worker processes (half of available {available_cores})")
        else:
            # If CPU-only, use all available cores
            num_cores = available_cores
            logger.info(f"CPU-OCR detected. Using all {num_cores} available cores")
    except ImportError:
        # If torch is not available, assume CPU-only
        num_cores = available_cores
        logger.info(f"Torch not available. Using all {num_cores} available cores")

    results = []
    zero_time_met = False
    zero_time_frame = None

    # Process batches with better error handling
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        
        # Submit all batch jobs
        for batch in batches:
            futures.append(executor.submit(process_batch, batch, video_path,
                                         display_rois, debug, zero_time_met))
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                # Check for zero time frame
                if zero_time_frame is None:
                    for frame_result in batch_results:
                        if frame_result.get("time") and frame_result["time"].get('hours') == 0 and \
                           frame_result["time"].get('minutes') == 0 and frame_result["time"].get('seconds') == 0:
                            zero_time_frame = frame_result["frame_number"]
                            logger.info(f"Found zero time frame at frame {zero_time_frame}")
                            break
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                logger.debug(traceback.format_exc())

    return results, zero_time_frame


def calculate_real_times(results: List[Dict], zero_time_frame: Optional[int], fps: float) -> List[Dict]:
    """
    Calculate real time for each frame.

    Args:
        results (List[Dict]): The processing results.
        zero_time_frame (Optional[int]): The frame number with time 0:0:0.
        fps (float): The frames per second of the video.

    Returns:
        List[Dict]: The updated results with real time calculations.
    """
    if zero_time_frame is None:
        return results
        
    for frame_result in results:
        frame_number = frame_result["frame_number"]
        if "error" in frame_result:
            continue
            
        if "time" in frame_result:
            seconds_from_zero = (frame_number - zero_time_frame) / fps
            frame_result["real_time_seconds"] = seconds_from_zero
            
            # Calculate time components
            hours = int(seconds_from_zero // 3600)
            minutes = int((seconds_from_zero % 3600) // 60)
            seconds = int(seconds_from_zero % 60)
            milliseconds = int((seconds_from_zero % 1) * 1000)
            
            frame_result["real_time"] = {
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
                "milliseconds": milliseconds
            }
    
    return results


def save_results(results: List[Dict], launch_number: int) -> None:
    """
    Save the processing results to a file.

    Args:
        results (List[Dict]): The processing results.
        launch_number (int): The launch number for saving results.
    """
    folder_name = os.path.join("results", f"launch_{launch_number}")
    os.makedirs(folder_name, exist_ok=True)

    result_path = os.path.join(folder_name, "results.json")
    try:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        
        # Try to save to a backup location
        backup_path = os.path.join("results", f"backup_results_{launch_number}.json")
        try:
            with open(backup_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to backup location: {backup_path}")
        except Exception as backup_error:
            logger.error(f"Failed to save results to backup location: {str(backup_error)}")


def iterate_through_frames(video_path: str, launch_number: int, display_rois: bool = False, debug: bool = False, max_frames: Optional[int] = None, batch_size: int = 10) -> None:
    """
    Iterate through all frames in a video and extract data.

    Args:
        video_path (str): The path to the video file.
        launch_number (int): The launch number for saving results.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        max_frames (int, optional): The maximum number of frames to process. Defaults to None.
        batch_size (int): The number of frames to process in each batch.
    """
    logger.info(f"Starting video processing for launch {launch_number}")
    logger.info(f"Video path: {video_path}, batch size: {batch_size}")
    
    if debug:
        logger.debug("Debug mode is enabled for video processing")
    
    # Ensure spawn method is used
    if multiprocessing.get_start_method() != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.debug("Set multiprocessing start method to 'spawn'")
        except RuntimeError:
            logger.warning("Could not set multiprocessing start method to 'spawn'")
            logger.warning("Processing will continue with current method, but may encounter CUDA issues")

    # Validate video file
    if not validate_video(video_path):
        logger.error("Video validation failed, aborting processing")
        return

    # Get video properties
    frame_count, fps = get_video_properties(video_path, max_frames)
    logger.info(f"Processing {frame_count} frames from video at {fps} fps")
    
    if debug:
        logger.debug(f"Video properties: {frame_count} frames, {fps} fps")
        
        # Get additional video properties for debugging
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.debug(f"Video resolution: {width}x{height}")
            video_codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((video_codec >> 8 * i) & 0xFF) for i in range(4)])
            logger.debug(f"Video codec: {codec_str}")
            cap.release()
    
    # Create batches
    batches = create_batches(frame_count, batch_size)
    logger.info(f"Created {len(batches)} batches of size {batch_size}")
    
    if debug and len(batches) > 0:
        logger.debug(f"First batch frame numbers: {batches[0]}")
        logger.debug(f"Last batch frame numbers: {batches[-1]}")

    # Process video frames
    logger.debug("Starting parallel video frame processing")
    results, zero_time_frame = process_video_frames(batches, video_path, display_rois, debug)
    logger.info(f"Processing complete. Analyzed {len(results)} frames successfully.")
    
    if debug:
        successful_frames = sum(1 for r in results if "error" not in r)
        failed_frames = sum(1 for r in results if "error" in r)
        logger.debug(f"Frame processing statistics: {successful_frames} successful, {failed_frames} failed")
    
    if zero_time_frame:
        logger.info(f"Zero time frame identified at frame {zero_time_frame}")
    else:
        logger.warning("No zero time frame was identified. Time calculations may be inaccurate.")
    
    # Calculate real time for each frame
    logger.debug("Starting real-time calculations")
    results = calculate_real_times(results, zero_time_frame, fps)
    logger.info(f"Time calculations complete.")

    # Save results
    logger.debug(f"Saving results to launch_{launch_number}")
    save_results(results, launch_number)
    logger.info(f"Video processing for launch {launch_number} completed successfully")
