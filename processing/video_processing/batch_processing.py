"""
Functions for batch processing of video frames.
"""
import os
import cv2
import traceback
import multiprocessing
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utils.logger import get_logger
from .frame_processing import process_frame
from ocr import roi_manager as roi_manager

logger = get_logger(__name__)


def create_batches(frame_count: int, batch_size: int, sample_rate: int = 1) -> List[List[int]]:
    """
    Create batches of frame numbers with optional sampling.

    Args:
        frame_count (int): The total number of frames.
        batch_size (int): The size of each batch.
        sample_rate (int): The sampling rate (process every Nth frame).

    Returns:
        List[List[int]]: A list of batches, where each batch is a list of frame numbers.
    """
    # Generate frame numbers based on sample_rate
    frame_numbers = list(range(0, frame_count, sample_rate))
    return [frame_numbers[i:i + batch_size] for i in range(0, len(frame_numbers), batch_size)]


def process_batch(batch: List[int], video_path: str, display_rois: bool, debug: bool, zero_time_met: bool, progress_counter=None) -> List[Dict]:
    """
    Process a batch of frames and extract data.

    Args:
        batch (list): A list of frame numbers to process.
        video_path (str): The path to the video file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        zero_time_met (bool): Whether a frame with time 0:0:0 has been met.
        progress_counter (multiprocessing.Value, optional): Shared counter for progress tracking.

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
            
            # Update progress counter if provided - fixed to work with ValueProxy
            if progress_counter is not None:
                # No need to call get_lock() on ValueProxy objects
                progress_counter.value += 1
                    
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


def process_video_frames(batches: List[List[int]], video_path: str, display_rois: bool, debug: bool) -> Tuple[List[Dict], int]:
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
    
    # Calculate total number of frames to process
    total_frames = sum(len(batch) for batch in batches)
    
    # Create a shared counter for progress tracking
    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0)
    
    # Process batches with better error handling
    # Try to propagate the selected ROI config to worker processes so they
    # use the same ROIManager as the main process. We pass the config path
    # as an init arg to the ProcessPoolExecutor initializer which will call
    # `roi_manager.set_default_manager_config(config_path)` in each worker.
    try:
        default_mgr = roi_manager.get_default_manager()
        init_config_path = str(default_mgr.config_path) if default_mgr and default_mgr.config_path else None
    except Exception:
        init_config_path = None

    if init_config_path:
        logger.debug(f"Initializing worker processes with ROI config: {init_config_path}")
        executor_kwargs = {"max_workers": num_cores, "initializer": roi_manager.set_default_manager_config, "initargs": (init_config_path,)}
    else:
        executor_kwargs = {"max_workers": num_cores}

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = []
        
        # Submit all batch jobs with the shared counter
        for batch in batches:
            futures.append(executor.submit(process_batch, batch, video_path,
                                         display_rois, debug, zero_time_met, progress_counter))
        
        # Create a progress bar that tracks frame processing, not batch completion
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            last_counter_value = 0
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Update progress bar based on shared counter
                    current_count = progress_counter.value
                    pbar.update(current_count - last_counter_value)
                    last_counter_value = current_count
                    
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


def summarize_batch(batch_results, start_frame, end_frame):
    """
    Create a summary of the results for a batch of frames.
    
    Args:
        batch_results (list): List of results for each frame in the batch
        start_frame (int): Start frame of the batch
        end_frame (int): End frame of the batch
        
    Returns:
        dict: Batch summary
    """
    # Create a summary of the batch results
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": len(batch_results),
        "results": batch_results
        # Add other summary statistics as needed
    }
