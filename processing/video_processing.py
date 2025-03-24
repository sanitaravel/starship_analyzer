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

# Set the start method to 'spawn' to avoid CUDA re-initialization issues
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Warning: 'spawn' start method already set or couldn't be set")


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
        print(f"Error processing frame {frame_number}: {str(e)}")
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
    # Ensure spawn method is used
    if multiprocessing.get_start_method() != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            print("Warning: Could not set multiprocessing start method to 'spawn'")
            print("Processing will continue with current method, but may encounter CUDA issues")

    # Verify video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Open video file and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video file at {video_path}")
        return
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    results: List[Dict] = []
    zero_time_frame: Optional[int] = None
    zero_time_met = False

    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    print(f"Processing {frame_count} frames from video at {fps} fps")
    
    # Create batches
    frame_numbers = list(range(frame_count))
    batches = [frame_numbers[i:i + batch_size]
               for i in range(0, len(frame_numbers), batch_size)]
    
    print(f"Created {len(batches)} batches of size {batch_size}")

    # Limit the number of workers based on available CPU cores
    # Using fewer workers can help avoid CUDA memory issues
    num_cores = min(os.cpu_count() or 4, 4)  # Limit to 4 cores max to avoid CUDA issues
    print(f"Using {num_cores} worker processes for parallel processing")

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
                if not zero_time_met:
                    for frame_result in batch_results:
                        if frame_result.get("time") and frame_result["time"].get('hours') == 0 and \
                           frame_result["time"].get('minutes') == 0 and frame_result["time"].get('seconds') == 0:
                            zero_time_frame = frame_result["frame_number"]
                            zero_time_met = True
                            print(f"Found zero time frame at frame {zero_time_frame}")
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print(traceback.format_exc())

    print(f"Processing complete. Analyzed {len(results)} frames successfully.")
    
    # Calculate real time for each frame
    for frame_result in results:
        frame_number = frame_result["frame_number"]
        if "error" in frame_result:
            continue
            
        if zero_time_frame is not None and "time" in frame_result:
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

    print(f"Time calculations complete.")

    # Save results
    folder_name = os.path.join("results", f"launch_{launch_number}")
    os.makedirs(folder_name, exist_ok=True)

    result_path = os.path.join(folder_name, "results.json")
    try:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {result_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        
        # Try to save to a backup location
        backup_path = os.path.join("results", f"backup_results_{launch_number}.json")
        try:
            with open(backup_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to backup location: {backup_path}")
        except:
            print("Failed to save results to backup location as well")
