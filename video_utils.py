import cv2
import random
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from extract_data import extract_data
import os


def extract_random_frame(video_path: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Extract a random frame from a video within a specified timeframe.

    Args:
        video_path (str): The path to the video file.
        start_time (int, optional): The start time in seconds for the timeframe.
        end_time (int, optional): The end time in seconds for the timeframe.

    Returns:
        tuple: A tuple containing the extracted frame and the frame number.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if start_time is not None:
        start_frame = int(start_time * fps)
    else:
        start_frame = 0
    
    if end_time is not None:
        end_frame = int(end_time * fps)
    else:
        end_frame = frame_count - 1
    
    if start_frame >= end_frame:
        raise ValueError("Start time must be less than end time")
    
    random_frame_number = random.randint(start_frame, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame, random_frame_number
    else:
        raise ValueError("Failed to extract frame from video")


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
    superheavy_data, starship_data, time_data = extract_data(frame, display_rois=display_rois, debug=debug, zero_time_met=zero_time_met)
    frame_result = {
        "frame_number": frame_number,
        "superheavy": superheavy_data,
        "starship": starship_data,
        "time": time_data
    }
    return frame_result


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
    cap = cv2.VideoCapture(video_path)
    results = []
    for frame_number in batch:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame_result = process_frame(frame_number, frame, display_rois, debug, zero_time_met)
            results.append(frame_result)
            if frame_result["time"] and frame_result["time"]['hours'] == 0 and frame_result["time"]['minutes'] == 0 and frame_result["time"]['seconds'] == 0:
                zero_time_met = True
    cap.release()
    return results


def iterate_through_frames(video_path: str, display_rois: bool = False, debug: bool = False, max_frames: Optional[int] = None, batch_size: int = 100) -> None:
    """
    Iterate through all frames in a video and extract data.

    Args:
        video_path (str): The path to the video file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        max_frames (int, optional): The maximum number of frames to process. Defaults to None.
        batch_size (int): The number of frames to process in each batch.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    results: List[Dict] = []
    zero_time_frame: Optional[int] = None
    zero_time_met = False
    
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    
    frame_numbers = list(range(frame_count))
    batches = [frame_numbers[i:i + batch_size] for i in tqdm(range(0, len(frame_numbers), batch_size), desc="Creating batches")]
    
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for processing.")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_batch, batch, video_path, display_rois, debug, zero_time_met) for batch in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            results.extend(batch_results)
            for frame_result in batch_results:
                if frame_result["time"] and frame_result["time"]['hours'] == 0 and frame_result["time"]['minutes'] == 0 and frame_result["time"]['seconds'] == 0 and not zero_time_frame:
                    zero_time_frame = frame_result["frame_number"]
                    zero_time_met = True
                
                if debug:
                    print(
                        f"\rFrame {frame_result['frame_number']} - Superheavy - Speed: {frame_result['superheavy']['speed']}, Altitude: {frame_result['superheavy']['altitude']} | "
                        f"Starship - Speed: {frame_result['starship']['speed']}, Altitude: {frame_result['starship']['altitude']} | "
                        f"Time: {frame_result['time']['sign']} {frame_result['time']['hours']:02}:{frame_result['time']['minutes']:02}:{frame_result['time']['seconds']:02}" if frame_result['time'] else "Time: Not found",
                        end=''
                    )
    
    # Calculate real time for each frame
    for frame_result in tqdm(results, desc="Calculating real time"):
        frame_number = frame_result["frame_number"]
        real_time = None
        if zero_time_frame is not None:
            real_time = (frame_number - zero_time_frame) / fps
        frame_result["real_time"] = real_time
        
        if debug:
            if real_time:
                real_time_str = f"{real_time['hours']:02}:{real_time['minutes']:02}:{real_time['seconds']:02}.{real_time['milliseconds']:03}"
                print(f"\rFrame {frame_number} - Real Time: {real_time_str}", end='')
            else:
                print(f"\rFrame {frame_number} - Real Time: Not calculated", end='')
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("JSON dumped successfully.")