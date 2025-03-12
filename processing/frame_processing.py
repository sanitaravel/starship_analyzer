import random
import os
import cv2
from typing import Optional
from ocr import extract_data


def process_image(image_path: str, display_rois: bool, debug: bool) -> None:
    """
    Process a single image and extract data.

    Args:
        image_path (str): The path to the image file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
    """
    image = cv2.imread(image_path)
    superheavy_data, starship_data, time_data = extract_data(
        image, display_rois=display_rois, debug=debug)
    if debug:
        print(
            f"Superheavy - Speed: {superheavy_data['speed']
                                   }, Altitude: {superheavy_data['altitude']}"
        )
        print(
            f"Starship - Speed: {starship_data['speed']
                                 }, Altitude: {starship_data['altitude']}"
        )

        if time_data:
            time_str = f"{time_data['sign']} {time_data['hours']:02}:{
                time_data['minutes']:02}:{time_data['seconds']:02}"
            print(f"Time: {time_str}")
        else:
            print("Time: Not found")


def process_video_frame(video_path: str, display_rois: bool, debug: bool, start_time: Optional[int], end_time: Optional[int]) -> None:
    """
    Extract data from a random frame in a video within a specified timeframe.

    Args:
        video_path (str): The path to the video file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        start_time (int, optional): The start time in seconds for the timeframe.
        end_time (int, optional): The end time in seconds for the timeframe.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if start_time is not None:
        start_frame = int(start_time * fps)
    else:
        start_frame = 0

    if end_time != -1:
        end_frame = int(end_time * fps)
    else:
        end_frame = frame_count - 1

    if start_frame >= end_frame:
        raise ValueError("Start time must be less than end time")

    random_frame_number = random.randint(start_frame, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to extract frame from video")

    tmp_dir = os.path.join('.', '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = os.path.join(tmp_dir, "random_frame.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Extracted frame number: {random_frame_number}")
    process_image(image_path, display_rois, debug)


def process_frame(video_path: str, frame_number: int, display_rois: bool, debug: bool, output_filename: str) -> None:
    """
    Extract data from a specified frame in a video.

    Args:
        video_path (str): The path to the video file.
        frame_number (int): The frame number to extract.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        output_filename (str): The filename to save the extracted frame as.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_filename, frame)
        image_path = output_filename
        print(f"Extracted frame number: {frame_number}")
        process_image(image_path, display_rois, debug)
    else:
        raise ValueError("Failed to extract frame from video")
