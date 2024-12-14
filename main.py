import argparse
import cv2
from typing import Optional
from extract_data import extract_data
from video_utils import extract_random_frame, iterate_through_frames
from analyze import analyze_results


def process_image(image_path: str, display_rois: bool, debug: bool) -> None:
    """
    Process a single image and extract data.

    Args:
        image_path (str): The path to the image file.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
    """
    image = cv2.imread(image_path)
    superheavy_data, starship_data, time_data = extract_data(image, display_rois=display_rois, debug=debug)
    if debug:
        print(
            f"Superheavy - Speed: {superheavy_data['speed']}, Altitude: {superheavy_data['altitude']}"
        )
        print(
            f"Starship - Speed: {starship_data['speed']}, Altitude: {starship_data['altitude']}"
        )
        
        if time_data:
            time_str = f"{time_data['sign']} {time_data['hours']:02}:{time_data['minutes']:02}:{time_data['seconds']:02}"
            print(f"Time: {time_str}")
        else:
            print("Time: Not found")


def process_video_frame(display_rois: bool, debug: bool, start_time: Optional[int], end_time: Optional[int]) -> None:
    """
    Extract data from a random frame in a video within a specified timeframe.

    Args:
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        start_time (int, optional): The start time in seconds for the timeframe.
        end_time (int, optional): The end time in seconds for the timeframe.
    """
    frame, frame_number = extract_random_frame("launch_file_1080p.mp4", start_time, end_time)
    cv2.imwrite("random_frame.jpg", frame)
    image_path = "random_frame.jpg"
    print(f"Extracted frame number: {frame_number}")
    process_image(image_path, display_rois, debug)


def process_specified_frame(frame_number: int, display_rois: bool, debug: bool) -> None:
    """
    Extract data from a specified frame in a video.

    Args:
        frame_number (int): The frame number to extract.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
    """
    cap = cv2.VideoCapture("launch_file_1080p.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("specified_frame.jpg", frame)
        image_path = "specified_frame.jpg"
        print(f"Extracted frame number: {frame_number}")
        process_image(image_path, display_rois, debug)
    else:
        raise ValueError("Failed to extract frame from video")


def process_user_specified_frame(frame_number: int, display_rois: bool, debug: bool) -> None:
    """
    Extract data from a user-specified frame in a video.

    Args:
        frame_number (int): The frame number to extract.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
    """
    cap = cv2.VideoCapture("launch_file_1080p.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("user_specified_frame.jpg", frame)
        image_path = "user_specified_frame.jpg"
        print(f"Extracted frame number: {frame_number}")
        process_image(image_path, display_rois, debug)
    else:
        raise ValueError("Failed to extract frame from video")


def main(args: argparse.Namespace) -> None:
    """
    Main function to handle command-line arguments and run the appropriate function.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    if args.random_frame_mode:
        process_video_frame(args.display_rois, args.debug, args.start_time, args.end_time)
    elif args.iterate_frames:
        max_frames = args.test_mode if args.test_mode else None
        iterate_through_frames("launch_file_1080p.mp4", display_rois=args.display_rois, debug=args.debug, max_frames=max_frames)
    elif args.analyze_results:
        analyze_results(args.json_path, args.plot_relative_to_frame)
    elif args.frame_number is not None:
        process_specified_frame(args.frame_number, args.display_rois, args.debug)
    elif args.user_frame_number is not None:
        process_user_specified_frame(args.user_frame_number, args.display_rois, args.debug)
    else:
        process_image(args.image_path, args.display_rois, args.debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from Starship launch stream.")
    parser.add_argument("image_path", type=str, nargs='?', help="Path to the image file.")
    parser.add_argument("--random_frame_mode", action="store_true", help="Extract data from a random frame in the video.")
    parser.add_argument("--iterate_frames", action="store_true", help="Iterate through all frames in the video.")
    parser.add_argument("--test_mode", type=int, help="Test mode: iterate through the first 1000 frames.")
    parser.add_argument("--display_rois", action="store_true", help="Display ROIs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints.")
    parser.add_argument("--analyze_results", action="store_true", help="Analyze the results stored in a JSON file.")
    parser.add_argument("--json_path", type=str, help="Path to the JSON file containing the results.")
    parser.add_argument("--start_time", type=int, help="Start time in seconds for the random frame extraction.")
    parser.add_argument("--end_time", type=int, help="End time in seconds for the random frame extraction.")
    parser.add_argument("--frame_number", type=int, help="Frame number to extract data from.")
    parser.add_argument("--plot_relative_to_frame", action="store_true", help="Plot data relative to frame number.")
    parser.add_argument("--user_frame_number", type=int, help="Frame number specified by the user to extract data from.")
    args = parser.parse_args()
    
    main(args)