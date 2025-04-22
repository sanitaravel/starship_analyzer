"""
Video processing menu and related functionality.
"""
import inquirer
from utils.logger import get_logger
from utils.terminal import clear_screen
from utils.validators import validate_number, validate_positive_number
from utils.video_utils import get_video_files_from_flight_recordings, display_video_info
from processing import process_video_frame, iterate_through_frames

logger = get_logger(__name__)

def video_processing_menu():
    """Submenu for video processing options."""
    from main import DEBUG_MODE  # Import here to avoid circular imports
    
    clear_screen()
    debug_status = "Enabled" if DEBUG_MODE else "Disabled"
    
    questions = [
        inquirer.List(
            'action',
            message="Video Processing Options:",
            choices=[
                'Process random video frame',
                'Process complete video',
                'Back to main menu'
            ],
        ),
    ]
    
    answers = inquirer.prompt(questions)
    
    logger.debug(f"Video processing menu: User selected: {answers['action']}")
    
    if answers['action'] == 'Process random video frame':
        process_random_frame()
        return video_processing_menu()
    elif answers['action'] == 'Process complete video':
        process_complete_video()
        return video_processing_menu()
    elif answers['action'] == 'Back to main menu':
        clear_screen()
        return True
    
    clear_screen()
    return True

def process_random_frame():
    """Handle the process random video frame menu option."""
    from main import DEBUG_MODE  # Import here to avoid circular imports
    
    clear_screen()
    video_files = get_video_files_from_flight_recordings()
    if not video_files:
        return True
    
    logger.debug("Starting random frame processing")
    
    # First, get the video path
    video_question = [
        inquirer.List(
            'video_path',
            message="Select a video file",
            choices=video_files,
        )
    ]
    video_answer = inquirer.prompt(video_question)
    
    # Display video information
    display_video_info(video_answer['video_path'])
    
    # Continue with other questions
    questions = [
        inquirer.Confirm(
            'display_rois', message="Display ROIs?", default=False),
        inquirer.Confirm(
            'debug', message="Enable debug prints?", default=False),
        inquirer.Text(
            'start_time', message="Start time in seconds (default: 0)", validate=validate_number),
        inquirer.Text(
            'end_time', message="End time in seconds (default: -1 for all)", validate=validate_number),
    ]
    answers = inquirer.prompt(questions)
    answers['video_path'] = video_answer['video_path']  # Combine answers
    
    start_time = int(answers['start_time']) if answers['start_time'] else 0
    end_time = int(answers['end_time']) if answers['end_time'] else -1
    
    logger.debug(f"Processing random frame from {answers['video_path']} with start_time={start_time}, end_time={end_time}")
    process_video_frame(
        answers['video_path'], answers['display_rois'], answers['debug'] or DEBUG_MODE, start_time, end_time)
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def select_video_file():
    """Select a video file from available flight recordings."""
    video_files = get_video_files_from_flight_recordings()
    if not video_files:
        return None
    
    video_question = [
        inquirer.List(
            'video_path',
            message="Select a video file",
            choices=video_files,
        )
    ]
    video_answer = inquirer.prompt(video_question)
    return video_answer['video_path']

def get_processing_parameters():
    """Get processing parameters from user."""
    questions = [
        inquirer.Text('launch_number', message="Launch number",
                    validate=validate_number),
        inquirer.Text('batch_size', 
                     message="Batch size for processing (default: 10)", 
                     validate=validate_positive_number),
        inquirer.Text('sample_rate', 
                     message="Sample rate (process every Nth frame, default: 1)", 
                     validate=validate_positive_number),
        # Add border options
        inquirer.List(
            'border_type',
            message="How would you like to specify processing borders?",
            choices=['Time-based (seconds)', 'Frame-based', 'Process entire video'],
            default='Process entire video'
        )
    ]
    return inquirer.prompt(questions)

def get_time_based_borders():
    """Get time-based processing borders from user."""
    time_questions = [
        inquirer.Text(
            'start_time', 
            message="Start time in seconds (default: 0)", 
            validate=validate_number
        ),
        inquirer.Text(
            'end_time', 
            message="End time in seconds (default: process to end)", 
            validate=validate_number
        )
    ]
    time_answers = inquirer.prompt(time_questions)
    
    start_time = float(time_answers['start_time']) if time_answers['start_time'] else 0
    end_time = float(time_answers['end_time']) if time_answers['end_time'] else None
    
    return start_time, end_time

def get_frame_based_borders():
    """Get frame-based processing borders from user."""
    frame_questions = [
        inquirer.Text(
            'start_frame', 
            message="Start frame number (default: 0)", 
            validate=validate_number
        ),
        inquirer.Text(
            'end_frame', 
            message="End frame number (default: process to end)", 
            validate=validate_number
        )
    ]
    frame_answers = inquirer.prompt(frame_questions)
    
    start_frame = int(frame_answers['start_frame']) if frame_answers['start_frame'] else 0
    end_frame = int(frame_answers['end_frame']) if frame_answers['end_frame'] else None
    
    return start_frame, end_frame

def process_video_with_parameters(video_path, launch_number, batch_size, sample_rate, 
                                 start_time=None, end_time=None, start_frame=None, end_frame=None):
    """Process the video with the provided parameters."""
    from main import DEBUG_MODE  # Import here to avoid circular imports
    
    logger.debug(f"Processing complete video {video_path} with launch_number={launch_number}, "
                f"batch_size={batch_size}, sample_rate={sample_rate}")
    logger.debug(f"Borders: start_time={start_time}, end_time={end_time}, "
                f"start_frame={start_frame}, end_frame={end_frame}")
    
    iterate_through_frames(
        video_path, int(launch_number), debug=DEBUG_MODE, 
        batch_size=batch_size, sample_rate=sample_rate,
        start_time=start_time, end_time=end_time, 
        start_frame=start_frame, end_frame=end_frame)

def process_complete_video():
    """Handle the process complete video menu option."""
    from main import DEBUG_MODE  # Import here to avoid circular imports
    
    clear_screen()
    video_path = select_video_file()
    if not video_path:
        return True
    
    logger.debug("Starting complete video processing")
    
    # Display video information
    display_video_info(video_path)
    
    # Get processing parameters
    answers = get_processing_parameters()
    
    batch_size = int(answers['batch_size']) if answers['batch_size'] else 10
    sample_rate = int(answers['sample_rate']) if answers['sample_rate'] else 1
    
    # Initialize borders
    start_time = None
    end_time = None
    start_frame = None
    end_frame = None
    
    # Get border information based on user's choice
    if answers['border_type'] == 'Time-based (seconds)':
        start_time, end_time = get_time_based_borders()
    elif answers['border_type'] == 'Frame-based':
        start_frame, end_frame = get_frame_based_borders()
    
    process_video_with_parameters(
        video_path, answers['launch_number'], batch_size, sample_rate,
        start_time, end_time, start_frame, end_frame
    )
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True
