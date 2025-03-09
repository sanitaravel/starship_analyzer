import inquirer
from plot import plot_flight_data, compare_multiple_launches
from processing import process_image, process_video_frame, process_frame, iterate_through_frames
import os
from inquirer import errors
import re


def validate_number(_, current):
    try:
        if current.strip() == "":  # Allow empty for default values
            return True
        value = int(current)
        return True
    except ValueError:
        raise errors.ValidationError('', reason='Please enter a valid number')


def get_video_files_from_flight_recordings():
    """Get a list of video files from the flight_recordings folder."""
    flight_recordings_folder = '.\\flight_recordings'
    video_files = []
    
    if os.path.exists(flight_recordings_folder):
        for root, _, files in os.walk(flight_recordings_folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    relative_path = os.path.relpath(os.path.join(root, file), '.')
                    video_files.append((file, relative_path))
    
    if not video_files:
        print("No video files found in flight_recordings folder.")
    
    return video_files


def display_menu() -> None:
    """
    Display a step-by-step menu for the user to navigate and select options.
    """
    questions = [
        inquirer.List(
            'action',
            message="What would you like to do?",
            choices=[
                'Process a single image',
                'Extract data from a random frame in a video',
                'Extract data from a specified frame in a video',
                'Extract data from a user-specified frame in a video',
                'Run through whole video',
                'Analyze flight data',
                'Compare multiple launches',
                'Exit'
            ],
        ),
    ]

    answers = inquirer.prompt(questions)

    if answers['action'] == 'Process a single image':
        questions = [
            inquirer.Path(
                'image_path',
                message="Path to the image file",
                exists=True,
                path_type=inquirer.Path.FILE,
            ),
            inquirer.Confirm(
                'display_rois', message="Display ROIs?", default=False),
            inquirer.Confirm(
                'debug', message="Enable debug prints?", default=False),
        ]
        answers = inquirer.prompt(questions)
        process_image(answers['image_path'],
                      answers['display_rois'], answers['debug'])

    elif answers['action'] == 'Extract data from a random frame in a video':
        # Get list of video files from flight_recordings folder
        video_files = get_video_files_from_flight_recordings()
        if not video_files:
            return
            
        questions = [
            inquirer.List(
                'video_path',
                message="Select a video file",
                choices=video_files,
            ),
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
        start_time = int(answers['start_time']) if answers['start_time'] else 0
        end_time = int(answers['end_time']) if answers['end_time'] else -1
        process_video_frame(
            answers['video_path'][1], answers['display_rois'], answers['debug'], start_time, end_time)

    elif answers['action'] == 'Extract data from a specified frame in a video':
        # Get list of video files from flight_recordings folder
        video_files = get_video_files_from_flight_recordings()
        if not video_files:
            return
            
        questions = [
            inquirer.List(
                'video_path',
                message="Select a video file",
                choices=video_files,
            ),
            inquirer.Text(
                'frame_number', message="Frame number to extract", validate=validate_number),
            inquirer.Confirm(
                'display_rois', message="Display ROIs?", default=False),
            inquirer.Confirm(
                'debug', message="Enable debug prints?", default=False),
        ]
        answers = inquirer.prompt(questions)
        process_frame(answers['video_path'][1], int(answers['frame_number']),
                      answers['display_rois'], answers['debug'], ".\\.tmp\\specified_frame.jpg")

    elif answers['action'] == 'Extract data from a user-specified frame in a video':
        # Get list of video files from flight_recordings folder
        video_files = get_video_files_from_flight_recordings()
        if not video_files:
            return
            
        questions = [
            inquirer.List(
                'video_path',
                message="Select a video file",
                choices=video_files,
            ),
            inquirer.Text(
                'frame_number', message="Frame number to extract", validate=validate_number),
            inquirer.Confirm(
                'display_rois', message="Display ROIs?", default=False),
            inquirer.Confirm(
                'debug', message="Enable debug prints?", default=False),
        ]
        answers = inquirer.prompt(questions)
        process_frame(answers['video_path'][1], int(answers['frame_number']),
                      answers['display_rois'], answers['debug'], ".\\.tmp\\user_specified_frame.jpg")

    elif answers['action'] == 'Run through whole video':
        # Get list of video files from flight_recordings folder
        video_files = get_video_files_from_flight_recordings()
        if not video_files:
            return
            
        questions = [
            inquirer.List(
                'video_path',
                message="Select a video file",
                choices=video_files,
            ),
            inquirer.Text('launch_number', message="Launch number",
                          validate=validate_number),
        ]
        answers = inquirer.prompt(questions)
        iterate_through_frames(
            answers['video_path'][1], int(answers['launch_number']))

    elif answers['action'] == 'Analyze flight data':
        launch_folders = [f for f in os.listdir('.\\results') if os.path.isdir(
            os.path.join('.\\results', f)) and f != 'compare_launches']

        if not launch_folders:
            print("No launch folders found in ./results directory.")
            return

        questions = [
            inquirer.List(
                'launch_folder',
                message="Select the launch folder",
                choices=launch_folders,
            ),
            inquirer.Text(
                'start_time', message="Start time in seconds (default: 0)", validate=validate_number),
            inquirer.Text(
                'end_time', message="End time in seconds (default: -1 for all data)", validate=validate_number),
            inquirer.Confirm(
                'show_figures', message="Display figures interactively?", default=True)
        ]
        answers = inquirer.prompt(questions)

        json_path = os.path.join(
            '.\\results', answers['launch_folder'], 'results.json')
        start_time = int(answers['start_time']) if answers['start_time'] else 0
        end_time = int(answers['end_time']) if answers['end_time'] else -1
        plot_flight_data(json_path, start_time, end_time, show_figures=answers['show_figures'])

    elif answers['action'] == 'Compare multiple launches':
        launch_folders = [f for f in os.listdir('.\\results') if os.path.isdir(
            os.path.join('.\\results', f)) and f != 'compare_launches']

        if len(launch_folders) < 2:
            print("Need at least two launch folders in ./results directory to compare.")
            return

        questions = [
            inquirer.Checkbox(
                'launches',
                message="Select the launches to compare (press space to select)",
                choices=launch_folders,
            ),
            inquirer.Text(
                'start_time', message="Start time in seconds (default: 0)", validate=validate_number),
            inquirer.Text(
                'end_time', message="End time in seconds (default: -1 for all data)", validate=validate_number),
            inquirer.Confirm(
                'show_figures', message="Display figures interactively?", default=True)
        ]
        answers = inquirer.prompt(questions)

        if len(answers['launches']) < 2:
            print("Please select at least two launches to compare.")
            return

        json_paths = [os.path.join('.\\results', folder, 'results.json')
                      for folder in answers['launches']]
        start_time = int(answers['start_time']) if answers['start_time'] else 0
        end_time = int(answers['end_time']) if answers['end_time'] else -1
        compare_multiple_launches(start_time, end_time, json_paths, show_figures=answers['show_figures'])

    elif answers['action'] == 'Exit':
        print("Exiting the program.")
        exit()


def main() -> None:
    """
    Main function to handle the step-by-step menu and run the appropriate function.
    """
    display_menu()


if __name__ == "__main__":
    main()
