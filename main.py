import inquirer
from plot import plot_flight_data, compare_multiple_launches
from processing import process_image, process_video_frame, process_frame, iterate_through_frames
import os

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
        image_path = input("Enter the path to the image file: ")
        display_rois = inquirer.confirm("Display ROIs?", default=False)
        debug = inquirer.confirm("Enable debug prints?", default=False)
        process_image(image_path, display_rois, debug)

    elif answers['action'] == 'Extract data from a random frame in a video':
        video_path = input("Enter the path to the video file: ")
        display_rois = inquirer.confirm("Display ROIs?", default=False)
        debug = inquirer.confirm("Enable debug prints?", default=False)
        start_time = int(input("Enter the start time in seconds (optional): ") or 0)
        end_time = int(input("Enter the end time in seconds (optional): ") or -1) 
        process_video_frame(video_path, display_rois, debug, start_time, end_time)

    elif answers['action'] == 'Extract data from a specified frame in a video':
        video_path = input("Enter the path to the video file: ")
        frame_number = int(input("Enter the frame number to extract: "))
        display_rois = inquirer.confirm("Display ROIs?", default=False)
        debug = inquirer.confirm("Enable debug prints?", default=False)
        process_frame(video_path, frame_number, display_rois, debug, ".\\.tmp\\specified_frame.jpg")

    elif answers['action'] == 'Extract data from a user-specified frame in a video':
        video_path = input("Enter the path to the video file: ")
        frame_number = int(input("Enter the frame number to extract: "))
        display_rois = inquirer.confirm("Display ROIs?", default=False)
        debug = inquirer.confirm("Enable debug prints?", default=False)
        process_frame(video_path, frame_number, display_rois, debug, ".\\.tmp\\user_specified_frame.jpg")

    elif answers['action'] == 'Run through whole video':
        video_path = input("Enter the path to the video file: ")
        launch_number = int(input("Enter the launch number: "))
        iterate_through_frames(video_path, launch_number)

    elif answers['action'] == 'Analyze flight data':
        launch_folders = [f for f in os.listdir('.\\results') if os.path.isdir(os.path.join('.\\results', f)) and f != 'compare_launches']
        launch_folder = inquirer.List(
            'launch_folder',
            message="Select the launch folder",
            choices=launch_folders,
        )
        selected_folder = inquirer.prompt([launch_folder])['launch_folder']
        json_path = os.path.join('.\\results', selected_folder, 'results.json')
        plot_flight_data(json_path)

    elif answers['action'] == 'Compare multiple launches':
        launch_folders = [f for f in os.listdir('.\\results') if os.path.isdir(os.path.join('.\\results', f)) and f != 'compare_launches']
        launch_selection = inquirer.Checkbox(
            'launches',
            message="Select the launches to compare (press space to select)",
            choices=launch_folders,
        )
        selected_folders = inquirer.prompt([launch_selection])['launches']
        
        if len(selected_folders) < 2:
            print("Please select at least two launches to compare.")
            return
        
        json_paths = [os.path.join('.\\results', folder, 'results.json') for folder in selected_folders]
        timeframe = int(input("Enter the timeframe in seconds to plot (optional): ") or -1)
        compare_multiple_launches(timeframe, *json_paths)

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
