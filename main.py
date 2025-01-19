import inquirer
from plot import plot_flight_data, compare_multiple_launches
from processing import process_image, process_video_frame, process_frame, iterate_through_frames


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
                'Analyze flight data',
                'Compare multiple launches',
                'Run through whole video',
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
        display_rois = inquirer.confirm("Display ROIs?", default=False)
        debug = inquirer.confirm("Enable debug prints?", default=False)
        iterate_through_frames(video_path, display_rois, debug)

    elif answers['action'] == 'Analyze flight data':
        json_path = input("Enter the path to the JSON file containing the results: ")
        plot_flight_data(json_path)

    elif answers['action'] == 'Compare multiple launches':
        json_paths = input("Enter the paths to the JSON files containing the results, separated by commas: ").split(',')
        timeframe = int(input("Enter the timeframe in seconds to plot (optional): ") or None)
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
