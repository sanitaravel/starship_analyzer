"""
Visualization menu and related functionality.
"""
import inquirer
import os
from utils.logger import get_logger
from utils.terminal import clear_screen
from utils.validators import validate_number
from plot import plot_flight_data, compare_multiple_launches

logger = get_logger(__name__)

def visualization_menu():
    """Submenu for data visualization options."""
    clear_screen()
    questions = [
        inquirer.List(
            'action',
            message="Visualization Options:",
            choices=[
                'Visualize flight data',
                'Visualize multiple launches data',
                'Back to main menu'
            ],
        ),
    ]
    
    answers = inquirer.prompt(questions)
    
    logger.debug(f"Visualization menu: User selected: {answers['action']}")
    
    if answers['action'] == 'Visualize flight data':
        visualize_flight_data()
        return visualization_menu()
    elif answers['action'] == 'Visualize multiple launches data':
        compare_multiple_launches_menu()
        return visualization_menu()
    elif answers['action'] == 'Back to main menu':
        clear_screen()
        return True
    
    clear_screen()
    return True

def visualize_flight_data():
    """Handle the visualize flight data menu option."""
    clear_screen()
    results_dir = os.path.join('.', 'results')
    launch_folders = [f for f in os.listdir(results_dir) if os.path.isdir(
        os.path.join(results_dir, f)) and f != 'compare_launches']

    if not launch_folders:
        print("No launch folders found in ./results directory.")
        input("\nPress Enter to continue...")
        clear_screen()
        return True
    
    logger.debug(f"Found {len(launch_folders)} launch folders for visualization")
    
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

    json_path = os.path.join(results_dir, answers['launch_folder'], 'results.json')
    start_time = int(answers['start_time']) if answers['start_time'] else 0
    end_time = int(answers['end_time']) if answers['end_time'] else -1
    
    logger.debug(f"Visualizing flight data from {json_path} with time window {start_time} to {end_time}")
    plot_flight_data(json_path, start_time, end_time, show_figures=answers['show_figures'])
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def compare_multiple_launches_menu():
    """Handle the compare multiple launches menu option."""
    clear_screen()
    launch_folders = get_launch_folders()
    
    if not validate_available_launches(launch_folders):
        return True
    
    answers = prompt_for_comparison_options(launch_folders)
    
    if not validate_selected_launches(answers['launches']):
        return True
    
    execute_launch_comparison(
        answers['launches'], 
        answers['start_time'], 
        answers['end_time'], 
        answers['show_figures']
    )
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def get_launch_folders():
    """Get available launch folders from results directory."""
    results_dir = os.path.join('.', 'results')
    launch_folders = [f for f in os.listdir(results_dir) if os.path.isdir(
        os.path.join(results_dir, f)) and f != 'compare_launches']
    
    logger.debug(f"Found {len(launch_folders)} launch folders for comparison")
    return launch_folders

def validate_available_launches(launch_folders):
    """Validate that there are enough launch folders to compare."""
    if len(launch_folders) < 2:
        print("Need at least two launch folders in ./results directory to compare.")
        input("\nPress Enter to continue...")
        clear_screen()
        return False
    return True

def prompt_for_comparison_options(launch_folders):
    """Prompt user for comparison options."""
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
    return inquirer.prompt(questions)

def validate_selected_launches(selected_launches):
    """Validate that user selected enough launches to compare."""
    if len(selected_launches) < 2:
        print("Please select at least two launches to compare.")
        input("\nPress Enter to continue...")
        clear_screen()
        return False
    return True

def execute_launch_comparison(launches, start_time_input, end_time_input, show_figures):
    """Execute the launch comparison with the provided parameters."""
    results_dir = os.path.join('.', 'results')
    json_paths = [os.path.join(results_dir, folder, 'results.json') for folder in launches]
    start_time = int(start_time_input) if start_time_input else 0
    end_time = int(end_time_input) if end_time_input else -1
    
    logger.debug(f"Comparing launches: {', '.join(launches)}")
    logger.debug(f"Time window: {start_time} to {end_time}")
    
    compare_multiple_launches(start_time, end_time, *json_paths, show_figures=show_figures)
