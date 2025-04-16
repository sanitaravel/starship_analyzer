"""
Menu interfaces for download operations.
"""
import inquirer
from download.utils import get_downloaded_launches, get_launch_data
from utils.logger import get_logger
from .downloader import download_twitter_broadcast, download_youtube_video
from utils.terminal import clear_screen

logger = get_logger(__name__)

def download_media_menu():
    """Combined menu for downloading media from different sources."""
    
    clear_screen()
    logger.debug("Starting media download menu")
    
    # Menu options
    menu_options = [
        'Download from launch list',
        'Download from custom URL',
        'Back to main menu'
    ]
    
    # Show media download menu
    menu_question = [
        inquirer.List(
            'option',
            message="Select download option:",
            choices=menu_options,
        ),
    ]
    
    menu_answer = inquirer.prompt(menu_question)
    
    if menu_answer['option'] == 'Back to main menu':
        clear_screen()
        return True
    
    if menu_answer['option'] == 'Download from launch list':
        return download_from_launch_list()
    else:  # Download from custom URL
        return download_from_custom_url()

def download_from_launch_list():
    """Download video from the GitHub flight list."""
    
    clear_screen()
    logger.debug("Downloading from flight list")
    
    # Get flight data from GitHub
    flight_data = get_launch_data()
    if not flight_data:
        print("Could not retrieve flight data. Please try again later.")
        input("\nPress Enter to continue...")
        clear_screen()
        return True
    
    # Get list of already downloaded flights
    downloaded_flights = get_downloaded_launches()
    
    # Create a list of available flights (not downloaded yet)
    available_flights = []
    for key, value in flight_data.items():
        try:
            flight_num = int(key.split("_")[1])
            if flight_num not in downloaded_flights:
                flight_type = "YouTube" if value["type"] == "youtube" else "Twitter/X"
                available_flights.append((f"Flight {flight_num} ({flight_type})", flight_num))
        except (IndexError, ValueError, KeyError):
            logger.warning(f"Skipping malformed flight entry: {key}")
            continue
    
    # Sort by flight number
    available_flights.sort(key=lambda x: x[1])
    
    if not available_flights:
        print("All flights have already been downloaded or no flights are available.")
        input("\nPress Enter to continue...")
        clear_screen()
        return True
    
    # Add a "Back" option
    choices = available_flights + [("Back to download menu", -1)]
    
    # Prompt user to select a flight
    flight_question = [
        inquirer.List(
            'selected_flight',
            message="Select a flight to download:",
            choices=choices,
        ),
    ]
    
    flight_answer = inquirer.prompt(flight_question)
    selected_flight_num = flight_answer['selected_flight']
    
    if selected_flight_num == -1:  # Back option
        clear_screen()
        return download_media_menu()
    
    # Get the flight info
    flight_key = f"flight_{selected_flight_num}"
    flight_info = flight_data.get(flight_key)
    
    if not flight_info:
        print(f"Flight information for {flight_key} not found.")
        input("\nPress Enter to continue...")
        clear_screen()
        return True
    
    url = flight_info['url']
    flight_type = flight_info['type']
    
    success = False
    print(f"Downloading {flight_key} from {url}...")
    if flight_type == "youtube":
        success = download_youtube_video(url, selected_flight_num)
    elif flight_type in ["twitter/x", "twitter", "x"]:
        success = download_twitter_broadcast(url, selected_flight_num)
    
    if success:
        print(f"Download of {flight_key} completed successfully.")
    else:
        print(f"Failed to download {flight_key}.")
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def download_from_custom_url():
    """Download media from a custom URL."""
    from main import clear_screen, validate_number  # Local import to avoid circular dependencies
    
    clear_screen()
    logger.debug("Downloading from custom URL")
    
    # First, select the platform
    platform_question = [
        inquirer.List(
            'platform',
            message="Select platform to download from",
            choices=[
                'Twitter/X Broadcast',
                'YouTube Video',
                'Back to download menu'
            ],
        ),
    ]
    
    platform_answer = inquirer.prompt(platform_question)
    
    if platform_answer['platform'] == 'Back to download menu':
        clear_screen()
        return download_media_menu()
    
    # Now prompt for URL and flight number
    questions = [
        inquirer.Text('url', message=f"Enter the {platform_answer['platform']} URL"),
        inquirer.Text('flight_number', message="Enter the flight number", 
                     validate=lambda _, current: validate_number(_, current))
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers or not answers['url'].strip():
        print("Download cancelled.")
        input("\nPress Enter to continue...")
        clear_screen()
        return True
    
    success = False
    if platform_answer['platform'] == 'Twitter/X Broadcast':
        success = download_twitter_broadcast(
            answers['url'].strip(), 
            int(answers['flight_number'])
        )
    elif platform_answer['platform'] == 'YouTube Video':
        success = download_youtube_video(
            answers['url'].strip(), 
            int(answers['flight_number'])
        )
    
    if success:
        print("Download completed successfully.")
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True
