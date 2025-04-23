"""
Menu interfaces for download operations.
"""
import inquirer
from download.utils import get_downloaded_launches, get_launch_data
from utils.logger import get_logger
from .downloader import download_twitter_broadcast, download_youtube_video
from utils.terminal import clear_screen
from utils.validators import validate_number, validate_url

logger = get_logger(__name__)

def download_media_menu():
    """Combined menu for downloading media from different sources."""
    clear_screen()
    logger.debug("Starting media download menu")
    
    menu_options = [
        'Download from launch list',
        'Download from custom URL',
        'Back to main menu'
    ]
    
    menu_answer = prompt_menu_options("Select download option:", menu_options)
    
    if menu_answer == 'Back to main menu':
        clear_screen()
        return True
    
    if menu_answer == 'Download from launch list':
        return download_from_launch_list()
    else:  # Download from custom URL
        return download_from_custom_url()

def prompt_menu_options(message, options):
    """Show a menu with the given options and return the selected option."""
    menu_question = [
        inquirer.List(
            'option',
            message=message,
            choices=options,
        ),
    ]
    
    menu_answer = inquirer.prompt(menu_question)
    return menu_answer['option']

def download_from_launch_list():
    """Download video from the GitHub flight list."""
    clear_screen()
    logger.debug("Downloading from flight list")
    
    flight_data = get_flight_data()
    if not flight_data:
        return handle_error("Could not retrieve flight data. Please try again later.")
    
    available_flights = get_available_flights(flight_data)
    
    if not available_flights:
        return handle_error("All flights have already been downloaded or no flights are available.")
    
    choices = available_flights + [("Back to download menu", -1)]
    
    selected_flight_num = display_flight_selection_menu(choices)
    
    if selected_flight_num == -1:  # Back option
        clear_screen()
        return download_media_menu()
    
    download_status = download_selected_flight(flight_data, selected_flight_num)
    
    return prompt_continue_after_download(download_status, selected_flight_num)

def get_flight_data():
    """Retrieve flight data from repository."""
    return get_launch_data()

def get_available_flights(flight_data):
    """Create a list of flights that haven't been downloaded yet."""
    downloaded_flights = get_downloaded_launches()
    
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
    return available_flights

def display_flight_selection_menu(choices):
    """Display menu for flight selection and return selected flight number."""
    flight_question = [
        inquirer.List(
            'selected_flight',
            message="Select a flight to download:",
            choices=choices,
        ),
    ]
    
    flight_answer = inquirer.prompt(flight_question)
    return flight_answer['selected_flight']

def download_selected_flight(flight_data, selected_flight_num):
    """Download the selected flight."""
    flight_key = f"flight_{selected_flight_num}"
    flight_info = flight_data.get(flight_key)
    
    if not flight_info:
        print(f"Flight information for {flight_key} not found.")
        return False
    
    url = flight_info['url']
    flight_type = flight_info['type']
    
    print(f"Downloading {flight_key} from {url}...")
    return execute_download(flight_type, url, selected_flight_num)

def handle_error(message):
    """Display error message and prompt to continue."""
    print(message)
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def prompt_continue_after_download(success, flight_num):
    """Show download status message and prompt to continue."""
    if success:
        print(f"Download of flight_{flight_num} completed successfully.")
    else:
        print(f"Failed to download flight_{flight_num}.")
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def download_from_custom_url():
    """Download media from a custom URL."""
    clear_screen()
    logger.debug("Downloading from custom URL")
    
    platform = select_platform()
    
    if platform == 'Back to download menu':
        clear_screen()
        return download_media_menu()
    
    url, flight_number = get_url_and_flight_number(platform)
    
    if not url:
        return handle_error("Download cancelled.")
    
    success = download_from_platform(platform, url, flight_number)
    
    if success:
        print("Download completed successfully.")
    
    input("\nPress Enter to continue...")
    clear_screen()
    return True

def select_platform():
    """Show menu to select download platform."""
    platform_choices = [
        'Twitter/X Broadcast',
        'YouTube Video',
        'Back to download menu'
    ]
    
    return prompt_menu_options("Select platform to download from", platform_choices)

def get_url_and_flight_number(platform):
    """Prompt for URL and flight number."""
    questions = [
        inquirer.Text('url', message=f"Enter the {platform} URL", 
                     validate=validate_url),
        inquirer.Text('flight_number', message="Enter the flight number", 
                     validate=validate_number)
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers or not answers['url'].strip():
        return None, None
    
    return answers['url'].strip(), int(answers['flight_number'])

def download_from_platform(platform, url, flight_number):
    """Execute download based on selected platform."""
    if platform == 'Twitter/X Broadcast':
        return download_twitter_broadcast(url, flight_number)
    elif platform == 'YouTube Video':
        return download_youtube_video(url, flight_number)
    return False

def execute_download(media_type, url, flight_num):
    """Execute download based on media type."""
    if media_type == "youtube":
        return download_youtube_video(url, flight_num)
    elif media_type in ["twitter/x", "twitter", "x"]:
        return download_twitter_broadcast(url, flight_num)
    return False
