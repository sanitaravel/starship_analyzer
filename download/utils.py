"""
Utility functions for download operations.
"""
import json
import os
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

FLIGHTS_URL = "https://raw.githubusercontent.com/sanitaravel/starship_launches/refs/heads/master/flights.json"

def get_launch_data():
    """
    Retrieve the flight data from GitHub.
    
    Returns:
        dict: A dictionary containing flight information, or None if there was an error.
    """
    try:
        logger.info(f"Fetching flight data from {FLIGHTS_URL}")
        
        response = requests.get(FLIGHTS_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        flight_data = response.json()
        logger.info(f"Successfully retrieved data for {len(flight_data)} flights")
        return flight_data
        
    except requests.RequestException as e:
        logger.error(f"Error fetching flight data: {e}")
        print(f"Error fetching flight data: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing flight JSON data: {e}")
        print(f"Error parsing flight data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching flight data: {e}")
        print(f"Error fetching flight data: {e}")
        return None

def get_downloaded_launches(output_path="flight_recordings"):
    """
    Get a list of already downloaded flight numbers.
    
    Args:
        output_path (str): Path to check for downloaded files
        
    Returns:
        list: List of downloaded flight numbers as integers
    """
    downloaded = []
    
    if not os.path.exists(output_path):
        return downloaded
    
    # Check for files matching the pattern "flight_X.*"
    for file in os.listdir(output_path):
        if file.startswith("flight_"):
            try:
                # Extract the flight number from the filename
                flight_num = int(file.split("_")[1].split(".")[0])
                downloaded.append(flight_num)
            except (IndexError, ValueError):
                continue
    
    logger.debug(f"Found already downloaded flights: {downloaded}")
    return downloaded
