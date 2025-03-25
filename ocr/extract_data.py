import numpy as np
from typing import Tuple, Dict
from utils import display_image
from .ocr import extract_values_from_roi
from .engine_detection import detect_engine_status
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def preprocess_image(image: np.ndarray, display_rois: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image to extract ROIs for Superheavy Speed, Superheavy Altitude, Starship Speed, Starship Altitude, and Time.

    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.

    Returns:
        tuple: A tuple containing the ROIs for Superheavy Speed, Superheavy Altitude, Starship Speed, Starship Altitude, and Time.
    """
    # Define ROIs based on the provided coordinates and dimensions
    sh_speed_roi = image[913:913+25, 359:359+83]  # SH Speed: 359;913, size 83x25
    sh_altitude_roi = image[948:948+25, 392:392+50]  # SH Altitude: 392;948, size 50x25
    ss_speed_roi = image[913:913+25, 1539:1539+83]  # SS Speed: 1539;913, size 83x25
    ss_altitude_roi = image[948:948+25, 1572:1572+50]  # SS Altitude: 1572;948, size 50x25
    
    # Define time ROI based on provided coordinates and dimensions
    time_roi = image[940:940+44, 860:860+197]  # Time: 860;940, size 197x44

    # Display ROIs for debugging if requested
    if display_rois:
        display_image(sh_speed_roi, "Superheavy Speed ROI")
        display_image(sh_altitude_roi, "Superheavy Altitude ROI")
        display_image(ss_speed_roi, "Starship Speed ROI")
        display_image(ss_altitude_roi, "Starship Altitude ROI")
        display_image(time_roi, "Time ROI")
        logger.debug("Displayed ROIs for visual inspection")

    return sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi


def extract_superheavy_data(sh_speed_roi: np.ndarray, sh_altitude_roi: np.ndarray, display_rois: bool, debug: bool) -> Dict:
    """
    Extract data for Superheavy from the ROIs.

    Args:
        sh_speed_roi (numpy.ndarray): The ROI for Superheavy speed.
        sh_altitude_roi (numpy.ndarray): The ROI for Superheavy altitude.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing the extracted data for Superheavy.
    """
    speed_data = extract_values_from_roi(sh_speed_roi, mode="speed", display_transformed=display_rois, debug=debug)
    altitude_data = extract_values_from_roi(sh_altitude_roi, mode="altitude", display_transformed=display_rois, debug=debug)
    
    if debug:
        logger.debug(f"Extracted Superheavy speed: {speed_data.get('value')}, altitude: {altitude_data.get('value')}")
    
    return {
        "speed": speed_data.get("value"),
        "altitude": altitude_data.get("value")
    }


def extract_starship_data(ss_speed_roi: np.ndarray, ss_altitude_roi: np.ndarray, display_rois: bool, debug: bool) -> Dict:
    """
    Extract data for Starship from the ROIs.

    Args:
        ss_speed_roi (numpy.ndarray): The ROI for Starship speed.
        ss_altitude_roi (numpy.ndarray): The ROI for Starship altitude.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing the extracted data for Starship.
    """
    speed_data = extract_values_from_roi(ss_speed_roi, mode="speed", display_transformed=display_rois, debug=debug)
    altitude_data = extract_values_from_roi(ss_altitude_roi, mode="altitude", display_transformed=display_rois, debug=debug)
    
    if debug:
        logger.debug(f"Extracted Starship speed: {speed_data.get('value')}, altitude: {altitude_data.get('value')}")
    
    return {
        "speed": speed_data.get("value"),
        "altitude": altitude_data.get("value")
    }


def extract_time_data(time_roi: np.ndarray, display_rois: bool, debug: bool, zero_time_met: bool) -> Dict:
    """
    Extract time data from the ROI.

    Args:
        time_roi (numpy.ndarray): The ROI for time.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        zero_time_met (bool): Whether a frame with time 0:0:0 has been met.

    Returns:
        dict: A dictionary containing the extracted time data.
    """
    if zero_time_met:
        if debug:
            logger.debug("Zero time already met, returning default zero time")
        return {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
    
    time_data = extract_values_from_roi(time_roi, mode="time", display_transformed=display_rois, debug=debug)
    
    if debug and time_data:
        logger.debug(f"Extracted time: {time_data.get('sign')} {time_data.get('hours', 0):02}:{time_data.get('minutes', 0):02}:{time_data.get('seconds', 0):02}")
    
    return time_data


def extract_data(image: np.ndarray, display_rois: bool = False, debug: bool = False, zero_time_met: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Extract data from an image.

    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        zero_time_met (bool): Whether a frame with time 0:0:0 has been met.

    Returns:
        tuple: A tuple containing the extracted data for Superheavy, Starship, and Time.
    """
    logger.debug("Starting data extraction from image")
    
    # Preprocess the image to get ROIs
    sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(
        image, display_rois)

    # Extract data for Superheavy
    superheavy_data = extract_superheavy_data(sh_speed_roi, sh_altitude_roi, display_rois, debug)
    
    # Extract data for Starship
    starship_data = extract_starship_data(ss_speed_roi, ss_altitude_roi, display_rois, debug)

    # If Starship extraction returns nothing, give it Superheavy data
    if not starship_data.get("speed") or not starship_data.get("altitude"):
        logger.debug("Starship data incomplete, using Superheavy data as fallback")
        starship_data = superheavy_data

    # Extract time separately
    time_data = extract_time_data(time_roi, display_rois, debug, zero_time_met)
    
    # Detect engine status
    logger.debug("Detecting engine status")
    engine_data = detect_engine_status(image, debug)
    
    # Add engine data to vehicle data
    superheavy_data["engines"] = engine_data["superheavy"]
    starship_data["engines"] = engine_data["starship"]

    logger.debug("Data extraction complete")
    return superheavy_data, starship_data, time_data
