import numpy as np
from typing import Tuple, Dict
from utils import display_image
from .ocr import extract_values_from_roi
from .engine_detection import detect_engine_status
from .fuel_level_extraction import extract_fuel_levels
from utils.logger import get_logger

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
    logger.debug(f"Preprocessing image of shape {image.shape}")
    
    try:
        # Define ROIs based on the provided coordinates and dimensions
        sh_speed_roi = image[913:913+25, 359:359+83]  # SH Speed: 359;913, size 83x25
        sh_altitude_roi = image[948:948+25, 392:392+50]  # SH Altitude: 392;948, size 50x25
        ss_speed_roi = image[913:913+25, 1539:1539+83]  # SS Speed: 1539;913, size 83x25
        ss_altitude_roi = image[948:948+25, 1572:1572+50]  # SS Altitude: 1572;948, size 50x25
        
        # Define time ROI based on provided coordinates and dimensions
        time_roi = image[940:940+44, 860:860+197]  # Time: 860;940, size 197x44

        logger.debug(f"ROI dimensions - SH Speed: {sh_speed_roi.shape}, SH Alt: {sh_altitude_roi.shape}, " +
                    f"SS Speed: {ss_speed_roi.shape}, SS Alt: {ss_altitude_roi.shape}, Time: {time_roi.shape}")

        # Display ROIs for debugging if requested
        if display_rois:
            logger.debug("Displaying ROIs for visual inspection")
            display_image(sh_speed_roi, "Superheavy Speed ROI")
            display_image(sh_altitude_roi, "Superheavy Altitude ROI")
            display_image(ss_speed_roi, "Starship Speed ROI")
            display_image(ss_altitude_roi, "Starship Altitude ROI")
            display_image(time_roi, "Time ROI")
    
    except Exception as e:
        logger.error(f"Error extracting ROIs: {str(e)}")
        logger.debug(f"Image shape: {image.shape if image is not None else 'None'}")
        # Return empty ROIs in case of error
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty_roi, empty_roi, empty_roi, empty_roi, empty_roi

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
    logger.debug("Extracting Superheavy data from ROIs")
    
    try:
        speed_data = extract_values_from_roi(sh_speed_roi, mode="speed", display_transformed=display_rois, debug=debug)
        altitude_data = extract_values_from_roi(sh_altitude_roi, mode="altitude", display_transformed=display_rois, debug=debug)
        
        if debug:
            logger.debug(f"Extracted Superheavy speed: {speed_data.get('value')}, altitude: {altitude_data.get('value')}")
            
            if speed_data.get('value') is None:
                logger.debug("No speed value found for Superheavy")
            if altitude_data.get('value') is None:
                logger.debug("No altitude value found for Superheavy")
        
        return {
            "speed": speed_data.get("value"),
            "altitude": altitude_data.get("value")
        }
    except Exception as e:
        logger.error(f"Error extracting Superheavy data: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"speed": None, "altitude": None}


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
    logger.debug("Extracting Starship data from ROIs")
    
    try:
        speed_data = extract_values_from_roi(ss_speed_roi, mode="speed", display_transformed=display_rois, debug=debug)
        altitude_data = extract_values_from_roi(ss_altitude_roi, mode="altitude", display_transformed=display_rois, debug=debug)
        
        if debug:
            logger.debug(f"Extracted Starship speed: {speed_data.get('value')}, altitude: {altitude_data.get('value')}")
            
            if speed_data.get('value') is None:
                logger.debug("No speed value found for Starship")
            if altitude_data.get('value') is None:
                logger.debug("No altitude value found for Starship")
        
        return {
            "speed": speed_data.get("value"),
            "altitude": altitude_data.get("value")
        }
    except Exception as e:
        logger.error(f"Error extracting Starship data: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"speed": None, "altitude": None}


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
    logger.debug("Extracting time data from ROI")
    
    if zero_time_met:
        if debug:
            logger.debug("Zero time already met, returning default zero time")
        return {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
    
    try:
        time_data = extract_values_from_roi(time_roi, mode="time", display_transformed=display_rois, debug=debug)
        
        if debug:
            if time_data:
                logger.debug(f"Extracted time: {time_data.get('sign')} " +
                           f"{time_data.get('hours', 0):02}:{time_data.get('minutes', 0):02}:{time_data.get('seconds', 0):02}")
                
                # Check if this is T-0 or T+0 time
                if time_data.get('hours', 0) == 0 and time_data.get('minutes', 0) == 0 and time_data.get('seconds', 0) == 0:
                    logger.debug("Found T-0/T+0 time point!")
            else:
                logger.debug("No time data extracted from time ROI")
        
        return time_data
    except Exception as e:
        logger.error(f"Error extracting time data: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


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
        if debug:
            logger.debug("Starship data incomplete, using Superheavy data as fallback")
            logger.debug(f"Before fallback - SS speed: {starship_data.get('speed')}, SS altitude: {starship_data.get('altitude')}")
            logger.debug(f"Fallback data - SH speed: {superheavy_data.get('speed')}, SH altitude: {superheavy_data.get('altitude')}")
        
        # Use Superheavy data selectively where Starship data is missing
        if not starship_data.get("speed"):
            starship_data["speed"] = superheavy_data.get("speed")
        if not starship_data.get("altitude"):
            starship_data["altitude"] = superheavy_data.get("altitude")

    # Extract time separately
    time_data = extract_time_data(time_roi, display_rois, debug, zero_time_met)
    
    # Extract fuel levels
    logger.debug("Extracting fuel levels")
    try:
        fuel_data = extract_fuel_levels(image, debug)
        
        # Add fuel level data to vehicle data
        superheavy_data["fuel"] = fuel_data["superheavy"]
        starship_data["fuel"] = fuel_data["starship"]
        
        if debug:
            logger.debug(f"Fuel levels - Superheavy: LOX {fuel_data['superheavy']['lox']['fullness']:.1f}%, " +
                        f"CH4 {fuel_data['superheavy']['ch4']['fullness']:.1f}%")
            logger.debug(f"Fuel levels - Starship: LOX {fuel_data['starship']['lox']['fullness']:.1f}%, " +
                        f"CH4 {fuel_data['starship']['ch4']['fullness']:.1f}%")
    except Exception as e:
        logger.error(f"Error extracting fuel levels: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # Add empty fuel data to avoid KeyError
        superheavy_data["fuel"] = {"lox": {"fullness": 0}, "ch4": {"fullness": 0}}
        starship_data["fuel"] = {"lox": {"fullness": 0}, "ch4": {"fullness": 0}}
    
    # Detect engine status
    logger.debug("Detecting engine status")
    try:
        engine_data = detect_engine_status(image, debug)
        
        # Add engine data to vehicle data
        superheavy_data["engines"] = engine_data["superheavy"]
        starship_data["engines"] = engine_data["starship"]
        
        if debug:
            sh_active = sum(sum(1 for e in engines if e) for engines in engine_data["superheavy"].values())
            sh_total = sum(len(engines) for engines in engine_data["superheavy"].values())
            ss_active = sum(sum(1 for e in engines if e) for engines in engine_data["starship"].values())
            ss_total = sum(len(engines) for engines in engine_data["starship"].values())
            logger.debug(f"Engine status summary - Superheavy: {sh_active}/{sh_total} active, Starship: {ss_active}/{ss_total} active")
    except Exception as e:
        logger.error(f"Error detecting engine status: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        # Add empty engine data to avoid KeyError
        superheavy_data["engines"] = {}
        starship_data["engines"] = {}

    logger.debug("Data extraction complete")
    
    if debug:
        logger.debug(f"Final extracted data - Superheavy: speed={superheavy_data.get('speed')}, altitude={superheavy_data.get('altitude')}")
        logger.debug(f"Final extracted data - Starship: speed={starship_data.get('speed')}, altitude={starship_data.get('altitude')}")
        if time_data:
            logger.debug(f"Final extracted data - Time: {time_data.get('sign')} {time_data.get('hours', 0):02}:{time_data.get('minutes', 0):02}:{time_data.get('seconds', 0):02}")
        else:
            logger.debug("Final extracted data - Time: None")
    
    return superheavy_data, starship_data, time_data
