import numpy as np
from typing import Tuple, Dict
from utils import display_image
from .ocr import extract_values_from_roi
from .engine_detection import detect_engine_status
from .fuel_level_extraction import extract_fuel_levels
from utils.logger import get_logger
import traceback  # Import at the top level instead of in exception handler

# Initialize logger
logger = get_logger(__name__)

# ROI coordinates and dimensions as constants (y_start, height, x_start, width)
SH_SPEED_ROI = (913, 25, 359, 83)
SH_ALT_ROI = (948, 25, 392, 50)
SS_SPEED_ROI = (913, 25, 1539, 83)
SS_ALT_ROI = (948, 25, 1572, 50)
TIME_ROI = (940, 44, 860, 197)

# Calculate required dimensions
REQUIRED_HEIGHT = max(y + h for y, h, _, _ in (SH_SPEED_ROI, SH_ALT_ROI, SS_SPEED_ROI, SS_ALT_ROI, TIME_ROI))
REQUIRED_WIDTH = max(x + w for _, _, x, w in (SH_SPEED_ROI, SH_ALT_ROI, SS_SPEED_ROI, SS_ALT_ROI, TIME_ROI))

def preprocess_image(image: np.ndarray, display_rois: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image to extract ROIs for Superheavy Speed, Superheavy Altitude, Starship Speed, Starship Altitude, and Time.

    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.

    Returns:
        tuple: A tuple containing the ROIs for Superheavy Speed, Superheavy Altitude, Starship Speed, Starship Altitude, and Time.
    """
    # Helper function to create empty ROIs
    def create_empty_rois():
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty_roi, empty_roi, empty_roi, empty_roi, empty_roi

    # Input validation
    if image is None:
        logger.error("Input image is None")
        return create_empty_rois()
    
    logger.debug(f"Preprocessing image of shape {image.shape}")
    
    if image.shape[0] < REQUIRED_HEIGHT or image.shape[1] < REQUIRED_WIDTH:
        logger.error(f"Image dimensions ({image.shape[0]}x{image.shape[1]}) are too small for ROI extraction. " 
                     f"Minimum required: {REQUIRED_HEIGHT}x{REQUIRED_WIDTH}")
        return create_empty_rois()
    
    try:
        # Extract ROIs (y:y+h, x:x+w format)
        sh_speed_roi = image[SH_SPEED_ROI[0]:SH_SPEED_ROI[0]+SH_SPEED_ROI[1], 
                            SH_SPEED_ROI[2]:SH_SPEED_ROI[2]+SH_SPEED_ROI[3]]
        
        sh_altitude_roi = image[SH_ALT_ROI[0]:SH_ALT_ROI[0]+SH_ALT_ROI[1], 
                               SH_ALT_ROI[2]:SH_ALT_ROI[2]+SH_ALT_ROI[3]]
        
        ss_speed_roi = image[SS_SPEED_ROI[0]:SS_SPEED_ROI[0]+SS_SPEED_ROI[1], 
                            SS_SPEED_ROI[2]:SS_SPEED_ROI[2]+SS_SPEED_ROI[3]]
        
        ss_altitude_roi = image[SS_ALT_ROI[0]:SS_ALT_ROI[0]+SS_ALT_ROI[1], 
                               SS_ALT_ROI[2]:SS_ALT_ROI[2]+SS_ALT_ROI[3]]
        
        time_roi = image[TIME_ROI[0]:TIME_ROI[0]+TIME_ROI[1], 
                        TIME_ROI[2]:TIME_ROI[2]+TIME_ROI[3]]
        
        # Debug logging
        logger.debug(f"ROI dimensions - SH Speed: {sh_speed_roi.shape}, SH Alt: {sh_altitude_roi.shape}, " +
                    f"SS Speed: {ss_speed_roi.shape}, SS Alt: {ss_altitude_roi.shape}, Time: {time_roi.shape}")

        # Display ROIs if requested
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
        return create_empty_rois()

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
    if debug:
        logger.debug("Extracting Superheavy data from ROIs")
    
    try:
        # Extract values from ROIs
        speed_data = extract_values_from_roi(sh_speed_roi, mode="speed", 
                                            display_transformed=display_rois, debug=debug)
        altitude_data = extract_values_from_roi(sh_altitude_roi, mode="altitude", 
                                               display_transformed=display_rois, debug=debug)
        
        # Store values directly to avoid repeated dictionary lookups
        speed_value = speed_data.get("value")
        altitude_value = altitude_data.get("value")
        
        # Debug logging
        if debug:
            missing = []
            if speed_value is None:
                missing.append("speed")
            if altitude_value is None:
                missing.append("altitude")
                
            logger.debug(f"Extracted Superheavy speed: {speed_value}, altitude: {altitude_value}")
            
            if missing:
                logger.debug(f"Missing Superheavy data: {', '.join(missing)}")
        
        # Return extracted data
        return {"speed": speed_value, "altitude": altitude_value}
    
    except Exception as e:
        logger.error(f"Error extracting Superheavy data: {str(e)}")
        if debug:
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
    if debug:
        logger.debug("Extracting Starship data from ROIs")
    
    try:
        # Extract values from ROIs
        speed_data = extract_values_from_roi(ss_speed_roi, mode="speed", 
                                           display_transformed=display_rois, debug=debug)
        altitude_data = extract_values_from_roi(ss_altitude_roi, mode="altitude", 
                                              display_transformed=display_rois, debug=debug)
        
        # Store values directly to avoid repeated dictionary lookups
        speed_value = speed_data.get("value")
        altitude_value = altitude_data.get("value")
        
        # Debug logging
        if debug:
            missing = []
            if speed_value is None:
                missing.append("speed")
            if altitude_value is None:
                missing.append("altitude")
                
            logger.debug(f"Extracted Starship speed: {speed_value}, altitude: {altitude_value}")
            
            if missing:
                logger.debug(f"Missing Starship data: {', '.join(missing)}")
        
        # Return extracted data
        return {"speed": speed_value, "altitude": altitude_value}
    
    except Exception as e:
        logger.error(f"Error extracting Starship data: {str(e)}")
        if debug:
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
    if debug:
        logger.debug("Starting data extraction from image")
    
    # Preprocess the image to get ROIs
    sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(
        image, display_rois)

    # Extract data for Superheavy and Starship
    superheavy_data = extract_superheavy_data(sh_speed_roi, sh_altitude_roi, display_rois, debug)
    starship_data = extract_starship_data(ss_speed_roi, ss_altitude_roi, display_rois, debug)

    # Cache values to avoid repeated dictionary lookups
    sh_speed = superheavy_data.get("speed")
    sh_altitude = superheavy_data.get("altitude")
    ss_speed = starship_data.get("speed")
    ss_altitude = starship_data.get("altitude")
    
    # Handle Starship data fallback
    if not ss_speed or not ss_altitude:
        if debug:
            logger.debug("Starship data incomplete, using Superheavy data as fallback")
            logger.debug(f"Before fallback - SS speed: {ss_speed}, SS altitude: {ss_altitude}")
            logger.debug(f"Fallback data - SH speed: {sh_speed}, SH altitude: {sh_altitude}")

        # Use Superheavy data selectively where Starship data is missing
        if not ss_speed:
            starship_data["speed"] = sh_speed
        if not ss_altitude:
            starship_data["altitude"] = sh_altitude

    # Extract time data
    time_data = extract_time_data(time_roi, display_rois, debug, zero_time_met)
    
    # Extract fuel levels
    if debug:
        logger.debug("Extracting fuel levels")
    
    try:
        fuel_data = extract_fuel_levels(image, debug)
        
        # Add fuel level data to vehicle data
        superheavy_data["fuel"] = fuel_data["superheavy"]
        starship_data["fuel"] = fuel_data["starship"]
        
        if debug:
            # Cache fuel values for logging
            sh_lox = fuel_data["superheavy"]["lox"]["fullness"]
            sh_ch4 = fuel_data["superheavy"]["ch4"]["fullness"]
            ss_lox = fuel_data["starship"]["lox"]["fullness"]
            ss_ch4 = fuel_data["starship"]["ch4"]["fullness"]
            logger.debug(f"Fuel levels - SH: LOX {sh_lox:.1f}%, CH4 {sh_ch4:.1f}%, "
                        f"SS: LOX {ss_lox:.1f}%, CH4 {ss_ch4:.1f}%")
    except Exception as e:
        logger.error(f"Error extracting fuel levels: {str(e)}")
        if debug:
            logger.debug(traceback.format_exc())
        # Reuse same empty dict for both to reduce allocations
        empty_fuel = {"lox": {"fullness": 0}, "ch4": {"fullness": 0}}
        superheavy_data["fuel"] = empty_fuel
        starship_data["fuel"] = empty_fuel
    
    # Detect engine status
    if debug:
        logger.debug("Detecting engine status")
    
    try:
        engine_data = detect_engine_status(image, debug)
        
        # Add engine data to vehicle data
        superheavy_data["engines"] = engine_data["superheavy"]
        starship_data["engines"] = engine_data["starship"]
        
        if debug:
            # Store engine data references for more efficient processing
            sh_engines = engine_data["superheavy"]
            ss_engines = engine_data["starship"]
            
            # Calculate engine stats more efficiently
            sh_active = sum(sum(1 for e in engines if e) for engines in sh_engines.values())
            sh_total = sum(len(engines) for engines in sh_engines.values())
            ss_active = sum(sum(1 for e in engines if e) for engines in ss_engines.values())
            ss_total = sum(len(engines) for engines in ss_engines.values())
            
            logger.debug(f"Engine status - SH: {sh_active}/{sh_total} active, SS: {ss_active}/{ss_total} active")
    except Exception as e:
        logger.error(f"Error detecting engine status: {str(e)}")
        if debug:
            logger.debug(traceback.format_exc())
        # Add empty engine data to avoid KeyError
        superheavy_data["engines"] = {}
        starship_data["engines"] = {}

    # Updated Starship speed/altitude values after possible fallback
    if debug:
        logger.debug("Data extraction complete")
        ss_speed = starship_data.get("speed")  # Get updated value after fallback
        ss_altitude = starship_data.get("altitude")  # Get updated value after fallback
        logger.debug(f"Final data - SH: speed={sh_speed}, altitude={sh_altitude}")
        logger.debug(f"Final data - SS: speed={ss_speed}, altitude={ss_altitude}")
        
        if time_data:
            # Format time string more efficiently
            sign = time_data.get("sign", "")
            h = time_data.get("hours", 0)
            m = time_data.get("minutes", 0)
            s = time_data.get("seconds", 0)
            logger.debug(f"Final data - Time: {sign} {h:02}:{m:02}:{s:02}")
        else:
            logger.debug("Final data - Time: None")
    
    return superheavy_data, starship_data, time_data
