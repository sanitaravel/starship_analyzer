import numpy as np
import cv2
from typing import Dict, Tuple
from utils import display_image
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def extract_fuel_rois(image: np.ndarray, display_rois: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract regions of interest for fuel levels from the image.
    
    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.
        
    Returns:
        tuple: A tuple containing the ROIs for Superheavy LOX, Superheavy CH4, Starship LOX, and Starship CH4.
    """
    logger.debug("Extracting fuel level ROIs from image")
    
    try:
        # Define ROIs based on the provided coordinates and dimensions
        # Reduced by 1px from top and bottom (original height was 13px, now 11px)
        sh_lox_roi = image[1001:1012, 274:274+241]  # LOX Superheavy level: 274*1001, size 241*11
        sh_ch4_roi = image[1036:1047, 274:274+241]  # CH4 Superheavy level: 274*1036, size 241*11
        ss_lox_roi = image[1001:1012, 1454:1454+241]  # LOX Starship level: 1454*1001, size 241*11
        ss_ch4_roi = image[1031:1042, 1454:1454+241]  # CH4 Starship level: 1454*1031, size 241*11

        logger.debug(f"ROI dimensions - SH LOX: {sh_lox_roi.shape}, SH CH4: {sh_ch4_roi.shape}, " +
                    f"SS LOX: {ss_lox_roi.shape}, SS CH4: {ss_ch4_roi.shape}")

        # Display ROIs for debugging if requested
        if display_rois:
            logger.debug("Displaying fuel level ROIs for visual inspection")
            display_image(sh_lox_roi, "Superheavy LOX Level ROI")
            display_image(sh_ch4_roi, "Superheavy CH4 Level ROI")
            display_image(ss_lox_roi, "Starship LOX Level ROI")
            display_image(ss_ch4_roi, "Starship CH4 Level ROI")
    
    except Exception as e:
        logger.error(f"Error extracting fuel level ROIs: {str(e)}")
        logger.debug(f"Image shape: {image.shape if image is not None else 'None'}")
        # Return empty ROIs in case of error
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty_roi, empty_roi, empty_roi, empty_roi

    return sh_lox_roi, sh_ch4_roi, ss_lox_roi, ss_ch4_roi

def analyze_fuel_level(roi: np.ndarray, display_transformed: bool = False, debug: bool = False) -> float:
    """
    Analyze the fuel level ROI to determine the percentage of fuel remaining.dth.
    Finds the rightmost white pixel and calculates its position as a percentage of the ROI width.
    
    Args:
        roi (numpy.ndarray): The region of interest containing the fuel level visualization.
        display_transformed (bool): Whether to display the transformed image for debugging.
        debug (bool): Whether to enable debug prints.
        
    Returns:
        float: The estimated fuel level percentage (0-100) or None if detection fails.
    """
    logger.debug("Analyzing fuel level ROI")
    
    try:
        # Convert to grayscale
        if len(roi.shape) == 3:  # If it's a color image
            gray = np.mean(roi, axis=2).astype(np.uint8)
        else:
            gray = roi.copy()
            
        # Apply thresholding to find the fuel level indicator
        # Values below 50 indicate empty fuel sections
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        if display_transformed:
            display_image(binary, "Binary Fuel Level")
            display_image(gray, "Grayscale Fuel Level")
        
        # Get the width of the ROI
        width = binary.shape[1]
        
        if width == 0:
            return None
        
        # Find the rightmost white pixel
        rightmost_x = 0
        for x in range(width-1, -1, -1):  # Start from the right side
            # Check if any pixel in this column is white
            if np.any(binary[:, x] > 0):
                rightmost_x = x
                break
        
        # Calculate fuel level as percentage of width
        fuel_percentage = (rightmost_x + 1) / width * 100
        
        if debug:
            logger.debug(f"Fuel level analysis: rightmost white pixel at x={rightmost_x}, total width={width}")
            logger.debug(f"Calculated fuel level: {fuel_percentage:.2f}%")
            
        return round(fuel_percentage, 2)
        
    except Exception as e:
        logger.error(f"Error analyzing fuel level: {str(e)}")
        if debug:
            import traceback
            logger.debug(traceback.format_exc())
        return None

def detect_fuel_levels(image: np.ndarray, display_rois: bool = False, debug: bool = False) -> Dict:
    """
    Detect fuel levels for both Superheavy and Starship.
    
    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.
        
    Returns:
        dict: A dictionary containing the fuel level data for Superheavy and Starship.
    """
    logger.debug("Starting fuel level detection")
    
    # Extract the fuel level ROIs
    sh_lox_roi, sh_ch4_roi, ss_lox_roi, ss_ch4_roi = extract_fuel_rois(image, display_rois)
    
    # Analyze each ROI to get the fuel level
    sh_lox_level = analyze_fuel_level(sh_lox_roi, display_rois, debug)
    sh_ch4_level = analyze_fuel_level(sh_ch4_roi, display_rois, debug)
    ss_lox_level = analyze_fuel_level(ss_lox_roi, display_rois, debug)
    ss_ch4_level = analyze_fuel_level(ss_ch4_roi, display_rois, debug)
    
    # Check if the indicator pixels are white (>= 250)
    sh_indicator_valid = False
    ss_indicator_valid = False
    
    try:
        # Check Superheavy indicator pixel (255, 1006)
        sh_indicator_pixel = image[1006, 255]
        sh_indicator_value = np.mean(sh_indicator_pixel) if len(sh_indicator_pixel.shape) > 0 else sh_indicator_pixel
        sh_indicator_valid = sh_indicator_value >= 250
        
        # Check Starship indicator pixel (1435, 1006)
        ss_indicator_pixel = image[1006, 1435]
        ss_indicator_value = np.mean(ss_indicator_pixel) if len(ss_indicator_pixel.shape) > 0 else ss_indicator_pixel
        ss_indicator_valid = ss_indicator_value >= 250
        
        if debug:
            logger.debug(f"Indicator pixel values - SH: {sh_indicator_value}, valid: {sh_indicator_valid}")
            logger.debug(f"Indicator pixel values - SS: {ss_indicator_value}, valid: {ss_indicator_valid}")
    
    except Exception as e:
        logger.error(f"Error checking indicator pixels: {str(e)}")
        if debug:
            import traceback
            logger.debug(traceback.format_exc())
    
    # Only keep fuel data if indicator pixels are valid
    if not sh_indicator_valid:
        logger.debug("Superheavy fuel indicator not valid, setting fuel levels to None")
        sh_lox_level = None
        sh_ch4_level = None
    
    if not ss_indicator_valid:
        logger.debug("Starship fuel indicator not valid, setting fuel levels to None")
        ss_lox_level = None
        ss_ch4_level = None
    
    if debug:
        logger.debug(f"Final fuel levels - SH LOX: {sh_lox_level}%, SH CH4: {sh_ch4_level}%, " +
                     f"SS LOX: {ss_lox_level}%, SS CH4: {ss_ch4_level}%")
    
    # Return the results in a structured format
    return {
        "superheavy": {
            "lox": sh_lox_level,
            "ch4": sh_ch4_level
        },
        "starship": {
            "lox": ss_lox_level,
            "ch4": ss_ch4_level
        }
    }
