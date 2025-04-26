import cv2
import numpy as np
from typing import Dict, List, Tuple
from numba import njit
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Strip coordinates (x, y) and dimensions
STRIP_COORDS = [
    (275, 1007),  # Superheavy LOX
    (275, 1042),  # Superheavy CH4
    (1455, 1007), # Starship LOX
    (1455, 1037)  # Starship CH4
]
STRIP_LENGTH = 240
STRIP_HEIGHT = 1  # 1-pixel tall strips

# Reference pixel coordinates for each strip (used to determine if bar is active)
REF_PIXEL_COORDS = [
    (255, 1006),  # Superheavy LOX
    (227, 1042),  # Superheavy CH4
    (1435, 1006), # Starship LOX
    (1407, 1037)  # Starship CH4
]

# Threshold for brightness detection
BRIGHTNESS_THRESHOLD = 0.9
REF_DIFF_THRESHOLD = 0.2  # Threshold for reference pixel difference

# Pre-compute strip parameters for faster lookup
STRIP_PARAMS = [
    {
        'x': STRIP_COORDS[i][0],
        'y': STRIP_COORDS[i][1],
        'ref_x': REF_PIXEL_COORDS[i][0],
        'ref_y': REF_PIXEL_COORDS[i][1],
        'ref_x2': REF_PIXEL_COORDS[i][0] + (5 if i in [0, 2] else -5),
        'ref_y2': REF_PIXEL_COORDS[i][1]
    }
    for i in range(4)
]

@njit
def process_strip_numba(gray_img: np.ndarray, x: int, y: int, ref_x: int, ref_y: int, ref_x2: int, ref_y2: int, strip_length: int, strip_height: int, brightness_threshold: float, ref_diff_threshold: float) -> Dict:
    """
    Optimized version of process_strip using numba for performance.
    """
    h, w = gray_img.shape
    if not (0 <= ref_y < h and 0 <= ref_x < w and 0 <= ref_y2 < h and 0 <= ref_x2 < w):
        return {"fullness": 0.0, "length": 0, "ref_diff": 0.0}

    # Get reference pixels directly without extracting regions
    ref_pixel1 = float(gray_img[ref_y, ref_x])  # Convert to float to ensure type consistency
    ref_pixel2 = float(gray_img[ref_y2, ref_x2])
    
    # Calculate reference region for min/max (small area around reference pixels)
    ref_region_x_min = min(ref_x, ref_x2) - 5
    ref_region_x_max = max(ref_x, ref_x2) + 5
    ref_region_y_min = min(ref_y, ref_y2) - 5
    ref_region_y_max = max(ref_y, ref_y2) + 5
    
    # Ensure bounds
    ref_region_x_min = max(0, ref_region_x_min)
    ref_region_x_max = min(w, ref_region_x_max)
    ref_region_y_min = max(0, ref_region_y_min)
    ref_region_y_max = min(h, ref_region_y_max)
    
    # Extract smaller reference region for min/max calculation
    ref_region = gray_img[ref_region_y_min:ref_region_y_max, ref_region_x_min:ref_region_x_max]
    
    # Calculate min/max on the smaller region (explicit float conversions)
    min_val = float(np.min(ref_region))
    max_val = float(np.max(ref_region))
    ptp_val = max_val - min_val or 1.0
    
    # Normalize reference pixels
    ref_pixel1_norm = (ref_pixel1 - min_val) / ptp_val
    ref_pixel2_norm = (ref_pixel2 - min_val) / ptp_val

    pixel_diff = abs(ref_pixel2_norm - ref_pixel1_norm)
    if pixel_diff <= ref_diff_threshold:
        return {"fullness": 0.0, "length": 0, "ref_diff": pixel_diff}

    # Calculate strip bounds - only extract exact pixels needed
    y_start = max(0, y)
    y_end = min(h, y + strip_height)
    x_end = min(w, x + strip_length)
    
    # Only extract if within bounds
    if y_start < y_end and x < x_end:
        strip = gray_img[y_start:y_end, x:x_end]
        
        # Create float brightness profile (consistent type)
        brightness_profile = np.zeros(strip.shape[1], dtype=np.float64)
        
        # Fill the profile - always use explicit looping for type consistency with Numba
        for col in range(strip.shape[1]):
            for row in range(strip.shape[0]):
                brightness_profile[col] += float(strip[row, col])
            
            # Only divide if height > 1 to avoid division by 1
            if strip_height > 1:
                brightness_profile[col] /= float(strip.shape[0])
        
        # Simple min/max normalization
        min_brightness = 0.0
        max_brightness = 0.0
        
        # Find min/max manually to ensure type consistency
        if len(brightness_profile) > 0:
            min_brightness = brightness_profile[0]
            max_brightness = brightness_profile[0]
            
            for i in range(1, len(brightness_profile)):
                if brightness_profile[i] < min_brightness:
                    min_brightness = brightness_profile[i]
                if brightness_profile[i] > max_brightness:
                    max_brightness = brightness_profile[i]
        
        range_brightness = max_brightness - min_brightness
        if range_brightness == 0:
            range_brightness = 1.0
            
        # Normalize and find bright pixels
        bright_indices = []
        for i in range(len(brightness_profile)):
            norm_value = (brightness_profile[i] - min_brightness) / range_brightness
            if norm_value > brightness_threshold:
                bright_indices.append(i)
        
        # Find rightmost bright index
        if len(bright_indices) > 0:
            rightmost_pos = bright_indices[-1]
            fullness_percentage = (rightmost_pos / strip_length) * 100.0
            return {"fullness": fullness_percentage, "length": rightmost_pos + 1, "ref_diff": pixel_diff}
    
    # Default return if strip couldn't be processed
    return {"fullness": 0.0, "length": 0, "ref_diff": pixel_diff}

def process_strip(gray_img: np.ndarray, strip_idx: int, debug: bool = False) -> Dict:
    """
    Process a single fuel level strip from the image.
    """
    if strip_idx < 0 or strip_idx > 3:
        logger.error(f"Invalid strip index: {strip_idx}, must be 0-3")
        return {"fullness": 0.0, "length": 0}

    # Use pre-computed parameters for speed
    params = STRIP_PARAMS[strip_idx]

    result = process_strip_numba(
        gray_img, 
        params['x'], params['y'], 
        params['ref_x'], params['ref_y'], 
        params['ref_x2'], params['ref_y2'], 
        STRIP_LENGTH, STRIP_HEIGHT, 
        BRIGHTNESS_THRESHOLD, REF_DIFF_THRESHOLD
    )

    if debug:
        logger.debug(f"Strip {strip_idx+1} - Length: {result['length']}, Fullness: {result['fullness']:.1f}%, Ref Diff: {result['ref_diff']:.3f}")

    return result

def extract_fuel_levels(image: np.ndarray, debug: bool = False) -> Dict:
    """
    Extract fuel level information from an image.
    
    Args:
        image (np.ndarray): The input image
        current_time (float): Current time in seconds
        debug (bool): Enable debug logging
        
    Returns:
        Dict: Dictionary containing fuel level data
    """
    logger.debug("Extracting fuel levels from image")
    
    try:
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        # Process each strip
        strip_results = []
        for i in range(4):
            result = process_strip(gray_img, i, debug)
            strip_results.append(result)
        
        # Create result dictionary directly from strip results without applying grouping rules
        fuel_data = {
            "superheavy": {
                "lox": {
                    "fullness": strip_results[0]["fullness"]
                },
                "ch4": {
                    "fullness": strip_results[1]["fullness"]
                }
            },
            "starship": {
                "lox": {
                    "fullness": strip_results[2]["fullness"]
                },
                "ch4": {
                    "fullness": strip_results[3]["fullness"]
                }
            }
        }
        
        logger.debug(f"Extracted fuel levels - SH: LOX {fuel_data['superheavy']['lox']['fullness']:.1f}%, " +
                    f"CH4 {fuel_data['superheavy']['ch4']['fullness']:.1f}%, " +
                    f"SS: LOX {fuel_data['starship']['lox']['fullness']:.1f}%, " +
                    f"CH4 {fuel_data['starship']['ch4']['fullness']:.1f}%")
        
        return fuel_data
        
    except Exception as e:
        logger.error(f"Error extracting fuel levels: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Return empty data structure in case of error
        return {
            "superheavy": {"lox": {"fullness": 0}, "ch4": {"fullness": 0}},
            "starship": {"lox": {"fullness": 0}, "ch4": {"fullness": 0}}
        }
