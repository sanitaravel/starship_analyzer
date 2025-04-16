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

@njit
def process_strip_numba(gray_img: np.ndarray, x: int, y: int, ref_x: int, ref_y: int, ref_x2: int, ref_y2: int, strip_length: int, strip_height: int, brightness_threshold: float, ref_diff_threshold: float) -> Dict:
    """
    Optimized version of process_strip using numba for performance.
    """
    h, w = gray_img.shape
    if not (0 <= ref_y < h and 0 <= ref_x < w and 0 <= ref_y2 < h and 0 <= ref_x2 < w):
        return {"fullness": 0.0, "length": 0, "ref_diff": 0.0}

    ref_pixel1 = gray_img[ref_y, ref_x]
    ref_pixel2 = gray_img[ref_y2, ref_x2]

    min_val = gray_img.min()
    max_val = gray_img.max()
    ptp_val = max_val - min_val or 1  # Replace np.ptp() with max - min
    ref_pixel1_norm = (ref_pixel1 - min_val) / ptp_val
    ref_pixel2_norm = (ref_pixel2 - min_val) / ptp_val

    pixel_diff = abs(ref_pixel2_norm - ref_pixel1_norm)
    if pixel_diff <= ref_diff_threshold:
        return {"fullness": 0.0, "length": 0, "ref_diff": pixel_diff}

    y_start = max(0, y - strip_height // 2)
    y_end = min(h, y + strip_height // 2 + 1)
    x_end = min(w, x + strip_length)

    strip = gray_img[y_start:y_end, x:x_end]
    
    # Compute the mean along the vertical axis manually
    brightness_profile = np.zeros(strip.shape[1], dtype=np.float64)
    for col in range(strip.shape[1]):
        brightness_profile[col] = strip[:, col].sum() / strip.shape[0]

    norm_brightness = (brightness_profile - brightness_profile.min()) / (brightness_profile.max() - brightness_profile.min() or 1)

    bright_pixels = norm_brightness > brightness_threshold
    bright_indices = np.where(bright_pixels)[0]
    if len(bright_indices) > 0:
        rightmost_pos = bright_indices.max()
        effective_length = rightmost_pos + 1
        fullness_percentage = (rightmost_pos / strip_length) * 100 if strip_length > 0 else 0
    else:
        effective_length = 0
        fullness_percentage = 0

    return {"fullness": fullness_percentage, "length": effective_length, "ref_diff": pixel_diff}

def process_strip(gray_img: np.ndarray, strip_idx: int, debug: bool = False) -> Dict:
    """
    Process a single fuel level strip from the image.
    """
    if strip_idx < 0 or strip_idx > 3:
        logger.error(f"Invalid strip index: {strip_idx}, must be 0-3")
        return {"fullness": 0.0, "length": 0}

    x, y = STRIP_COORDS[strip_idx]
    ref_x, ref_y = REF_PIXEL_COORDS[strip_idx]
    ref_x2 = ref_x + 5 if strip_idx in [0, 2] else ref_x - 5
    ref_y2 = ref_y

    result = process_strip_numba(
        gray_img, x, y, ref_x, ref_y, ref_x2, ref_y2, STRIP_LENGTH, STRIP_HEIGHT, BRIGHTNESS_THRESHOLD, REF_DIFF_THRESHOLD
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
