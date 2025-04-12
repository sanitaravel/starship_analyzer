import cv2
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Check if CUDA is available for GPU acceleration
try:
    cv2.cuda.getCudaEnabledDeviceCount()
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        logger.info(f"CUDA is available for fuel level extraction with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    else:
        logger.info("CUDA is not available for fuel level extraction, using CPU")
except:
    CUDA_AVAILABLE = False
    logger.info("CUDA not available in OpenCV, using CPU for fuel level extraction")

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

def process_strip(gray_img: np.ndarray, strip_idx: int, debug: bool = False) -> Dict:
    """
    Process a single fuel level strip from the image.
    
    Args:
        gray_img (np.ndarray): Grayscale image
        strip_idx (int): Index of the strip to process (0-3)
        debug (bool): Whether to enable debug logging
        
    Returns:
        Dict: Dictionary with strip data including fullness percentage
    """
    if strip_idx < 0 or strip_idx > 3:
        logger.error(f"Invalid strip index: {strip_idx}, must be 0-3")
        return {"fullness": 0.0, "length": 0}
        
    # Get coordinates
    x, y = STRIP_COORDS[strip_idx]
    ref_x, ref_y = REF_PIXEL_COORDS[strip_idx]
    
    # Calculate second reference pixel position
    if strip_idx == 0 or strip_idx == 2:  # For strips 1 and 3, shift 5px to the right
        ref_x2 = ref_x + 5
        ref_y2 = ref_y
    else:  # For strips 2 and 4, shift 5px to the left
        ref_x2 = ref_x - 5
        ref_y2 = ref_y
    
    # Check if reference pixels are within image bounds
    h, w = gray_img.shape
    if not (0 <= ref_y < h and 0 <= ref_x < w and 0 <= ref_y2 < h and 0 <= ref_x2 < w):
        if debug:
            logger.debug(f"Reference pixels out of bounds for strip {strip_idx+1}")
        return {"fullness": 0.0, "length": 0}
    
    # Get reference pixel values and normalize them
    ref_pixel1 = gray_img[ref_y, ref_x]
    ref_pixel2 = gray_img[ref_y2, ref_x2]
    
    # Normalize reference pixels
    min_val = gray_img.min()
    ptp_val = np.ptp(gray_img) or 1
    ref_pixel1_norm = (ref_pixel1 - min_val) / ptp_val
    ref_pixel2_norm = (ref_pixel2 - min_val) / ptp_val
    
    # Check if the difference between reference pixels is noticeable
    pixel_diff = abs(ref_pixel2_norm - ref_pixel1_norm)
    ref_is_active = pixel_diff > REF_DIFF_THRESHOLD
    
    if debug:
        logger.debug(f"Strip {strip_idx+1} - Reference pixels: {ref_pixel1_norm:.3f}, {ref_pixel2_norm:.3f}, diff: {pixel_diff:.3f}")
    
    # If reference check fails, return zeros
    if not ref_is_active:
        return {"fullness": 0.0, "length": 0, "ref_diff": pixel_diff}
    
    # Extract the strip
    y_start = max(0, y - STRIP_HEIGHT//2)
    y_end = min(h, y + STRIP_HEIGHT//2 + 1)
    x_end = min(w, x + STRIP_LENGTH)
    
    strip = gray_img[y_start:y_end, x:x_end]
    
    # Average brightness across the strip vertically
    brightness_profile = strip.mean(axis=0)
    
    # Normalize brightness (0 to 1)
    norm_brightness = (brightness_profile - brightness_profile.min()) / (np.ptp(brightness_profile) or 1)
    
    # Find bright pixels
    bright_pixels = norm_brightness > BRIGHTNESS_THRESHOLD
    
    # Find the rightmost bright pixel
    bright_indices = np.where(bright_pixels)[0]
    if len(bright_indices) > 0:
        rightmost_pos = bright_indices.max()
        effective_length = rightmost_pos + 1  # +1 because indices are 0-based
        fullness_percentage = (rightmost_pos / STRIP_LENGTH) * 100 if STRIP_LENGTH > 0 else 0
    else:
        effective_length = 0
        fullness_percentage = 0
    
    if debug:
        logger.debug(f"Strip {strip_idx+1} - Length: {effective_length}, Fullness: {fullness_percentage:.1f}%")
    
    return {
        "fullness": fullness_percentage,
        "length": effective_length,
        "ref_diff": pixel_diff
    }

def apply_grouping_rules(results: List[Dict], current_time: float = 0) -> List[Dict]:
    """
    Apply grouping rules to the results. For bars 1-2 and 3-4, if difference > 30%,
    use max value if time < 200s, otherwise use min value.
    
    Args:
        results (List[Dict]): List of strip results
        current_time (float): Current time in seconds
        
    Returns:
        List[Dict]: Processed results with grouping rules applied
    """
    if len(results) != 4:
        logger.error(f"Expected 4 results for grouping, got {len(results)}")
        return results
    
    # Group 1: bars 0-1 (LOX and CH4 for Superheavy)
    if abs(results[0]["fullness"] - results[1]["fullness"]) > 30:
        # Use max value in first 200s, min value after
        if current_time < 200:
            chosen_value = max(results[0]["fullness"], results[1]["fullness"])
        else:
            chosen_value = min(results[0]["fullness"], results[1]["fullness"])
            
        # Update both bars in the group
        results[0]["fullness"] = chosen_value
        results[1]["fullness"] = chosen_value
        logger.debug(f"Group 1 (Superheavy): Applied grouping rule, using value {chosen_value:.1f}%")
    
    # Group 2: bars 2-3 (LOX and CH4 for Starship)
    if abs(results[2]["fullness"] - results[3]["fullness"]) > 30:
        # Use max value in first 200s, min value after
        if current_time < 200:
            chosen_value = max(results[2]["fullness"], results[3]["fullness"])
        else:
            chosen_value = min(results[2]["fullness"], results[3]["fullness"])
            
        # Update both bars in the group
        results[2]["fullness"] = chosen_value
        results[3]["fullness"] = chosen_value
        logger.debug(f"Group 2 (Starship): Applied grouping rule, using value {chosen_value:.1f}%")
    
    return results

def extract_fuel_levels(image: np.ndarray, current_time: float = 0, debug: bool = False) -> Dict:
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
            if CUDA_AVAILABLE:
                # GPU-accelerated grayscale conversion
                try:
                    # Upload image to GPU
                    gpu_image = cv2.cuda_GpuMat()
                    gpu_image.upload(image)
                    
                    # Convert to grayscale on GPU
                    gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                    
                    # Download result back to CPU
                    gray_img = gpu_gray.download()
                    
                    if debug:
                        logger.debug("Used GPU acceleration for grayscale conversion")
                except Exception as e:
                    logger.warning(f"GPU grayscale conversion failed, falling back to CPU: {str(e)}")
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # CPU fallback
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        # Process each strip
        strip_results = []
        for i in range(4):
            result = process_strip(gray_img, i, debug)
            strip_results.append(result)
            
        # Apply grouping rules
        processed_results = apply_grouping_rules(strip_results, current_time)
        
        # Create result dictionary
        fuel_data = {
            "superheavy": {
                "lox": {
                    "fullness": processed_results[0]["fullness"]
                },
                "ch4": {
                    "fullness": processed_results[1]["fullness"]
                }
            },
            "starship": {
                "lox": {
                    "fullness": processed_results[2]["fullness"]
                },
                "ch4": {
                    "fullness": processed_results[3]["fullness"]
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
