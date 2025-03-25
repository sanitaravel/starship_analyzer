import numpy as np
from typing import Dict
from constants import SUPERHEAVY_ENGINES, STARSHIP_ENGINES, WHITE_THRESHOLD
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def check_engines(image: np.ndarray, engine_coords: Dict, debug: bool, engine_type: str) -> Dict:
    """
    Check the status of engines based on pixel values at specific coordinates.
    
    Args:
        image (numpy.ndarray): The image to process.
        engine_coords (Dict): Dictionary with engine sections as keys and coordinates as values.
        debug (bool): Whether to enable debug prints.
        engine_type (str): Type of engine ('Superheavy' or 'Starship') for debug messages.
        
    Returns:
        Dict: Dictionary with engine sections as keys and lists of boolean status as values.
    """
    # Initialize engine status dictionary
    engine_status = {section: [] for section in engine_coords.keys()}
    
    # Check engines
    for section, coordinates in engine_coords.items():
        for i, (x, y) in enumerate(coordinates):
            # Check if coordinates are within image boundaries
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                # Get pixel value
                pixel = image[y, x]
                # Check if all channels are above threshold (close to white)
                is_on = all(channel >= WHITE_THRESHOLD for channel in pixel)
                engine_status[section].append(is_on)
                if debug:
                    logger.debug(f"{engine_type} {section} engine {i+1}: {is_on} (pixel value: {pixel})")
            else:
                # If coordinates are out of bounds, consider engine off
                engine_status[section].append(False)
                if debug:
                    logger.warning(f"{engine_type} {section} engine {i+1}: Coordinates out of bounds")
                    
    return engine_status


def detect_engine_status(image: np.ndarray, debug: bool = False) -> Dict:
    """
    Detect whether engines are turned on by checking pixel values at specific coordinates.

    Args:
        image (numpy.ndarray): The image to process.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing engine statuses for Superheavy and Starship.
    """
    # Check Superheavy engines
    superheavy_engines = check_engines(image, SUPERHEAVY_ENGINES, debug, "Superheavy")
    
    # Check Starship engines
    starship_engines = check_engines(image, STARSHIP_ENGINES, debug, "Starship")
    
    return {
        "superheavy": superheavy_engines,
        "starship": starship_engines
    }
