import numpy as np
from typing import Tuple, Dict
from utils import display_image
from .ocr import extract_values_from_roi


def preprocess_image(image: np.ndarray, display_rois: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image to extract ROIs for Superheavy, Starship, and Time.

    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.

    Returns:
        tuple: A tuple containing the ROIs for Superheavy, Starship, and Time.
    """
    # Crop the image to exclude top and bottom black stripes
    height, width, _ = image.shape
    # Adjust this value to change the top cropping
    top_crop = int(height * (1 - 0.178423236515))
    bottom_crop = 0
    cropped_image = image[top_crop: height -
                          bottom_crop, :]  # Apply the cropping

    # Display the cropped image for debugging if requested
    if display_rois:
        display_image(cropped_image, "Cropped Image")

    # Define ROIs for Superheavy, Starship, and Time
    # height = 86, width = 858
    height, width, _ = cropped_image.shape

    superheavy_roi_left_edge = int(width * 0.179487179487)
    superheavy_roi_right_edge = width - int(width * 0.765734265734)
    superheavy_roi_bottom_edge = height - int(height * 0.5116279070)
    superheavy_roi = cropped_image[:superheavy_roi_bottom_edge,
                                   superheavy_roi_left_edge: superheavy_roi_right_edge]

    starship_roi_left_edge = int(width * 0.7937062937)
    starship_roi_right_edge = width - int(width * 0.1503496503)
    starship_roi_bottom_edge = height - int(height * 0.5116279070)
    starship_roi = cropped_image[: starship_roi_bottom_edge,
                                 starship_roi_left_edge: starship_roi_right_edge]

    time_roi_width = 120
    time_roi_bottom_edge = height - int(height * 0.4516279070)
    time_roi = cropped_image[:time_roi_bottom_edge, width // 2 - time_roi_width // 2:  width // 2 + time_roi_width // 2]

    # Display ROIs for debugging if requested
    if display_rois:
        display_image(superheavy_roi, "Superheavy ROI")
        display_image(starship_roi, "Starship ROI")
        display_image(time_roi, "Time ROI")

    return superheavy_roi, starship_roi, time_roi


def extract_superheavy_data(superheavy_roi: np.ndarray, display_rois: bool, debug: bool) -> Dict:
    """
    Extract data for Superheavy from the ROI.

    Args:
        superheavy_roi (numpy.ndarray): The ROI for Superheavy.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing the extracted data for Superheavy.
    """
    return extract_values_from_roi(superheavy_roi, mode="data", display_transformed=display_rois, debug=debug)


def extract_starship_data(starship_roi: np.ndarray, display_rois: bool, debug: bool) -> Dict:
    """
    Extract data for Starship from the ROI.

    Args:
        starship_roi (numpy.ndarray): The ROI for Starship.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing the extracted data for Starship.
    """
    return extract_values_from_roi(starship_roi, mode="data", display_transformed=display_rois, debug=debug)


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
        return {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
    return extract_values_from_roi(time_roi, mode="time", display_transformed=display_rois, debug=debug)


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
    # Preprocess the image to get ROIs
    superheavy_roi, starship_roi, time_roi = preprocess_image(
        image, display_rois)

    # Extract data for Superheavy
    superheavy_data = extract_superheavy_data(
        superheavy_roi, display_rois, debug)

    # Extract data for Starship
    starship_data = extract_starship_data(starship_roi, display_rois, debug)

    # If Starship extraction returns nothing, give it Superheavy data
    if not starship_data.get("speed") or not starship_data.get("altitude"):
        starship_data = superheavy_data

    # Extract time separately
    time_data = extract_time_data(time_roi, display_rois, debug, zero_time_met)

    return superheavy_data, starship_data, time_data
