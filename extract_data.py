import cv2
import numpy as np
import pytesseract
from typing import Tuple, Dict
from ocr import extract_values_from_roi
from utils import display_image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_data(image: np.ndarray, display_rois: bool = False, debug: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Extract data from an image.

    Args:
        image (numpy.ndarray): The image to process.
        display_rois (bool): Whether to display the ROIs.
        debug (bool): Whether to enable debug prints.

    Returns:
        tuple: A tuple containing the extracted data for Superheavy, Starship, and Time.
    """
    # Crop the image to exclude top and bottom black stripes
    height, width, _ = image.shape
    bottom_crop = top_crop = height // 3 + 75  # Adjust this value to change the top cropping
    cropped_image = image[top_crop : height - bottom_crop, :]  # Apply the cropping
    
    # Display the cropped image for debugging if requested
    if display_rois:
        display_image(cropped_image, "Cropped Image")
    
    # Define ROIs for Superheavy, Starship, and Time
    superheavy_roi = cropped_image[:, : cropped_image.shape[1] // 4]
    starship_roi = cropped_image[:, cropped_image.shape[1] // 4 * 3 :]
    time_roi = cropped_image[:, cropped_image.shape[1] // 2 - 150 : cropped_image.shape[1] // 2 + 150]  # Adjust as needed
    
    # Display ROIs for debugging if requested
    if display_rois:
        display_image(superheavy_roi, "Superheavy ROI")
        display_image(starship_roi, "Starship ROI")
        display_image(time_roi, "Time ROI")
    
    # Extract data for Superheavy
    superheavy_data = extract_values_from_roi(superheavy_roi, mode="data", display_transformed=display_rois, debug=debug)
    
    # Extract data for Starship
    starship_data = extract_values_from_roi(starship_roi, mode="data", display_transformed=display_rois, debug=debug)
    
    # Extract time separately
    time_data = extract_values_from_roi(time_roi, mode="time", display_transformed=display_rois, debug=debug)
    
    return superheavy_data, starship_data, time_data