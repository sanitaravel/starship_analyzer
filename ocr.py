import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, Tuple, Optional
from utils import display_image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_values_from_roi(roi: np.ndarray, mode: str = "data", display_transformed: bool = False, debug: bool = False) -> Dict:
    """
    Extract values from a region of interest (ROI) in an image.

    Args:
        roi (numpy.ndarray): The region of interest in the image.
        mode (str): The mode of extraction ("data" or "time").
        display_transformed (bool): Whether to display the transformed ROI.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing the extracted values.
    """
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Remove antialiasing by applying a bilateral filter
    roi = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
    
    if display_transformed:
        display_image(roi, "Antialiased ROI")
    
    # Increase sharpness of the ROI
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    roi = cv2.filter2D(roi, -1, kernel)
    
    if display_transformed:
        display_image(roi, "Sharpened ROI")
    
    # Mask everything out besides the color FAF8FA and its closest ones
    lower_bound = np.array([230, 230, 230])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(roi, lower_bound, upper_bound)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
    
    # Display the transformed image if requested
    if display_transformed:
        display_image(gray, "Transformed ROI")
    
    # Use Tesseract to do OCR on the ROI
    text = pytesseract.image_to_string(gray).lower()
    
    # Clean the OCR result
    cleaned_text = clean_ocr_result(text)
    
    # Display the OCR result for debugging
    if debug:
        print(f"OCR Result for ROI: {cleaned_text}")
    
    if mode == "data":
        # Extract speed and altitude from the cleaned text
        speed, altitude = extract_speed_and_altitude(cleaned_text)
        return {"speed": speed, "altitude": altitude}
    elif mode == "time":
        # Extract time from the cleaned text
        time = extract_time(cleaned_text)
        return time

def clean_ocr_result(text: str) -> str:
    """
    Clean the OCR result by removing unwanted characters.

    Args:
        text (str): The OCR result text.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r'[^0-9\s+-:]', '', text)
    return cleaned_text

def extract_speed_and_altitude(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract speed and altitude from the cleaned text.

    Args:
        text (str): The cleaned text.

    Returns:
        tuple: A tuple containing the extracted speed and altitude.
    """
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def extract_time(text: str) -> Optional[Dict[str, int]]:
    """
    Extract time from the cleaned text.

    Args:
        text (str): The cleaned text.

    Returns:
        dict: A dictionary containing the extracted time.
    """
    match = re.search(r'[+-]\d{2}:\d{2}:\d{2}', text)
    if match:
        time_str = match.group(0)
        sign = time_str[0]
        hours, minutes, seconds = map(int, time_str[1:].split(':'))
        return {"sign": sign, "hours": hours, "minutes": minutes, "seconds": seconds}
    return None