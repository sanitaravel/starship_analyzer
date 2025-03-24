import re
import easyocr
import numpy as np
import torch
from typing import Dict, Tuple, Optional

# Initialize EasyOCR only when needed, not at module level
# This helps prevent CUDA initialization issues with multiprocessing
reader = None

def get_reader():
    """
    Get or initialize the EasyOCR reader.
    
    Returns:
        easyocr.Reader: An initialized EasyOCR reader instance
    """
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return reader

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
    try:
        # Get the EasyOCR reader
        ocr_reader = get_reader()
        
        # Use EasyOCR to extract text
        # allowlist limits the characters that can be detected
        results = ocr_reader.readtext(roi, detail=0, 
                                 allowlist='0123456789T+-:')
        
        # Join all detected text segments
        text = ' '.join(results) if results else ""
        
        if debug:
            print(f"Raw OCR result: {text}")
            
    except Exception as e:
        print(f"Program failed to read text: {str(e)}")
        return {}
    
    # Clean the OCR result
    cleaned_text = clean_ocr_result(text)
    
    # Display the OCR result for debugging
    if debug:
        print(f"OCR Result for ROI: {cleaned_text}")
    
    if mode == "speed":
        # Extract just the speed value
        speed = extract_single_value(cleaned_text)
        return {"value": speed}
    elif mode == "altitude":
        # Extract just the altitude value
        altitude = extract_single_value(cleaned_text)
        return {"value": altitude}
    elif mode == "time":
        # Extract time from the cleaned text
        time = extract_time(cleaned_text)
        return time
    else:
        return {}

def extract_single_value(text: str) -> Optional[int]:
    """
    Extract a single numeric value from the cleaned text.
    
    Args:
        text (str): The cleaned text.
        
    Returns:
        Optional[int]: The extracted numeric value, or None if no value was found.
    """
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return None

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