import re
import easyocr
import numpy as np
import torch
import os
import threading
from typing import Dict, Tuple, Optional

# Use thread-local storage to ensure each process has its own reader instance
# This helps avoid sharing CUDA contexts across processes
_thread_local = threading.local()

def get_reader():
    """
    Get or initialize the EasyOCR reader safely for multiprocessing.
    
    Returns:
        easyocr.Reader: An initialized EasyOCR reader instance
    """
    if not hasattr(_thread_local, 'reader'):
        # Get process ID for debugging
        pid = os.getpid()
        print(f"Process {pid}: Initializing EasyOCR reader")
        
        # Check if CUDA is available
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            # Try to explicitly set the device to avoid conflicts
            try:
                torch.cuda.set_device(0)
                device_name = torch.cuda.get_device_name(0)
                print(f"Process {pid}: Using GPU: {device_name}")
            except Exception as e:
                print(f"Process {pid}: Error setting CUDA device: {e}")
                # Fall back to CPU if there's an issue with GPU
                gpu_available = False
                
        # Initialize the reader
        try:
            _thread_local.reader = easyocr.Reader(['en'], gpu=gpu_available, verbose=False)
            print(f"Process {pid}: EasyOCR initialized {'with GPU' if gpu_available else 'on CPU'}")
        except Exception as e:
            print(f"Process {pid}: Failed to initialize EasyOCR with GPU, falling back to CPU: {e}")
            # Fall back to CPU mode if GPU initialization fails
            try:
                _thread_local.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print(f"Process {pid}: EasyOCR initialized on CPU (fallback)")
            except Exception as e2:
                print(f"Process {pid}: Critical error initializing EasyOCR: {e2}")
                raise
    
    return _thread_local.reader

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
        # Get the EasyOCR reader safely
        ocr_reader = get_reader()
        
        # Guard against empty or invalid ROIs
        if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            if debug:
                print("Warning: Empty or invalid ROI provided to OCR")
            return {}
        
        # Use EasyOCR to extract text with appropriate parameters for each mode
        allowlist = '0123456789T+-:' if mode == "time" else '0123456789'
        
        # Process the image with error handling
        try:
            results = ocr_reader.readtext(roi, detail=0, allowlist=allowlist)
            text = ' '.join(results) if results else ""
        except RuntimeError as e:
            # Handle CUDA out-of-memory errors
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error in OCR. Falling back to CPU.")
                # Try again with CPU
                torch.cuda.empty_cache()  # Clear CUDA memory
                _thread_local.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                results = _thread_local.reader.readtext(roi, detail=0, allowlist=allowlist)
                text = ' '.join(results) if results else ""
            else:
                raise
        
        if debug:
            print(f"Raw OCR result for {mode}: {text}")
            
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return {}
    
    # Clean the OCR result
    cleaned_text = clean_ocr_result(text)
    
    if debug:
        print(f"Cleaned OCR result for {mode}: {cleaned_text}")
    
    # Process according to mode
    if mode == "speed":
        speed = extract_single_value(cleaned_text)
        return {"value": speed}
    elif mode == "altitude":
        altitude = extract_single_value(cleaned_text)
        return {"value": altitude}
    elif mode == "time":
        time = extract_time(cleaned_text)
        return time if time else {}
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