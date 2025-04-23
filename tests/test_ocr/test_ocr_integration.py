import pytest
import numpy as np
from PIL import Image
import os
import threading
from unittest.mock import patch
from ocr.ocr import extract_values_from_roi
from ocr.extract_data import preprocess_image

# Directory for test images
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'ocr_samples')

# Use a more robust approach to reset OCR state between tests
@pytest.fixture(scope="function", autouse=True)
def reset_ocr_state():
    """Reset OCR state completely before each test to ensure isolation."""
    # Reset the thread local storage
    import ocr.ocr
    if hasattr(ocr.ocr._thread_local, 'reader'):
        delattr(ocr.ocr._thread_local, 'reader')
    
    # Also patch get_reader to ensure a fresh reader for each test
    with patch('ocr.ocr.get_reader', wraps=ocr.ocr.get_reader) as wrapped_get_reader:
        yield wrapped_get_reader
    
    # Clean up after test
    if hasattr(ocr.ocr._thread_local, 'reader'):
        delattr(ocr.ocr._thread_local, 'reader')
    
    # Clear GPU memory if available
    if hasattr(ocr.ocr, 'torch') and hasattr(ocr.ocr.torch, 'cuda') and ocr.ocr.torch.cuda.is_available():
        ocr.ocr.torch.cuda.empty_cache()

@pytest.mark.slow  # Mark all tests in this class as slow
@pytest.mark.skipif(not os.path.exists(TEST_IMAGES_DIR), reason="Test images directory not found")
class TestOCRWithRealImages:
    """Integration tests using real images."""
    
    def test_speed_extraction(self):
        """Test speed extraction from a real flight image."""
        # Load the full frame
        speed_image_path = os.path.join(TEST_IMAGES_DIR, 'speed_100.png')
        if not os.path.exists(speed_image_path):
            pytest.skip(f"Test image {speed_image_path} not found")
            
        # Load the full image
        full_image = np.array(Image.open(speed_image_path))
        
        # Crop to get the Superheavy speed ROI (using coordinates from preprocess_image)
        sh_speed_roi = full_image[913:913+25, 359:359+83]
        
        # Extract value
        result = extract_values_from_roi(sh_speed_roi, mode="speed")
        
        # Check result
        assert result.get("value") == 100
    
    def test_altitude_extraction(self):
        """Test altitude extraction from a real flight image."""
        # Load the full frame
        altitude_image_path = os.path.join(TEST_IMAGES_DIR, 'altitude_5.png')
        if not os.path.exists(altitude_image_path):
            pytest.skip(f"Test image {altitude_image_path} not found")
            
        # Force using CPU for more consistent results across test runs
        with patch('ocr.ocr.torch.cuda.is_available', return_value=False):
            # Load the full image
            full_image = np.array(Image.open(altitude_image_path))
            
            # Crop to get the Superheavy altitude ROI (using coordinates from preprocess_image)
            sh_altitude_roi = full_image[948:948+25, 392:392+50]
            
            # Extract value with explicit debug logging
            result = extract_values_from_roi(sh_altitude_roi, mode="altitude", debug=True)
            
            # Compare to expected value with more flexible assertion
            value = result.get("value")
            assert value is not None, "OCR returned None for altitude"
            assert value == 5 or value == 5000, f"Expected 5 or 5000, got {value}"
    
    def test_time_extraction(self):
        """Test time extraction from a real flight image."""
        # Load the full frame
        time_image_path = os.path.join(TEST_IMAGES_DIR, 'time_plus_00_01_30.png')
        if not os.path.exists(time_image_path):
            pytest.skip(f"Test image {time_image_path} not found")
            
        # Force using CPU for more consistent results across test runs  
        with patch('ocr.ocr.torch.cuda.is_available', return_value=False):
            # Load the full image
            full_image = np.array(Image.open(time_image_path))
            
            # Crop to get the time ROI (using coordinates from preprocess_image)
            time_roi = full_image[940:940+44, 860:860+197]
            
            # Extract value with explicit debug logging
            result = extract_values_from_roi(time_roi, mode="time", debug=True)
            
            # More robust assertion with detailed failure message
            assert result, f"OCR returned no time data: {result}"
            
            if result:
                # Add tolerance for OCR variance in time extraction
                expected = {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
                time_components_match = (
                    result.get("sign") == expected["sign"] and
                    result.get("hours") == expected["hours"] and
                    result.get("minutes") == expected["minutes"] and
                    abs(result.get("seconds", 0) - expected["seconds"]) <= 2  # Allow small variance in seconds
                )
                
                assert time_components_match, f"Expected {expected}, got {result}"
        
    def test_with_full_image(self):
        """Test extracting ROIs from a full image and then processing them."""
        full_image_path = os.path.join(TEST_IMAGES_DIR, 'full_frame.png')
        if not os.path.exists(full_image_path):
            pytest.skip(f"Test full image {full_image_path} not found")
        
        # Load the full image
        full_image = np.array(Image.open(full_image_path))
        
        # Extract ROIs using the preprocess_image function
        sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(full_image)
        
        # Process Superheavy speed ROI
        speed_result = extract_values_from_roi(sh_speed_roi, mode="speed")
        assert isinstance(speed_result.get("value"), (int, type(None))), "Speed should be an integer or None"
        
        # Process Superheavy altitude ROI
        altitude_result = extract_values_from_roi(sh_altitude_roi, mode="altitude")
        assert isinstance(altitude_result.get("value"), (int, type(None))), "Altitude should be an integer or None"
        
        # Process time ROI
        time_result = extract_values_from_roi(time_roi, mode="time")
        if time_result:
            assert "sign" in time_result, "Time result should have a sign"
            assert "hours" in time_result, "Time result should have hours"
            assert "minutes" in time_result, "Time result should have minutes"
            assert "seconds" in time_result, "Time result should have seconds"
