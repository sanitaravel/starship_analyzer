import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import os

from ocr.fuel_level_extraction import (
    extract_fuel_levels,
    process_strip,
    STRIP_COORDS,
    REF_PIXEL_COORDS,
    STRIP_LENGTH,
    STRIP_HEIGHT,
    BRIGHTNESS_THRESHOLD,
    REF_DIFF_THRESHOLD
)

@pytest.fixture
def synthetic_image():
    """Create a synthetic image for testing."""
    # Create a black image with the right dimensions
    height, width = 1080, 1920
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add fuel level bars at the specified coordinates
    # Superheavy LOX at 75% fullness
    bar_length = int(STRIP_LENGTH * 0.75)
    x, y = STRIP_COORDS[0]
    image[y:y+STRIP_HEIGHT, x:x+bar_length] = 255
    
    # Superheavy CH4 at 60% fullness
    bar_length = int(STRIP_LENGTH * 0.6)
    x, y = STRIP_COORDS[1]
    image[y:y+STRIP_HEIGHT, x:x+bar_length] = 255
    
    # Starship LOX at 80% fullness
    bar_length = int(STRIP_LENGTH * 0.8)
    x, y = STRIP_COORDS[2]
    image[y:y+STRIP_HEIGHT, x:x+bar_length] = 255
    
    # Starship CH4 at 70% fullness
    bar_length = int(STRIP_LENGTH * 0.7)
    x, y = STRIP_COORDS[3]
    image[y:y+STRIP_HEIGHT, x:x+bar_length] = 255
    
    # Add reference pixels for all strips
    for i, (ref_x, ref_y) in enumerate(REF_PIXEL_COORDS):
        # Make sure the reference pixels have different values to pass the difference check
        image[ref_y, ref_x] = 100     # First reference pixel
        ref_x2 = ref_x + 5 if i in [0, 2] else ref_x - 5
        image[ref_y, ref_x2] = 200    # Second reference pixel
    
    return image

@pytest.fixture
def empty_image():
    """Create an empty image with no fuel bars."""
    # Create a black image with the right dimensions
    height, width = 1080, 1920
    image = np.zeros((height, width), dtype=np.uint8)
    return image

class TestProcessStrip:
    """Tests for process_strip function."""
    
    def test_valid_strip(self, synthetic_image):
        """Test processing a valid strip."""
        # Process Superheavy LOX strip (index 0)
        result = process_strip(synthetic_image, 0, debug=True)
        
        # Check result contains expected keys
        assert "fullness" in result
        assert "length" in result
        assert "ref_diff" in result
        
        # Verify fullness is close to expected value (75%)
        assert 70 <= result["fullness"] <= 80, f"Expected fullness around 75%, got {result['fullness']}%"
        
        # Verify length is close to expected value (75% of STRIP_LENGTH)
        expected_length = int(STRIP_LENGTH * 0.75)
        assert abs(result["length"] - expected_length) <= 5, f"Expected length around {expected_length}, got {result['length']}"
    
    def test_invalid_strip_index(self, synthetic_image):
        """Test handling of invalid strip index."""
        with patch('ocr.fuel_level_extraction.logger') as mock_logger:
            # Test with invalid index
            result = process_strip(synthetic_image, 10, debug=True)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            
            # Verify empty result
            assert result["fullness"] == 0.0
            assert result["length"] == 0
    
    def test_empty_strip(self, empty_image):
        """Test processing a strip with no fuel bar."""
        # Process any strip from the empty image
        result = process_strip(empty_image, 0, debug=True)
        
        # Fullness should be 0 when no bright pixels are found
        assert result["fullness"] == 0.0
        assert result["length"] == 0
    
    def test_all_strips(self, synthetic_image):
        """Test processing all strips."""
        # Expected fullness values for each strip
        expected_fullness = [75, 60, 80, 70]
        
        for i, expected in enumerate(expected_fullness):
            result = process_strip(synthetic_image, i, debug=True)
            
            # Allow for some tolerance in the detection
            assert abs(result["fullness"] - expected) <= 5, f"Strip {i}: Expected fullness around {expected}%, got {result['fullness']}%"


class TestExtractFuelLevels:
    """Tests for extract_fuel_levels function."""
    
    def test_basic_functionality(self, synthetic_image):
        """Test basic functionality of extract_fuel_levels."""
        # Call the function
        result = extract_fuel_levels(synthetic_image, debug=True)
        
        # Verify structure of result
        assert "superheavy" in result
        assert "starship" in result
        assert "lox" in result["superheavy"]
        assert "ch4" in result["superheavy"]
        assert "lox" in result["starship"]
        assert "ch4" in result["starship"]
        
        # Verify fullness values are close to expected
        assert 70 <= result["superheavy"]["lox"]["fullness"] <= 80
        assert 55 <= result["superheavy"]["ch4"]["fullness"] <= 65
        assert 75 <= result["starship"]["lox"]["fullness"] <= 85
        assert 65 <= result["starship"]["ch4"]["fullness"] <= 75
    
    def test_color_image(self):
        """Test extract_fuel_levels with a color image."""
        # Create a color image (3 channels)
        height, width = 1080, 1920
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a test fuel bar
        x, y = STRIP_COORDS[0]
        bar_length = int(STRIP_LENGTH * 0.5)
        color_image[y:y+STRIP_HEIGHT, x:x+bar_length] = [255, 255, 255]
        
        # Add reference pixels
        ref_x, ref_y = REF_PIXEL_COORDS[0]
        color_image[ref_y, ref_x] = [100, 100, 100]
        color_image[ref_y, ref_x+5] = [200, 200, 200]
        
        # Test with mocked process_strip
        with patch('ocr.fuel_level_extraction.process_strip') as mock_process_strip:
            # Configure mock to return predefined values
            mock_process_strip.side_effect = [
                {"fullness": 50, "length": bar_length},
                {"fullness": 40, "length": bar_length * 0.8},
                {"fullness": 60, "length": bar_length * 1.2},
                {"fullness": 45, "length": bar_length * 0.9}
            ]
            
            # Call the function
            result = extract_fuel_levels(color_image, debug=True)
            
            # Verify convert color to grayscale was called implicitly
            assert mock_process_strip.call_count == 4
            
            # Verify results use the mocked values
            assert result["superheavy"]["lox"]["fullness"] == 50
            assert result["superheavy"]["ch4"]["fullness"] == 40
            assert result["starship"]["lox"]["fullness"] == 60
            assert result["starship"]["ch4"]["fullness"] == 45
    
    def test_error_handling(self):
        """Test error handling in extract_fuel_levels."""
        # Create an invalid image (None)
        with patch('ocr.fuel_level_extraction.logger') as mock_logger, \
             patch('ocr.fuel_level_extraction.cv2.cvtColor') as mock_cvtColor:
            
            # Make cv2.cvtColor raise an exception
            mock_cvtColor.side_effect = Exception("Test error")
            
            # Call the function
            result = extract_fuel_levels(np.zeros((10, 10, 3)), debug=True)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            
            # Verify default values are returned
            assert result["superheavy"]["lox"]["fullness"] == 0
            assert result["superheavy"]["ch4"]["fullness"] == 0
            assert result["starship"]["lox"]["fullness"] == 0
            assert result["starship"]["ch4"]["fullness"] == 0


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'fuel_samples')), reason="Fuel sample images not found")
class TestWithRealImages:
    """Integration tests with real images."""
    
    def test_real_image(self):
        """Test fuel level extraction with a real flight image."""
        # Path to test image
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'fuel_samples', 'fuel_levels.png')
        
        if not os.path.exists(image_path):
            pytest.skip(f"Test image {image_path} not found")
        
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"Failed to load image at {image_path}"
        
        # Extract fuel levels
        result = extract_fuel_levels(image, debug=True)
        
        # Verify extraction produced some values (not testing exact values as they depend on the image)
        assert isinstance(result["superheavy"]["lox"]["fullness"], (int, float))
        assert isinstance(result["superheavy"]["ch4"]["fullness"], (int, float))
        assert isinstance(result["starship"]["lox"]["fullness"], (int, float))
        assert isinstance(result["starship"]["ch4"]["fullness"], (int, float))
