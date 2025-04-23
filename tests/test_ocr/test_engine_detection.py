import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock, ANY
from numpy.testing import assert_array_equal

from ocr.engine_detection import (
    check_engines_numba,
    check_engines,
    detect_engine_status
)

# Test frames should be placed in this directory
TEST_FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'engine_frames')

@pytest.fixture
def test_image():
    """Create a test image with some white and black pixels."""
    # Create a 10x10 black image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Set some pixels to white (255, 255, 255)
    # Engine ON pixels
    img[2, 3] = [255, 255, 255]  # y=2, x=3
    img[5, 7] = [255, 255, 255]  # y=5, x=7
    
    # Set some pixels to gray (partial on)
    img[4, 4] = [200, 200, 200]  # y=4, x=4
    
    # Set some pixels to dimmer values (engine off)
    img[6, 2] = [100, 100, 100]  # y=6, x=2
    
    return img

@pytest.fixture
def test_engine_coords():
    """Create test engine coordinates dictionary."""
    return {
        "inner": np.array([(3, 2), (7, 5)]),  # White pixels (ON) - using numpy arrays to avoid numba warnings
        "middle": np.array([(4, 4)]),         # Gray pixel (depends on threshold)
        "outer": np.array([(2, 6), (8, 8)])   # Black pixel and out of bounds (OFF)
    }

class TestCheckEnginesNumba:
    """Tests for check_engines_numba function."""
    
    def test_basic_functionality(self):
        """Test the numba-optimized engine check function."""
        # Create a simple test image
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[1, 2] = [255, 255, 255]  # White pixel at x=2, y=1
        image[3, 3] = [100, 100, 100]  # Dark pixel at x=3, y=3
        
        # Test coordinates - using numpy arrays to avoid numba warnings
        coordinates = np.array([(2, 1), (3, 3), (10, 10)])  # Last one is out of bounds
        
        # Test with high threshold (only very bright pixels are ON)
        high_threshold = 200
        result_high = check_engines_numba(image, coordinates, high_threshold)
        assert result_high == [True, False, False]
        
        # Test with low threshold (both bright and dark pixels are ON)
        low_threshold = 50
        result_low = check_engines_numba(image, coordinates, low_threshold)
        assert result_low == [True, True, False]
    
    def test_out_of_bounds(self):
        """Test behavior with out-of-bounds coordinates."""
        # Create a test image
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        
        # All coordinates are out of bounds
        coordinates = np.array([(-1, -1), (5, 5), (3, 3)])
        
        # Check result
        result = check_engines_numba(image, coordinates, 100)
        assert result == [False, False, False]


class TestCheckEngines:
    """Tests for check_engines function."""
    
    def test_basic_functionality(self, test_image, test_engine_coords):
        """Test the check_engines function."""
        with patch('ocr.engine_detection.logger') as mock_logger:
            # Test without debug
            result = check_engines(test_image, test_engine_coords, False, "Test")
            
            # Verify structure of result
            assert set(result.keys()) == set(test_engine_coords.keys())
            
            # Test with standard threshold (200)
            result_high = check_engines(test_image, test_engine_coords, True, "Test")
            
            # Inner engines should be ON
            assert all(result_high["inner"])
            # Middle engines depend on threshold
            # Outer engines should be OFF
            assert not any(result_high["outer"])
            
            # Verify debug logs were called when debug=True
            assert mock_logger.debug.called
    
    @patch('ocr.engine_detection.WHITE_THRESHOLD', 200)  # Patch the threshold value
    def test_debug_logging(self, test_image, test_engine_coords):
        """Test debug logging functionality."""
        with patch('ocr.engine_detection.logger') as mock_logger:
            # Call with debug=True
            check_engines(test_image, test_engine_coords, True, "TestEngine")
            
            # Verify specific log messages - now with patched WHITE_THRESHOLD
            mock_logger.debug.assert_any_call("Checking TestEngine engines with threshold: 200")
            mock_logger.debug.assert_any_call(
                f"Image shape: {test_image.shape}, checking {sum(len(coords) for coords in test_engine_coords.values())} engine points"
            )


class TestDetectEngineStatus:
    """Tests for detect_engine_status function."""
    
    @patch('ocr.engine_detection.SUPERHEAVY_ENGINES', {'test_section': np.array([(1, 1)])})
    @patch('ocr.engine_detection.STARSHIP_ENGINES', {'test_section': np.array([(2, 2)])})
    @patch('ocr.engine_detection.check_engines')
    def test_basic_functionality(self, mock_check_engines, test_image):
        """Test the detect_engine_status function."""
        # Configure the mock to return predefined values
        mock_check_engines.side_effect = [
            {"test_section": [True]},  # For Superheavy
            {"test_section": [False]}  # For Starship
        ]
        
        # Call the function with debug=True
        result = detect_engine_status(test_image, debug=True)
        
        # Verify the correct structure of the result
        assert set(result.keys()) == {"superheavy", "starship"}
        assert result["superheavy"] == {"test_section": [True]}
        assert result["starship"] == {"test_section": [False]}
        
        # Verify check_engines was called twice with correct parameters
        assert mock_check_engines.call_count == 2
        
        # Use ANY to avoid NumPy array comparison issues with mock assertions
        mock_check_engines.assert_any_call(test_image, ANY, True, "Superheavy")
        mock_check_engines.assert_any_call(test_image, ANY, True, "Starship")
        
        # Verify the numpy arrays manually
        calls = mock_check_engines.call_args_list
        superheavy_call = next(call for call in calls if call[0][3] == "Superheavy")
        starship_call = next(call for call in calls if call[0][3] == "Starship")
        
        # Check that the coordinate arrays match the expected values
        assert len(superheavy_call[0][1]) == 1  # One section
        assert_array_equal(
            superheavy_call[0][1]['test_section'], 
            np.array([(1, 1)])
        )
        
        assert len(starship_call[0][1]) == 1  # One section
        assert_array_equal(
            starship_call[0][1]['test_section'], 
            np.array([(2, 2)])
        )
    
    def test_integration(self, test_image):
        """Test the full integration of detect_engine_status with real dependencies."""
        with patch('ocr.engine_detection.SUPERHEAVY_ENGINES', {'inner': np.array([(3, 2)])}), \
             patch('ocr.engine_detection.STARSHIP_ENGINES', {'outer': np.array([(8, 8)])}), \
             patch('ocr.engine_detection.WHITE_THRESHOLD', 200):
            
            result = detect_engine_status(test_image, debug=False)
            
            # Verify expected results from our test image
            assert result["superheavy"] == {'inner': [True]}  # ON pixel
            assert result["starship"] == {'outer': [False]}   # Out of bounds pixel
    
    @patch('ocr.engine_detection.logger')
    def test_debug_output(self, mock_logger, test_image):
        """Test debug output for detect_engine_status."""
        with patch('ocr.engine_detection.SUPERHEAVY_ENGINES', {'inner': np.array([(3, 2)])}), \
             patch('ocr.engine_detection.STARSHIP_ENGINES', {'outer': np.array([(8, 8)])}):
            
            detect_engine_status(test_image, debug=True)
            
            # Verify debug log was called
            mock_logger.debug.assert_any_call(f"Starting engine detection on image of shape {test_image.shape}")


@pytest.mark.skipif(not os.path.exists(TEST_FRAMES_DIR), reason="Test frames directory not found")
class TestWithRealFrames:
    """Tests using real frame captures from SpaceX videos."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Check if test frames directory exists and contains required files."""
        if not os.path.exists(TEST_FRAMES_DIR):
            os.makedirs(TEST_FRAMES_DIR, exist_ok=True)
            pytest.skip("Test frames directory is empty")
    
    def _load_test_frame(self, filename):
        """Helper to load a test frame and convert to numpy array."""
        from PIL import Image
        frame_path = os.path.join(TEST_FRAMES_DIR, filename)
        if not os.path.exists(frame_path):
            pytest.skip(f"Test frame {filename} not found")
        return np.array(Image.open(frame_path))
    
    def _count_active_engines(self, result):
        """Helper to count active engines in each vehicle."""
        sh_active = sum(sum(status) for status in result["superheavy"].values())
        sh_total = sum(len(status) for status in result["superheavy"].values())
        ss_active = sum(sum(status) for status in result["starship"].values())
        ss_total = sum(len(status) for status in result["starship"].values())
        return sh_active, sh_total, ss_active, ss_total
            
    def test_engines_off(self):
        """Test engine detection when all engines are off (pre-ignition)."""
        image = self._load_test_frame('engines_off.png')
        result = detect_engine_status(image)
        
        # Count active engines
        sh_active, sh_total, ss_active, ss_total = self._count_active_engines(result)
        
        # Generate detailed failure messages
        assert sh_active == 0, f"Expected 0 active Superheavy engines, but found {sh_active}/{sh_total}"
        assert ss_active == 0, f"Expected 0 active Starship engines, but found {ss_active}/{ss_total}"
    
    def test_superheavy_ignition(self):
        """Test Superheavy engine ignition sequence."""
        image = self._load_test_frame('superheavy_ignition.png')
        result = detect_engine_status(image)
        
        # Count active engines
        sh_active, sh_total, ss_active, ss_total = self._count_active_engines(result)
        
        # During ignition, some but not all SuperHeavy engines should be active
        assert 0 < sh_active < sh_total, f"Superheavy ignition: Expected some active engines, found {sh_active}/{sh_total}"
        assert ss_active == 0, f"Expected 0 active Starship engines, but found {ss_active}/{ss_total}"
    
    def test_starship_ignition(self):
        """Test Starship engine ignition during stage separation."""
        image = self._load_test_frame('starship_ignition.png')
        result = detect_engine_status(image)
        
        # Count active engines
        sh_active, sh_total, ss_active, ss_total = self._count_active_engines(result)
        
        # Both Superheavy and Starship engines should be on
        assert sh_active > 0, f"Expected some active Superheavy engines, found {sh_active}/{sh_total}"
        assert ss_active > 0, f"Expected some active Starship engines, found {ss_active}/{ss_total}"
    
    def test_full_thrust(self):
        """Test full thrust mode with all Superheavy engines firing."""
        # Skip if the file doesn't exist
        if not os.path.exists(os.path.join(TEST_FRAMES_DIR, 'full_thrust.png')):
            pytest.skip("Full thrust test frame not available")
        
        image = self._load_test_frame('full_thrust.png')
        result = detect_engine_status(image)
        
        # Count active engines
        sh_active, sh_total, ss_active, ss_total = self._count_active_engines(result)
        
        # Almost all Superheavy engines should be active (allowing for minor detection errors)
        assert sh_active >= 0.9 * sh_total, f"Full thrust: Expected most engines active, found {sh_active}/{sh_total}"
    
    def test_landing_burn(self):
        """Test Superheavy landing burn with only center engines firing."""
        # Skip if the file doesn't exist
        if not os.path.exists(os.path.join(TEST_FRAMES_DIR, 'landing_burn.png')):
            pytest.skip("Landing burn test frame not available")
        
        image = self._load_test_frame('landing_burn.png')
        result = detect_engine_status(image)
        
        # Count active engines
        sh_active, sh_total, ss_active, ss_total = self._count_active_engines(result)
        
        # Only some Superheavy engines should be active during landing
        assert 0 < sh_active < sh_total * 0.5, f"Landing burn: Expected few engines active, found {sh_active}/{sh_total}"
