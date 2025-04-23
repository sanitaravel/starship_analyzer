import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ocr.extract_data import (
    preprocess_image,
    extract_superheavy_data,
    extract_starship_data,
    extract_time_data,
    extract_data
)

@pytest.fixture
def test_image():
    """Create a test image with appropriate dimensions."""
    # Create an image large enough for ROI extraction
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return img

@pytest.fixture
def test_rois():
    """Create test ROIs for vehicle data."""
    sh_speed_roi = np.zeros((25, 83, 3), dtype=np.uint8)
    sh_altitude_roi = np.zeros((25, 50, 3), dtype=np.uint8)
    ss_speed_roi = np.zeros((25, 83, 3), dtype=np.uint8)
    ss_altitude_roi = np.zeros((25, 50, 3), dtype=np.uint8)
    time_roi = np.zeros((44, 197, 3), dtype=np.uint8)
    return sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi

@pytest.fixture
def mock_extract_values():
    """Mock for extract_values_from_roi with default returns."""
    with patch('ocr.extract_data.extract_values_from_roi') as mock:
        # Configure default returns for different modes
        def side_effect(roi, mode=None, **kwargs):
            if mode == "speed":
                return {"value": 100}
            elif mode == "altitude":
                return {"value": 5000}
            elif mode == "time":
                return {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
            return {}
            
        mock.side_effect = side_effect
        yield mock

@pytest.fixture
def mock_detect_engine_status():
    """Mock for engine detection."""
    with patch('ocr.extract_data.detect_engine_status') as mock:
        mock.return_value = {
            "superheavy": {"inner": [True, False, True], "outer": [False, True]},
            "starship": {"raptor": [True, True, False]}
        }
        yield mock

@pytest.fixture
def mock_extract_fuel_levels():
    """Mock for fuel level extraction."""
    with patch('ocr.extract_data.extract_fuel_levels') as mock:
        mock.return_value = {
            "superheavy": {
                "lox": {"fullness": 85.5},
                "ch4": {"fullness": 90.2}
            },
            "starship": {
                "lox": {"fullness": 75.0},
                "ch4": {"fullness": 80.3}
            }
        }
        yield mock


class TestPreprocessImage:
    """Tests for preprocess_image function."""
    
    def test_basic_functionality(self, test_image):
        """Test basic functionality of preprocess_image."""
        # Call the function
        sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(test_image)
        
        # Verify ROI shapes
        assert sh_speed_roi.shape == (25, 83, 3)
        assert sh_altitude_roi.shape == (25, 50, 3)
        assert ss_speed_roi.shape == (25, 83, 3)
        assert ss_altitude_roi.shape == (25, 50, 3)
        assert time_roi.shape == (44, 197, 3)
    
    def test_display_rois_option(self, test_image):
        """Test the display_rois option."""
        with patch('ocr.extract_data.display_image') as mock_display:
            # Call with display_rois=True
            preprocess_image(test_image, display_rois=True)
            
            # Verify display_image was called 5 times (for each ROI)
            assert mock_display.call_count == 5
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with None image
        with patch('ocr.extract_data.logger') as mock_logger:
            sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(None)
            
            # Verify error was logged
            mock_logger.error.assert_called_once_with("Input image is None")
            
            # Verify empty ROIs are returned
            assert sh_speed_roi.shape == (1, 1, 3)
        
        # Test with invalid image dimensions
        with patch('ocr.extract_data.logger') as mock_logger:
            small_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Too small for ROIs
            sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi = preprocess_image(small_image)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            
            # Verify that all returned ROIs are empty (1x1x3) as defined in the function
            for roi in [sh_speed_roi, sh_altitude_roi, ss_speed_roi, ss_altitude_roi, time_roi]:
                assert roi.shape == (1, 1, 3), f"Expected empty ROI with shape (1, 1, 3), got {roi.shape}"


class TestExtractSuperheavyData:
    """Tests for extract_superheavy_data function."""
    
    def test_basic_functionality(self, test_rois, mock_extract_values):
        """Test basic functionality of extract_superheavy_data."""
        sh_speed_roi, sh_altitude_roi, _, _, _ = test_rois
        
        # Call the function
        result = extract_superheavy_data(sh_speed_roi, sh_altitude_roi, display_rois=False, debug=False)
        
        # Verify results
        assert result["speed"] == 100
        assert result["altitude"] == 5000
        
        # Verify extract_values_from_roi was called with correct parameters
        calls = mock_extract_values.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] is sh_speed_roi
        assert calls[0][1]["mode"] == "speed"
        assert calls[1][0][0] is sh_altitude_roi
        assert calls[1][1]["mode"] == "altitude"
    
    def test_debug_output(self, test_rois, mock_extract_values):
        """Test debug output."""
        sh_speed_roi, sh_altitude_roi, _, _, _ = test_rois
        
        with patch('ocr.extract_data.logger') as mock_logger:
            # Call with debug=True
            extract_superheavy_data(sh_speed_roi, sh_altitude_roi, display_rois=False, debug=True)
            
            # Verify debug logs were called
            mock_logger.debug.assert_any_call("Extracting Superheavy data from ROIs")
            mock_logger.debug.assert_any_call("Extracted Superheavy speed: 100, altitude: 5000")
    
    def test_error_handling(self, test_rois):
        """Test error handling."""
        sh_speed_roi, sh_altitude_roi, _, _, _ = test_rois
        
        with patch('ocr.extract_data.extract_values_from_roi') as mock:
            # Mock to raise an exception
            mock.side_effect = Exception("Test error")
            
            with patch('ocr.extract_data.logger') as mock_logger:
                # Call the function
                result = extract_superheavy_data(sh_speed_roi, sh_altitude_roi, display_rois=False, debug=True)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                
                # Verify default values are returned
                assert result == {"speed": None, "altitude": None}


class TestExtractStarshipData:
    """Tests for extract_starship_data function."""
    
    def test_basic_functionality(self, test_rois, mock_extract_values):
        """Test basic functionality of extract_starship_data."""
        _, _, ss_speed_roi, ss_altitude_roi, _ = test_rois
        
        # Call the function
        result = extract_starship_data(ss_speed_roi, ss_altitude_roi, display_rois=False, debug=False)
        
        # Verify results
        assert result["speed"] == 100
        assert result["altitude"] == 5000
        
        # Verify extract_values_from_roi was called with correct parameters
        calls = mock_extract_values.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] is ss_speed_roi
        assert calls[0][1]["mode"] == "speed"
        assert calls[1][0][0] is ss_altitude_roi
        assert calls[1][1]["mode"] == "altitude"
    
    def test_debug_output(self, test_rois, mock_extract_values):
        """Test debug output."""
        _, _, ss_speed_roi, ss_altitude_roi, _ = test_rois
        
        with patch('ocr.extract_data.logger') as mock_logger:
            # Call with debug=True
            extract_starship_data(ss_speed_roi, ss_altitude_roi, display_rois=False, debug=True)
            
            # Verify debug logs were called
            mock_logger.debug.assert_any_call("Extracting Starship data from ROIs")
            mock_logger.debug.assert_any_call("Extracted Starship speed: 100, altitude: 5000")
    
    def test_error_handling(self, test_rois):
        """Test error handling."""
        _, _, ss_speed_roi, ss_altitude_roi, _ = test_rois
        
        with patch('ocr.extract_data.extract_values_from_roi') as mock:
            # Mock to raise an exception
            mock.side_effect = Exception("Test error")
            
            with patch('ocr.extract_data.logger') as mock_logger:
                # Call the function
                result = extract_starship_data(ss_speed_roi, ss_altitude_roi, display_rois=False, debug=True)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                
                # Verify default values are returned
                assert result == {"speed": None, "altitude": None}


class TestExtractTimeData:
    """Tests for extract_time_data function."""
    
    def test_basic_functionality(self, test_rois, mock_extract_values):
        """Test basic functionality of extract_time_data."""
        _, _, _, _, time_roi = test_rois
        
        # Call the function
        result = extract_time_data(time_roi, display_rois=False, debug=False, zero_time_met=False)
        
        # Verify results
        assert result["sign"] == "+"
        assert result["hours"] == 0
        assert result["minutes"] == 1
        assert result["seconds"] == 30
        
        # Verify extract_values_from_roi was called with correct parameters
        mock_extract_values.assert_called_once_with(time_roi, mode="time", display_transformed=False, debug=False)
    
    def test_zero_time_already_met(self, test_rois, mock_extract_values):
        """Test when zero_time_met is True."""
        _, _, _, _, time_roi = test_rois
        
        # Call the function with zero_time_met=True
        result = extract_time_data(time_roi, display_rois=False, debug=False, zero_time_met=True)
        
        # Verify default zero time is returned
        assert result["sign"] == "+"
        assert result["hours"] == 0
        assert result["minutes"] == 0
        assert result["seconds"] == 0
        
        # Verify extract_values_from_roi was not called
        mock_extract_values.assert_not_called()
    
    def test_debug_output(self, test_rois, mock_extract_values):
        """Test debug output."""
        _, _, _, _, time_roi = test_rois
        
        with patch('ocr.extract_data.logger') as mock_logger:
            # Call with debug=True
            extract_time_data(time_roi, display_rois=False, debug=True, zero_time_met=False)
            
            # Verify debug logs were called
            mock_logger.debug.assert_any_call("Extracting time data from ROI")
            mock_logger.debug.assert_any_call(f"Extracted time: + 00:01:30")
    
    def test_error_handling(self, test_rois):
        """Test error handling."""
        _, _, _, _, time_roi = test_rois
        
        with patch('ocr.extract_data.extract_values_from_roi') as mock:
            # Mock to raise an exception
            mock.side_effect = Exception("Test error")
            
            with patch('ocr.extract_data.logger') as mock_logger:
                # Call the function
                result = extract_time_data(time_roi, display_rois=False, debug=True, zero_time_met=False)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                
                # Verify empty dict is returned
                assert result == {}


class TestExtractData:
    """Tests for the main extract_data function."""
    
    @patch('ocr.extract_data.preprocess_image')
    def test_basic_functionality(self, mock_preprocess, test_rois, test_image, 
                                 mock_extract_values, mock_detect_engine_status, 
                                 mock_extract_fuel_levels):
        """Test basic functionality of extract_data."""
        # Setup preprocess_image to return test ROIs
        mock_preprocess.return_value = test_rois
        
        # Call the function
        superheavy_data, starship_data, time_data = extract_data(test_image)
        
        # Verify vehicle data
        assert superheavy_data["speed"] == 100
        assert superheavy_data["altitude"] == 5000
        assert starship_data["speed"] == 100
        assert starship_data["altitude"] == 5000
        
        # Verify time data
        assert time_data["sign"] == "+"
        assert time_data["hours"] == 0
        assert time_data["minutes"] == 1
        assert time_data["seconds"] == 30
        
        # Verify fuel data was added
        assert superheavy_data["fuel"]["lox"]["fullness"] == 85.5
        assert superheavy_data["fuel"]["ch4"]["fullness"] == 90.2
        assert starship_data["fuel"]["lox"]["fullness"] == 75.0
        assert starship_data["fuel"]["ch4"]["fullness"] == 80.3
        
        # Verify engine data was added
        assert superheavy_data["engines"]["inner"] == [True, False, True]
        assert superheavy_data["engines"]["outer"] == [False, True]
        assert starship_data["engines"]["raptor"] == [True, True, False]
    
    @patch('ocr.extract_data.preprocess_image')
    def test_starship_data_fallback(self, mock_preprocess, test_image):
        """Test fallback to Superheavy data when Starship data is missing."""
        # Create test ROIs
        test_rois = tuple(np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5))
        mock_preprocess.return_value = test_rois
        
        # Mock extract_values_from_roi to return values for Superheavy but None for Starship
        with patch('ocr.extract_data.extract_values_from_roi') as mock:
            def mock_extract(roi, mode=None, **kwargs):
                # Return values for Superheavy, None for Starship speed
                if mode == "speed":
                    if roi is test_rois[0]:  # Superheavy speed
                        return {"value": 100}
                    else:  # Starship speed
                        return {"value": None}
                elif mode == "altitude":
                    if roi is test_rois[1]:  # Superheavy altitude
                        return {"value": 5000}
                    else:  # Starship altitude
                        return {"value": None}
                elif mode == "time":
                    return {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
                return {}
                
            mock.side_effect = mock_extract
            
            # Mock other dependencies
            with patch('ocr.extract_data.detect_engine_status') as mock_engines:
                mock_engines.return_value = {
                    "superheavy": {"inner": [True]},
                    "starship": {"raptor": [False]}
                }
                
                with patch('ocr.extract_data.extract_fuel_levels') as mock_fuel:
                    mock_fuel.return_value = {
                        "superheavy": {"lox": {"fullness": 85.5}, "ch4": {"fullness": 90.2}},
                        "starship": {"lox": {"fullness": 75.0}, "ch4": {"fullness": 80.3}}
                    }
                    
                    # Call the function with debug for coverage
                    superheavy_data, starship_data, _ = extract_data(test_image, debug=True)
                    
                    # Verify Starship data was populated with Superheavy values
                    assert starship_data["speed"] == 100
                    assert starship_data["altitude"] == 5000
    
    @patch('ocr.extract_data.preprocess_image')
    def test_fuel_extraction_error(self, mock_preprocess, test_image, test_rois):
        """Test handling of errors during fuel extraction."""
        mock_preprocess.return_value = test_rois
        
        # Setup normal value extraction
        with patch('ocr.extract_data.extract_values_from_roi') as mock_extract:
            def mock_extract_values(roi, mode=None, **kwargs):
                if mode == "speed":
                    return {"value": 100}
                elif mode == "altitude":
                    return {"value": 5000}
                elif mode == "time":
                    return {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
                return {}
                
            mock_extract.side_effect = mock_extract_values
            
            # Mock fuel extraction to raise an exception
            with patch('ocr.extract_data.extract_fuel_levels') as mock_fuel:
                mock_fuel.side_effect = Exception("Fuel extraction error")
                
                # Mock engine detection to succeed
                with patch('ocr.extract_data.detect_engine_status') as mock_engines:
                    mock_engines.return_value = {
                        "superheavy": {"inner": [True]},
                        "starship": {"raptor": [False]}
                    }
                    
                    with patch('ocr.extract_data.logger') as mock_logger:
                        # Call the function
                        superheavy_data, starship_data, _ = extract_data(test_image)
                        
                        # Verify error was logged
                        mock_logger.error.assert_any_call("Error extracting fuel levels: Fuel extraction error")
                        
                        # Verify default fuel values were set
                        assert superheavy_data["fuel"]["lox"]["fullness"] == 0
                        assert superheavy_data["fuel"]["ch4"]["fullness"] == 0
                        assert starship_data["fuel"]["lox"]["fullness"] == 0
                        assert starship_data["fuel"]["ch4"]["fullness"] == 0
    
    @patch('ocr.extract_data.preprocess_image')
    def test_engine_detection_error(self, mock_preprocess, test_image, test_rois):
        """Test handling of errors during engine detection."""
        mock_preprocess.return_value = test_rois
        
        # Setup normal value and fuel extraction
        with patch('ocr.extract_data.extract_values_from_roi') as mock_extract:
            def mock_extract_values(roi, mode=None, **kwargs):
                if mode == "speed":
                    return {"value": 100}
                elif mode == "altitude":
                    return {"value": 5000}
                elif mode == "time":
                    return {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
                return {}
                
            mock_extract.side_effect = mock_extract_values
            
            with patch('ocr.extract_data.extract_fuel_levels') as mock_fuel:
                mock_fuel.return_value = {
                    "superheavy": {"lox": {"fullness": 85.5}, "ch4": {"fullness": 90.2}},
                    "starship": {"lox": {"fullness": 75.0}, "ch4": {"fullness": 80.3}}
                }
                
                # Mock engine detection to raise an exception
                with patch('ocr.extract_data.detect_engine_status') as mock_engines:
                    mock_engines.side_effect = Exception("Engine detection error")
                    
                    with patch('ocr.extract_data.logger') as mock_logger:
                        # Call the function
                        superheavy_data, starship_data, _ = extract_data(test_image)
                        
                        # Verify error was logged
                        mock_logger.error.assert_any_call("Error detecting engine status: Engine detection error")
                        
                        # Verify empty engine dicts were set
                        assert superheavy_data["engines"] == {}
                        assert starship_data["engines"] == {}
