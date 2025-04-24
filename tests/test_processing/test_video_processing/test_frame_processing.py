import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from processing.video_processing.frame_processing import (
    process_frame,
    process_single_frame,
    process_video_frame
)

@pytest.fixture
def test_frame():
    """Create a test frame for processing."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def mock_extract_data():
    """Mock the extract_data function."""
    with patch('processing.video_processing.frame_processing.extract_data') as mock:
        # Configure mock to return sample data
        mock.return_value = (
            {"speed": 100, "altitude": 5000, "engines": {"inner": [True, False], "outer": [True]}},
            {"speed": 200, "altitude": 10000, "engines": {"raptor": [True, True, False]}},
            {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
        )
        yield mock

class TestProcessFrame:
    
    def test_normal_processing(self, test_frame, mock_extract_data):
        """Test normal frame processing."""
        # Call the function
        result = process_frame(1000, test_frame, False, False, False)
        
        # Verify extract_data was called with correct parameters
        mock_extract_data.assert_called_once_with(test_frame, display_rois=False, debug=False, zero_time_met=False)
        
        # Verify result structure
        assert result["frame_number"] == 1000
        assert "superheavy" in result
        assert "starship" in result
        assert "time" in result
        assert result["superheavy"]["speed"] == 100
        assert result["starship"]["altitude"] == 10000
    
    def test_error_handling(self, test_frame):
        """Test error handling during processing."""
        with patch('processing.video_processing.frame_processing.extract_data',
                  side_effect=Exception("Test error")):
            result = process_frame(1000, test_frame, False, False, False)
            
            # Verify error was handled properly
            assert "error" in result
            assert result["frame_number"] == 1000
            assert result["superheavy"] == {}
            assert result["starship"] == {}
            assert result["time"] is None

class TestProcessSingleFrame:
    
    def test_normal_processing(self, test_frame, mock_extract_data):
        """Test normal frame processing in the alternative path."""
        # Call the function
        result = process_single_frame(1000, test_frame, False, False, False)
        
        # Verify extract_data was called with correct parameters
        mock_extract_data.assert_called_once_with(test_frame, display_rois=False, debug=False)
        
        # Verify result structure
        assert result["frame_number"] == 1000
        assert "superheavy" in result
        assert "starship" in result
        assert "time" in result
        assert result["superheavy"]["speed"] == 100
        assert result["starship"]["altitude"] == 10000
    
    def test_show_progress(self, test_frame, mock_extract_data):
        """Test with progress display enabled."""
        with patch('processing.video_processing.frame_processing.logger') as mock_logger:
            process_single_frame(1000, test_frame, False, False, True)
            
            # Verify debug message was logged
            mock_logger.debug.assert_called_once_with("Processed frame 1000")
    
    def test_error_handling(self, test_frame):
        """Test error handling during processing."""
        with patch('processing.video_processing.frame_processing.extract_data',
                  side_effect=Exception("Test error")):
            result = process_single_frame(1000, test_frame, False, False, False)
            
            # Verify error was handled properly
            assert "error" in result
            assert result["frame_number"] == 1000
            assert result["superheavy"] == {}
            assert result["starship"] == {}
            assert result["time"] is None

class TestProcessVideoFrame:
    
    @patch('processing.video_processing.frame_processing.cv2.VideoCapture')
    @patch('processing.video_processing.frame_processing.random.randint', return_value=500)
    def test_normal_processing(self, mock_randint, mock_video_capture, test_frame, mock_extract_data):
        """Test processing a random frame from a video."""
        # Configure mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 1000}[prop]
        mock_cap.read.return_value = (True, test_frame)
        
        # Call the function
        result = process_video_frame("test_video.mp4", False, False, 10, 20)
        
        # Verify the video was opened
        mock_video_capture.assert_called_once_with("test_video.mp4")
        
        # Verify random frame selection
        mock_randint.assert_called_once_with(300, 600)  # 10 seconds * 30 fps = 300, 20 seconds * 30 fps = 600
        
        # Verify frame was read
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 500)
        mock_cap.read.assert_called_once()
        
        # Verify result
        assert "frame_number" in result
        assert "superheavy" in result
        assert "starship" in result
        assert "time" in result
    
    @patch('processing.video_processing.frame_processing.cv2.VideoCapture')
    def test_video_open_failure(self, mock_video_capture):
        """Test handling of video open failure."""
        # Configure mock to return failure
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Call the function
        result = process_video_frame("nonexistent.mp4", False, False, 10, 20)
        
        # Verify error was handled
        assert "error" in result
        assert result["error"] == "Failed to open video file"
    
    @patch('processing.video_processing.frame_processing.cv2.VideoCapture')
    def test_invalid_time_range(self, mock_video_capture):
        """Test handling of invalid time range."""
        # Configure mock
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 1000}[prop]
        
        # Call with start time > end time
        result = process_video_frame("test_video.mp4", False, False, 20, 10)
        
        # Verify error was handled
        assert "error" in result
        assert "Invalid time range" in result["error"]
    
    @patch('processing.video_processing.frame_processing.cv2.VideoCapture')
    @patch('processing.video_processing.frame_processing.random.randint', return_value=500)
    def test_frame_read_failure(self, mock_randint, mock_video_capture):
        """Test handling of frame read failure."""
        # Configure mock
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 1000}[prop]
        mock_cap.read.return_value = (False, None)  # Read failure
        
        # Call the function
        result = process_video_frame("test_video.mp4", False, False, 10, 20)
        
        # Verify error was handled
        assert "error" in result
        assert "Failed to read frame" in result["error"]
