import pytest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from processing.frame_processing import (
    process_image,
    process_video_frame,
    process_frame
)

@pytest.fixture
def mock_cv2_imread():
    with patch('processing.frame_processing.cv2.imread') as mock:
        # Create a test image
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock.return_value = test_image
        yield mock

@pytest.fixture
def mock_extract_data():
    with patch('processing.frame_processing.extract_data') as mock:
        # Configure mock to return sample data
        mock.return_value = (
            {"speed": 100, "altitude": 5000, "engines": {"inner": [True, False], "outer": [True]}},
            {"speed": 200, "altitude": 10000, "engines": {"raptor": [True, True, False]}},
            {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
        )
        yield mock

@pytest.fixture
def mock_video_capture():
    with patch('processing.frame_processing.cv2.VideoCapture') as mock_cap_class:
        # Create a mock video capture instance
        mock_cap = MagicMock()
        mock_cap_class.return_value = mock_cap
        
        # Configure the mock cap object
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        
        # Configure read to return a test frame
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        yield mock_cap_class

# Tests for process_image
class TestProcessImage:
    
    def test_basic_functionality(self, mock_cv2_imread, mock_extract_data):
        """Test the basic functionality of process_image."""
        # Call the function
        process_image("test_image.jpg", display_rois=False, debug=True)
        
        # Verify imread was called with the correct path
        mock_cv2_imread.assert_called_once_with("test_image.jpg")
        
        # Verify extract_data was called
        mock_extract_data.assert_called_once()
    
    def test_image_load_failure(self, mock_cv2_imread, mock_extract_data):
        """Test handling when image loading fails."""
        # Set up imread to return None (failure to load)
        mock_cv2_imread.return_value = None
        
        # Call the function
        process_image("nonexistent.jpg", display_rois=False, debug=True)
        
        # Verify imread was called
        mock_cv2_imread.assert_called_once_with("nonexistent.jpg")
        
        # Verify extract_data was not called
        mock_extract_data.assert_not_called()
    
    def test_exception_handling(self, mock_cv2_imread, mock_extract_data):
        """Test exception handling in process_image."""
        # Set up extract_data to raise an exception
        mock_extract_data.side_effect = Exception("Test exception")
        
        # Call the function - should not raise an exception
        process_image("test_image.jpg", display_rois=False, debug=True)
        
        # Verify imread was called
        mock_cv2_imread.assert_called_once_with("test_image.jpg")
        
        # Verify extract_data was called but exception was caught
        mock_extract_data.assert_called_once()

# Tests for process_video_frame
class TestProcessVideoFrame:
    
    def test_basic_functionality(self, mock_video_capture, mock_extract_data):
        """Test the basic functionality of process_video_frame."""
        with patch('processing.frame_processing.process_image') as mock_process_image, \
             patch('processing.frame_processing.cv2.imwrite') as mock_imwrite, \
             patch('processing.frame_processing.random.randint', return_value=500):
            
            # Call the function
            process_video_frame("test_video.mp4", display_rois=False, debug=True,
                               start_time=10, end_time=20)
            
            # Verify VideoCapture was called with the correct path
            mock_video_capture.assert_called_once_with("test_video.mp4")
            
            # Verify a frame was saved and process_image was called
            assert mock_imwrite.called
            mock_process_image.assert_called_once()
    
    def test_video_load_failure(self, mock_video_capture):
        """Test handling when video loading fails."""
        # Set up isOpened to return False (failure to load)
        mock_video_capture.return_value.isOpened.return_value = False
        
        # Call the function
        process_video_frame("nonexistent.mp4", display_rois=False, debug=True,
                           start_time=10, end_time=20)
        
        # Verify VideoCapture was called
        mock_video_capture.assert_called_once_with("nonexistent.mp4")
    
    def test_invalid_time_range(self, mock_video_capture):
        """Test handling of invalid time range."""
        # Call with end before start
        process_video_frame("test_video.mp4", display_rois=False, debug=True,
                           start_time=30, end_time=10)
        
        # Verify video was opened and then closed
        mock_video_capture.assert_called_once_with("test_video.mp4")
        mock_video_capture.return_value.release.assert_called_once()
    
    def test_frame_extraction_failure(self, mock_video_capture):
        """Test handling of frame extraction failure."""
        # Set up read to return False (extraction failure)
        mock_video_capture.return_value.read.return_value = (False, None)
        
        with patch('processing.frame_processing.process_image') as mock_process_image:
            # Call the function
            process_video_frame("test_video.mp4", display_rois=False, debug=True,
                               start_time=10, end_time=20)
            
            # Verify VideoCapture was called
            mock_video_capture.assert_called_once_with("test_video.mp4")
            
            # Verify process_image was not called
            mock_process_image.assert_not_called()

# Tests for process_frame
class TestProcessFrame:
    
    def test_basic_functionality(self, mock_video_capture, mock_extract_data):
        """Test the basic functionality of process_frame."""
        with patch('processing.frame_processing.process_image') as mock_process_image, \
             patch('processing.frame_processing.cv2.imwrite') as mock_imwrite:
            
            # Call the function
            process_frame("test_video.mp4", frame_number=50, display_rois=False,
                         debug=True, output_filename="output.jpg")
            
            # Verify VideoCapture was called with the correct path
            mock_video_capture.assert_called_once_with("test_video.mp4")
            
            # Verify frame was saved and process_image was called
            assert mock_imwrite.called
            mock_process_image.assert_called_once_with("output.jpg", False, True)
    
    def test_video_load_failure(self, mock_video_capture):
        """Test handling when video loading fails."""
        # Set up isOpened to return False (failure to load)
        mock_video_capture.return_value.isOpened.return_value = False
        
        with patch('processing.frame_processing.process_image') as mock_process_image:
            # Call the function
            process_frame("nonexistent.mp4", frame_number=50, display_rois=False,
                         debug=True, output_filename="output.jpg")
            
            # Verify VideoCapture was called
            mock_video_capture.assert_called_once_with("nonexistent.mp4")
            
            # Verify process_image was not called
            mock_process_image.assert_not_called()
    
    def test_frame_number_out_of_range(self, mock_video_capture):
        """Test handling of frame number out of range."""
        # Call with frame number > frame count
        with patch('processing.frame_processing.process_image') as mock_process_image:
            process_frame("test_video.mp4", frame_number=2000, display_rois=False,
                         debug=True, output_filename="output.jpg")
            
            # Verify video was opened and then closed
            mock_video_capture.assert_called_once_with("test_video.mp4")
            mock_video_capture.return_value.release.assert_called_once()
            
            # Verify process_image was not called
            mock_process_image.assert_not_called()
    
    def test_frame_extraction_failure(self, mock_video_capture):
        """Test handling of frame extraction failure."""
        # Set up read to return False (extraction failure)
        mock_video_capture.return_value.read.return_value = (False, None)
        
        with patch('processing.frame_processing.process_image') as mock_process_image:
            # Call the function
            process_frame("test_video.mp4", frame_number=50, display_rois=False,
                         debug=True, output_filename="output.jpg")
            
            # Verify VideoCapture was called
            mock_video_capture.assert_called_once_with("test_video.mp4")
            
            # Verify process_image was not called
            mock_process_image.assert_not_called()
