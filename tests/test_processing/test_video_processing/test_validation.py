import pytest
import os
import tempfile
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from processing.video_processing.validation import validate_video, get_video_properties


@pytest.fixture
def mock_video_path():
    return "/path/to/mock/video.mp4"


@pytest.fixture
def mock_cv2():
    with patch('processing.video_processing.validation.cv2') as mock:
        # Set up VideoCapture mock
        mock_cap = MagicMock()
        mock.VideoCapture.return_value = mock_cap
        
        # Configure basic properties
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: int.from_bytes(b'mp4v', byteorder='little'),
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        
        # Configure read to return a test frame
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        
        yield mock


class TestValidateVideo:
    
    def test_nonexistent_file(self, mock_video_path):
        """Test validation of a non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = validate_video(mock_video_path)
            assert result is False
    
    def test_small_file(self, mock_video_path):
        """Test validation of a suspiciously small file."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=500000), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            
            # Create a completely controlled mock for this specific test
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            
            # Configure the mock to return appropriate values
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,  # Must be > 0 to pass validation
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FOURCC: int.from_bytes(b'mp4v', byteorder='little')
            }.get(prop, 0)
            
            # Make sure read always returns a valid frame
            mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            
            result = validate_video(mock_video_path)
            assert result is True  # Still valid, just a warning
    
    def test_non_mp4_file(self, mock_video_path):
        """Test validation of a non-MP4 file."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=10*1024*1024), \
             patch('os.path.splitext', return_value=("video", ".avi")), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            
            # Create a completely controlled mock for this specific test
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            
            # Configure the mock to return appropriate values
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,  # Must be > 0 to pass validation
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FOURCC: int.from_bytes(b'avi1', byteorder='little')
            }.get(prop, 0)
            
            # Make sure read always returns a valid frame
            mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            
            result = validate_video(mock_video_path)
            assert result is True  # Still valid, just a warning
    
    def test_unopenable_file(self, mock_video_path):
        """Test validation when file can't be opened."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=10*1024*1024), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap:
            
            # Make isOpened return False
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_cap.return_value = mock_instance
            
            result = validate_video(mock_video_path)
            assert result is False
    
    def test_zero_frames(self, mock_video_path):
        """Test validation when video has no frames."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=10*1024*1024), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap:
            
            # Configure mock
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: 0 if prop == cv2.CAP_PROP_FRAME_COUNT else 30
            mock_cap.return_value = mock_instance
            
            result = validate_video(mock_video_path)
            assert result is False
    
    def test_zero_fps(self, mock_video_path):
        """Test validation when FPS is zero."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=10*1024*1024), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            
            # Create a completely controlled mock for this specific test
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            
            # Configure the mock to return appropriate values
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,  # Must be > 0 to pass validation
                cv2.CAP_PROP_FPS: 0,  # Zero FPS (this should trigger a warning but still pass)
                cv2.CAP_PROP_FOURCC: int.from_bytes(b'mp4v', byteorder='little')
            }.get(prop, 0)
            
            # Make sure read always returns a valid frame
            mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            
            result = validate_video(mock_video_path)
            assert result is True  # Still valid, just a warning
    
    def test_frame_read_failure(self, mock_video_path, mock_cv2):
        """Test validation when frame reading fails."""
        # Make read return False
        mock_cv2.VideoCapture().read.return_value = (False, None)
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=10*1024*1024):
            result = validate_video(mock_video_path)
            assert result is False
    
    def test_valid_video(self, mock_video_path):
        """Test validation of a completely valid video."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100*1024*1024), \
             patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            
            # Create a completely controlled mock for this specific test
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            
            # Set all necessary properties for a valid video
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FOURCC: int.from_bytes(b'mp4v', byteorder='little')
            }.get(prop, 0)
            
            # Make sure read always returns a valid frame for all test positions
            mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            
            result = validate_video(mock_video_path)
            assert result is True


class TestGetVideoProperties:
    
    def test_basic_functionality(self, mock_video_path):
        """Test basic functionality of get_video_properties."""
        with patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            frame_count, fps = get_video_properties(mock_video_path)
            assert frame_count == 1000
            assert fps == 30.0
    
    def test_with_max_frames(self, mock_video_path):
        """Test with max_frames parameter."""
        with patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            frame_count, fps = get_video_properties(mock_video_path, max_frames=500)
            assert frame_count == 500  # Should be limited
            assert fps == 30.0
    
    def test_with_max_frames_larger_than_total(self, mock_video_path):
        """Test with max_frames larger than total frame count."""
        with patch('processing.video_processing.validation.cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 1000,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            frame_count, fps = get_video_properties(mock_video_path, max_frames=1500)
            assert frame_count == 1000  # Should be limited to actual count
            assert fps == 30.0


@pytest.mark.integration
class TestWithRealVideo:
    
    def create_test_video(self, duration=3, fps=30, width=640, height=480):
        """Create a real test video file for testing."""        
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        
        # Create a VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        # Write frames
        frame_count = int(duration * fps)
        for i in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add frame number as text
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
            
        out.release()
        return path
    
    @pytest.mark.skipif(not hasattr(cv2, 'VideoWriter'), reason="OpenCV VideoWriter not available")
    def test_real_video_validation(self):
        """Test validation with a real video file."""
        try:
            video_path = self.create_test_video()
            result = validate_video(video_path)
            assert result is True
            
            # Test properties
            frame_count, fps = get_video_properties(video_path)
            assert frame_count == 90  # 3 seconds * 30 fps
            assert fps == 30.0
            
        finally:
            # Clean up
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
