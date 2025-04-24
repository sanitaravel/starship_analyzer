import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY

from processing.video_processing.batch_processing import (
    create_batches,
    process_batch,
    process_video_frames,
    summarize_batch
)

@pytest.fixture
def sample_batch():
    """Create a sample batch of frame numbers."""
    return [0, 5, 10, 15, 20]

@pytest.fixture
def sample_batch_results():
    """Create sample results for a batch of frames."""
    return [
        {
            "frame_number": 0,
            "superheavy": {"speed": 100},
            "starship": {"altitude": 5000},
            "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
        },
        {
            "frame_number": 5,
            "superheavy": {"speed": 120},
            "starship": {"altitude": 6000},
            "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 10}
        },
        {
            "frame_number": 10,
            "error": "Failed to process"
        },
        {
            "frame_number": 15,
            "superheavy": {"speed": 140},
            "starship": {"altitude": 7000},
            "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 30}
        },
        {
            "frame_number": 20,
            "superheavy": {"speed": 160},
            "starship": {"altitude": 8000},
            "time": {"sign": "+", "hours": 0, "minutes": 1, "seconds": 0}
        }
    ]

class TestCreateBatches:
    
    def test_basic_functionality(self):
        """Test basic batch creation."""
        batches = create_batches(10, 3)
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    
    def test_with_sampling(self):
        """Test batch creation with sampling."""
        batches = create_batches(20, 3, sample_rate=2)
        assert batches == [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18]]
    
    def test_empty_result(self):
        """Test with zero frames."""
        batches = create_batches(0, 5)
        assert batches == []

class TestProcessBatch:
    
    @patch('processing.video_processing.batch_processing.cv2.VideoCapture')
    @patch('processing.video_processing.batch_processing.process_frame')
    def test_basic_functionality(self, mock_process_frame, mock_video_capture, sample_batch):
        """Test basic batch processing."""
        # Configure mocks
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3)))
        
        # Configure process_frame to return success
        mock_process_frame.return_value = {
            "frame_number": 0,
            "superheavy": {"speed": 100},
            "starship": {"altitude": 5000},
            "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
        }
        
        # Call the function
        results = process_batch(sample_batch, "test_video.mp4", False, False, False)
        
        # Verify results
        assert len(results) == 5
        assert all("frame_number" in r for r in results)
        
        # Verify process_frame was called for each frame
        assert mock_process_frame.call_count == 5
    
    @patch('processing.video_processing.batch_processing.cv2.VideoCapture')
    def test_video_open_failure(self, mock_video_capture, sample_batch):
        """Test handling of video open failure."""
        # Configure mock to fail opening
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Call the function
        results = process_batch(sample_batch, "test_video.mp4", False, False, False)
        
        # Verify error results
        assert len(results) == 5
        assert all("error" in r for r in results)
    
    @patch('processing.video_processing.batch_processing.cv2.VideoCapture')
    @patch('processing.video_processing.batch_processing.process_frame')
    def test_progress_counter(self, mock_process_frame, mock_video_capture, sample_batch):
        """Test progress counter updating."""
        # Configure mocks
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3)))
        
        # Configure process_frame to return a complete result with time key
        mock_process_frame.return_value = {
            "frame_number": 0, 
            "superheavy": {},
            "starship": {},
            "time": {"sign": "+", "hours": 1, "minutes": 0, "seconds": 0}
        }
        
        # Create a mock counter
        counter = MagicMock()
        counter.value = 0
        
        # Call the function
        process_batch(sample_batch, "test_video.mp4", False, False, False, counter)
        
        # Verify counter was updated
        assert counter.value == 5
    
    @patch('processing.video_processing.batch_processing.cv2.VideoCapture')
    def test_frame_read_failure(self, mock_video_capture, sample_batch):
        """Test handling of frame read failure."""
        # Configure mock to fail reading
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        
        # Call the function
        results = process_batch(sample_batch, "test_video.mp4", False, False, False)
        
        # Verify results still has items for each frame
        assert len(results) == 0  # No results added since reads failed

class TestProcessVideoFrames:
    
    @patch('processing.video_processing.batch_processing.ProcessPoolExecutor')
    @patch('processing.video_processing.batch_processing.tqdm')
    @patch('processing.video_processing.batch_processing.as_completed')  # Add direct patch for as_completed
    def test_basic_functionality(self, mock_as_completed, mock_tqdm, mock_executor):
        """Test basic functionality of process_video_frames."""
        # Configure mock executor
        mock_future = MagicMock()
        mock_future.result.return_value = [
            {
                "frame_number": 0,
                "superheavy": {"speed": 100},
                "starship": {"altitude": 5000},
                "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 0}
            }
        ]
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance
        
        # Configure as_completed to immediately return the futures
        mock_as_completed.return_value = [mock_future]
        
        # Setup mock tqdm - properly configure the context manager
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
        
        # Call the function
        with patch('processing.video_processing.batch_processing.multiprocessing.Manager') as mock_manager:
            # Configure Manager value
            mock_value = MagicMock()
            mock_value.value = 0
            mock_manager.return_value.Value.return_value = mock_value
            
            # Patch torch to avoid GPU-related code
            with patch('processing.video_processing.batch_processing.torch', create=True) as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                
                results, zero_time_frame = process_video_frames([[0]], "test_video.mp4", False, False)
        
        # Verify results
        assert len(results) == 1
        assert results[0]["frame_number"] == 0
        assert zero_time_frame == 0  # This frame has time 0:0:0
    
    @patch('processing.video_processing.batch_processing.ProcessPoolExecutor')
    @patch('processing.video_processing.batch_processing.tqdm')
    @patch('processing.video_processing.batch_processing.as_completed')  # Add direct patch for as_completed
    def test_error_handling(self, mock_as_completed, mock_tqdm, mock_executor):
        """Test error handling in process_video_frames."""
        # Configure mock executor to raise an exception
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Test error")
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance
        
        # Configure as_completed to immediately return the futures
        mock_as_completed.return_value = [mock_future]
        
        # Setup mock tqdm - properly configure the context manager
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
        
        # Call the function
        with patch('processing.video_processing.batch_processing.multiprocessing.Manager') as mock_manager, \
             patch('processing.video_processing.batch_processing.logger') as mock_logger, \
             patch('processing.video_processing.batch_processing.torch', create=True) as mock_torch:
            
            # Configure Manager value
            mock_value = MagicMock()
            mock_value.value = 0
            mock_manager.return_value.Value.return_value = mock_value
            
            # Configure torch mock to avoid GPU complexity
            mock_torch.cuda.is_available.return_value = False
            
            results, zero_time_frame = process_video_frames([[0]], "test_video.mp4", False, False)
        
        # Verify error was logged
        mock_logger.error.assert_called_once_with(
            "Error processing batch: Test error"
        )
        
        # Verify empty results
        assert results == []
        assert zero_time_frame is None

class TestSummarizeBatch:
    
    def test_basic_functionality(self, sample_batch_results):
        """Test basic functionality of summarize_batch."""
        summary = summarize_batch(sample_batch_results, 0, 25)
        
        # Verify summary structure
        assert summary["start_frame"] == 0
        assert summary["end_frame"] == 25
        assert summary["frame_count"] == 5
        assert summary["results"] == sample_batch_results
    
    def test_empty_batch(self):
        """Test summarizing an empty batch."""
        summary = summarize_batch([], 0, 10)
        
        # Verify summary structure for empty batch
        assert summary["start_frame"] == 0
        assert summary["end_frame"] == 10
        assert summary["frame_count"] == 0
        assert summary["results"] == []
