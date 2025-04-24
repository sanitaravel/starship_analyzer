import pytest
import cv2
from unittest.mock import patch, MagicMock, ANY

from processing.video_processing.main_processing import iterate_through_frames, process_frames


class TestIterateThroughFrames:
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    @patch('processing.video_processing.main_processing.process_video_frames')
    @patch('processing.video_processing.main_processing.calculate_real_times')
    @patch('processing.video_processing.main_processing.save_results')
    def test_basic_functionality(self, mock_save, mock_calculate, mock_process, mock_cap, mock_validate):
        """Test the basic functionality of iterate_through_frames."""
        # Configure mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        
        # Set up process_video_frames return value
        mock_process.return_value = ([], 500)  # empty results, zero_time_frame=500
        
        # Call the function
        iterate_through_frames("test_video.mp4", 42, debug=True)
        
        # Verify validate_video was called
        mock_validate.assert_called_once_with("test_video.mp4")
        
        # Verify process_video_frames was called
        mock_process.assert_called_once()
        
        # Verify calculate_real_times was called
        mock_calculate.assert_called_once_with([], 500, 30)
        
        # Verify save_results was called
        mock_save.assert_called_once()
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=False)
    def test_validation_failure(self, mock_validate):
        """Test handling of video validation failure."""
        with patch('processing.video_processing.main_processing.logger') as mock_logger:
            iterate_through_frames("invalid_video.mp4", 42)
            
            # Verify error was logged
            mock_logger.error.assert_called_with("Video validation failed, aborting processing")
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    @patch('processing.video_processing.main_processing.process_video_frames')
    def test_with_time_bounds(self, mock_process, mock_cap, mock_validate):
        """Test with time bounds specified."""
        # Configure mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        
        # Set up process_video_frames return value
        mock_process.return_value = ([], None)  # empty results, no zero_time_frame
        
        # Call with time bounds
        iterate_through_frames("test_video.mp4", 42, start_time=10, end_time=20)
        
        # Get the created batches from the call to process_video_frames
        batches_arg = mock_process.call_args[0][0]
        
        # Verify start and end frames based on time bounds
        first_frame = batches_arg[0][0] if batches_arg else None
        last_frames = batches_arg[-1] if batches_arg else []
        last_frame = last_frames[-1] if last_frames else None
        
        # start_time=10 at 30fps should give frame 300
        # end_time=20 at 30fps should give frame 600
        assert 295 <= first_frame <= 305
        assert 595 <= last_frame <= 605
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    @patch('processing.video_processing.main_processing.process_video_frames')
    def test_with_frame_bounds(self, mock_process, mock_cap, mock_validate):
        """Test with frame bounds specified."""
        # Configure mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        
        # Set up process_video_frames return value
        mock_process.return_value = ([], None)
        
        # Call with frame bounds
        iterate_through_frames("test_video.mp4", 42, start_frame=300, end_frame=600)
        
        # Get the created batches from the call to process_video_frames
        batches_arg = mock_process.call_args[0][0]
        
        # Verify correct frame bounds
        first_frame = batches_arg[0][0] if batches_arg else None
        last_frames = batches_arg[-1] if batches_arg else []
        last_frame = last_frames[-1] if last_frames else None
        
        # Should use exact frame numbers
        assert first_frame == 300
        assert 590 <= last_frame <= 599  # Last batch may not end exactly at 600
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    @patch('processing.video_processing.main_processing.process_video_frames')
    def test_with_max_frames(self, mock_process, mock_cap, mock_validate):
        """Test with max_frames limit."""
        # Configure mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        
        # Set up process_video_frames return value
        mock_process.return_value = ([], None)
        
        # Call with max_frames
        iterate_through_frames("test_video.mp4", 42, max_frames=100)
        
        # Get the created batches from the call to process_video_frames
        batches_arg = mock_process.call_args[0][0]
        
        # Count frames in batches
        total_frames = sum(len(batch) for batch in batches_arg)
        
        # Should not exceed max_frames
        assert total_frames <= 100


class TestProcessFrames:
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    @patch('processing.video_processing.main_processing.process_single_frame')
    @patch('processing.video_processing.main_processing.summarize_batch')
    def test_basic_functionality(self, mock_summarize, mock_process, mock_cap, mock_validate):
        """Test basic functionality of process_frames."""
        # Configure mocks
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        mock_cap_instance.read.return_value = (True, MagicMock())
        
        # Configure process_single_frame to return test data
        frame_result = {"frame_number": 0, "superheavy": {}, "starship": {}, "time": {}}
        mock_process.return_value = frame_result
        
        # Configure summarize_batch to return test summary
        mock_summarize.return_value = {"start_frame": 0, "end_frame": 30, "results": [frame_result] * 30}
        
        # Call the function
        result = process_frames("test_video.mp4", batch_size=30)
        
        # Verify process_single_frame was called for each frame
        assert mock_process.call_count == 100  # Should be called for all frames
        
        # Verify summarize_batch was called for each batch (100 frames / 30 batch_size = 4 batches)
        assert mock_summarize.call_count == 4
        
        # Check the first and last batch boundaries
        first_call_args = mock_summarize.call_args_list[0][0]
        assert first_call_args[1] == 0  # start_frame of first batch
        assert first_call_args[2] == 30  # end_frame of first batch
        
        last_call_args = mock_summarize.call_args_list[3][0]
        assert last_call_args[1] == 90  # start_frame of last batch
        assert last_call_args[2] == 100  # end_frame of last batch

    @patch('processing.video_processing.main_processing.validate_video', return_value=False)
    def test_validation_failure(self, mock_validate):
        """Test handling of video validation failure."""
        result = process_frames("invalid_video.mp4")
        
        # Should return None when validation fails
        assert result is None
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    def test_with_time_bounds(self, mock_cap, mock_validate):
        """Test with time bounds."""
        # Configure mock
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        mock_cap_instance.read.return_value = (False, None)  # To end loop early
        
        # Call with time bounds
        with patch('processing.video_processing.main_processing.logger') as mock_logger:
            process_frames("test_video.mp4", start_time=10, end_time=20)
        
        # Verify cap.set was called to position at the right frame
        mock_cap_instance.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 300)  # 10 seconds * 30 fps
    
    @patch('processing.video_processing.main_processing.validate_video', return_value=True)
    @patch('processing.video_processing.main_processing.cv2.VideoCapture')
    def test_frame_read_failure(self, mock_cap, mock_validate):
        """Test handling of frame read failure."""
        # Configure mock to fail reading
        mock_cap_instance = MagicMock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        mock_cap_instance.read.return_value = (False, None)
        
        # Call the function
        with patch('processing.video_processing.main_processing.logger') as mock_logger:
            result = process_frames("test_video.mp4")
        
        # Warning should be logged
        mock_logger.warning.assert_called_with(ANY)
        
        # Should return empty results
        assert result == []
