"""
Tests for the video_menu module.
"""
import pytest
from unittest.mock import patch, MagicMock
import inquirer

# Import the module to test
from ui.video_menu import (
    video_processing_menu,
    process_random_frame,
    select_video_file,
    get_processing_parameters,
    get_time_based_borders,
    get_frame_based_borders,
    process_video_with_parameters,
    process_complete_video
)

class TestVideoMenu:
    """Tests for the video menu functions."""
    
    @patch('ui.video_menu.inquirer.prompt')
    @patch('ui.video_menu.process_random_frame')
    @patch('ui.video_menu.process_complete_video')
    @patch('ui.video_menu.clear_screen')
    def test_video_processing_menu_random_frame(self, mock_clear, mock_process_complete, 
                                              mock_process_random, mock_prompt):
        """Test video processing menu when selecting to process a random frame."""
        # Setup mock responses
        mock_prompt.side_effect = [
            {'action': 'Process random video frame'},
            {'action': 'Back to main menu'}
        ]
        
        # Call function
        result = video_processing_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        mock_process_random.assert_called_once()
        mock_process_complete.assert_not_called()
        assert mock_prompt.call_count == 2
    
    @patch('ui.video_menu.inquirer.prompt')
    @patch('ui.video_menu.process_random_frame')
    @patch('ui.video_menu.process_complete_video')
    @patch('ui.video_menu.clear_screen')
    def test_video_processing_menu_complete_video(self, mock_clear, mock_process_complete, 
                                                mock_process_random, mock_prompt):
        """Test video processing menu when selecting to process a complete video."""
        # Setup mock responses
        mock_prompt.side_effect = [
            {'action': 'Process complete video'},
            {'action': 'Back to main menu'}
        ]
        
        # Call function
        result = video_processing_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        mock_process_complete.assert_called_once()
        mock_process_random.assert_not_called()
        assert mock_prompt.call_count == 2
    
    @patch('ui.video_menu.inquirer.prompt')
    @patch('ui.video_menu.clear_screen')
    def test_video_processing_menu_back(self, mock_clear, mock_prompt):
        """Test video processing menu when selecting to go back to main menu."""
        # Setup mock responses
        mock_prompt.return_value = {'action': 'Back to main menu'}
        
        # Call function
        result = video_processing_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        assert mock_prompt.call_count == 1
    
    @patch('ui.video_menu.get_video_files_from_flight_recordings')
    @patch('ui.video_menu.inquirer.prompt')
    @patch('ui.video_menu.display_video_info')
    @patch('ui.video_menu.process_video_frame')
    @patch('ui.video_menu.input')
    @patch('ui.video_menu.clear_screen')
    def test_process_random_frame(self, mock_clear, mock_input, mock_process_frame, 
                                 mock_display_info, mock_prompt, mock_get_videos):
        """Test processing a random video frame."""
        # Setup mocks
        mock_get_videos.return_value = ['video1.mp4', 'video2.mp4']
        mock_prompt.side_effect = [
            {'video_path': 'video1.mp4'},
            {
                'display_rois': True, 
                'debug': False, 
                'start_time': '10', 
                'end_time': '20'
            }
        ]
        
        # Mock DEBUG_MODE import
        with patch.dict('sys.modules', {'main': MagicMock()}):
            import sys
            sys.modules['main'].DEBUG_MODE = False
            
            # Call function
            result = process_random_frame()
            
            # Verify results
            assert result is True
            mock_get_videos.assert_called_once()
            assert mock_prompt.call_count == 2
            mock_display_info.assert_called_once_with('video1.mp4')
            mock_process_frame.assert_called_once_with('video1.mp4', True, False, 10, 20)
            mock_input.assert_called_once()
            mock_clear.assert_called()
    
    @patch('ui.video_menu.get_video_files_from_flight_recordings')
    @patch('ui.video_menu.inquirer.prompt')
    def test_select_video_file_successful(self, mock_prompt, mock_get_videos):
        """Test selecting a video file successfully."""
        # Setup mocks
        mock_get_videos.return_value = ['video1.mp4', 'video2.mp4']
        mock_prompt.return_value = {'video_path': 'video1.mp4'}
        
        # Call function
        result = select_video_file()
        
        # Verify results
        assert result == 'video1.mp4'
        mock_get_videos.assert_called_once()
        mock_prompt.assert_called_once()
    
    @patch('ui.video_menu.get_video_files_from_flight_recordings')
    def test_select_video_file_no_videos(self, mock_get_videos):
        """Test selecting a video file when no videos are available."""
        # Setup mocks
        mock_get_videos.return_value = []
        
        # Call function
        result = select_video_file()
        
        # Verify results
        assert result is None
        mock_get_videos.assert_called_once()
    
    @patch('ui.video_menu.inquirer.prompt')
    def test_get_processing_parameters(self, mock_prompt):
        """Test getting processing parameters."""
        # Setup mock
        expected_result = {
            'launch_number': '5',
            'batch_size': '10',
            'sample_rate': '2',
            'border_type': 'Time-based (seconds)'
        }
        mock_prompt.return_value = expected_result
        
        # Call function
        result = get_processing_parameters()
        
        # Verify results
        assert result == expected_result
        mock_prompt.assert_called_once()
        
        # Verify that the questions list was passed to prompt
        args, _ = mock_prompt.call_args
        assert len(args[0]) == 4  # Should have 4 questions
        assert args[0][0].name == 'launch_number'
        assert args[0][3].name == 'border_type'
    
    @patch('ui.video_menu.inquirer.prompt')
    def test_get_time_based_borders(self, mock_prompt):
        """Test getting time-based borders."""
        # Setup mock
        mock_prompt.return_value = {
            'start_time': '5',
            'end_time': '30'
        }
        
        # Call function
        start_time, end_time = get_time_based_borders()
        
        # Verify results
        assert start_time == 5.0
        assert end_time == 30.0
        mock_prompt.assert_called_once()
    
    @patch('ui.video_menu.inquirer.prompt')
    def test_get_time_based_borders_defaults(self, mock_prompt):
        """Test getting time-based borders with defaults."""
        # Setup mock
        mock_prompt.return_value = {
            'start_time': '',
            'end_time': ''
        }
        
        # Call function
        start_time, end_time = get_time_based_borders()
        
        # Verify results
        assert start_time == 0
        assert end_time is None
        mock_prompt.assert_called_once()
    
    @patch('ui.video_menu.inquirer.prompt')
    def test_get_frame_based_borders(self, mock_prompt):
        """Test getting frame-based borders."""
        # Setup mock
        mock_prompt.return_value = {
            'start_frame': '100',
            'end_frame': '500'
        }
        
        # Call function
        start_frame, end_frame = get_frame_based_borders()
        
        # Verify results
        assert start_frame == 100
        assert end_frame == 500
        mock_prompt.assert_called_once()
    
    @patch('ui.video_menu.inquirer.prompt')
    def test_get_frame_based_borders_defaults(self, mock_prompt):
        """Test getting frame-based borders with defaults."""
        # Setup mock
        mock_prompt.return_value = {
            'start_frame': '',
            'end_frame': ''
        }
        
        # Call function
        start_frame, end_frame = get_frame_based_borders()
        
        # Verify results
        assert start_frame == 0
        assert end_frame is None
        mock_prompt.assert_called_once()
    
    @patch('ui.video_menu.iterate_through_frames')
    @patch('ui.video_menu.logger')
    def test_process_video_with_parameters(self, mock_logger, mock_iterate):
        """Test processing a video with parameters."""
        # Mock DEBUG_MODE import
        with patch.dict('sys.modules', {'main': MagicMock()}):
            import sys
            sys.modules['main'].DEBUG_MODE = False
            
            # Call function
            process_video_with_parameters(
                'video1.mp4', '5', 10, 2, 
                start_time=5, end_time=30, 
                start_frame=None, end_frame=None
            )
            
            # Verify results
            mock_logger.debug.assert_called()
            mock_iterate.assert_called_once_with(
                'video1.mp4', 5, debug=False, 
                batch_size=10, sample_rate=2,
                start_time=5, end_time=30, 
                start_frame=None, end_frame=None
            )
    
    @patch('ui.video_menu.select_video_file')
    @patch('ui.video_menu.display_video_info')
    @patch('ui.video_menu.get_processing_parameters')
    @patch('ui.video_menu.get_time_based_borders')
    @patch('ui.video_menu.get_frame_based_borders')
    @patch('ui.video_menu.process_video_with_parameters')
    @patch('ui.video_menu.input')
    @patch('ui.video_menu.clear_screen')
    def test_process_complete_video_time_based(self, mock_clear, mock_input, mock_process, 
                                             mock_get_frame, mock_get_time, mock_get_params, 
                                             mock_display, mock_select):
        """Test processing a complete video with time-based borders."""
        # Setup mocks
        mock_select.return_value = 'video1.mp4'
        mock_get_params.return_value = {
            'launch_number': '5',
            'batch_size': '10',
            'sample_rate': '2',
            'border_type': 'Time-based (seconds)'
        }
        mock_get_time.return_value = (5.0, 30.0)
        
        # Mock DEBUG_MODE import
        with patch.dict('sys.modules', {'main': MagicMock()}):
            # Call function
            result = process_complete_video()
            
            # Verify results
            assert result is True
            mock_select.assert_called_once()
            mock_display.assert_called_once_with('video1.mp4')
            mock_get_params.assert_called_once()
            mock_get_time.assert_called_once()
            mock_get_frame.assert_not_called()
            mock_process.assert_called_once_with(
                'video1.mp4', '5', 10, 2, 5.0, 30.0, None, None
            )
            mock_input.assert_called_once()
            mock_clear.assert_called()
    
    @patch('ui.video_menu.select_video_file')
    @patch('ui.video_menu.display_video_info')
    @patch('ui.video_menu.get_processing_parameters')
    @patch('ui.video_menu.get_time_based_borders')
    @patch('ui.video_menu.get_frame_based_borders')
    @patch('ui.video_menu.process_video_with_parameters')
    @patch('ui.video_menu.input')
    @patch('ui.video_menu.clear_screen')
    def test_process_complete_video_frame_based(self, mock_clear, mock_input, mock_process, 
                                              mock_get_frame, mock_get_time, mock_get_params, 
                                              mock_display, mock_select):
        """Test processing a complete video with frame-based borders."""
        # Setup mocks
        mock_select.return_value = 'video1.mp4'
        mock_get_params.return_value = {
            'launch_number': '5',
            'batch_size': '10',
            'sample_rate': '2',
            'border_type': 'Frame-based'
        }
        mock_get_frame.return_value = (100, 500)
        
        # Mock DEBUG_MODE import
        with patch.dict('sys.modules', {'main': MagicMock()}):
            # Call function
            result = process_complete_video()
            
            # Verify results
            assert result is True
            mock_select.assert_called_once()
            mock_display.assert_called_once_with('video1.mp4')
            mock_get_params.assert_called_once()
            mock_get_time.assert_not_called()
            mock_get_frame.assert_called_once()
            mock_process.assert_called_once_with(
                'video1.mp4', '5', 10, 2, None, None, 100, 500
            )
            mock_input.assert_called_once()
            mock_clear.assert_called()
    
    @patch('ui.video_menu.select_video_file')
    def test_process_complete_video_no_video(self, mock_select):
        """Test processing a complete video when no video is selected."""
        # Setup mocks
        mock_select.return_value = None
        
        # Call function
        result = process_complete_video()
        
        # Verify results
        assert result is True
        mock_select.assert_called_once()
