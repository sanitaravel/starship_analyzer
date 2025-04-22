"""
Tests for video utility functions in utils/video_utils.py.
"""
import pytest
import os
import cv2
import subprocess
import json
from unittest.mock import patch, MagicMock, mock_open

from utils.video_utils import (
    get_video_files_from_flight_recordings,
    display_video_info,
    get_video_info,
    try_alternative_decoder
)

class TestVideoUtils:
    """Test suite for video utilities."""
    
    @patch('os.path.exists')
    @patch('os.walk')
    def test_get_video_files_from_flight_recordings(self, mock_walk, mock_exists):
        """Test getting video files from the flight_recordings folder."""
        # Setup mock responses
        mock_exists.return_value = True
        mock_walk.return_value = [
            ('./flight_recordings', ['folder1'], ['video1.mp4', 'text.txt']),
            ('./flight_recordings/folder1', [], ['video2.avi', 'video3.mov', 'image.jpg'])
        ]
        
        # Call the function
        result = get_video_files_from_flight_recordings()
        
        # Verify results
        assert len(result) == 3
        assert ('video1.mp4', 'flight_recordings\\video1.mp4') in result
        assert ('video2.avi', 'flight_recordings\\folder1\\video2.avi') in result
        assert ('video3.mov', 'flight_recordings\\folder1\\video3.mov') in result
        
    @patch('os.path.exists')
    @patch('os.walk')
    def test_get_video_files_empty(self, mock_walk, mock_exists):
        """Test getting video files when none exist."""
        # Setup mock responses
        mock_exists.return_value = True
        mock_walk.return_value = [
            ('./flight_recordings', [], ['text.txt']),
        ]
        
        # Call the function with patched print to check output
        with patch('builtins.print') as mock_print:
            result = get_video_files_from_flight_recordings()
            
            # Verify results
            assert len(result) == 0
            mock_print.assert_called_with("No video files found in flight_recordings folder.")
    
    @patch('builtins.print')
    @patch('cv2.VideoCapture')
    def test_display_video_info(self, mock_video_capture, mock_print):
        """Test displaying video information."""
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 9000.0,
            cv2.CAP_PROP_FOURCC: 875967048.0,  # ASCII for 'H264'
        }.get(prop, 0.0)
        
        mock_video_capture.return_value = mock_cap
        
        # Call the function
        display_video_info('test_video.mp4')
        
        # Verify function calls
        mock_video_capture.assert_called_once_with('test_video.mp4')
        mock_cap.isOpened.assert_called_once()
        
        # Verify print statements are called
        mock_print.assert_any_call("\n----- Video Information -----")
        mock_print.assert_any_call("Resolution: 1920x1080")
        mock_print.assert_any_call("Frame Rate: 30.00 fps")
        mock_print.assert_any_call("Total Frames: 9000")
        mock_print.assert_any_call("Duration: 00:05:00")  # 9000 frames at 30 fps = 5 minutes
        
    @patch('cv2.VideoCapture')
    def test_display_video_info_cant_open(self, mock_video_capture):
        """Test displaying video information when video can't be opened."""
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        # Call the function with mocked print
        with patch('builtins.print') as mock_print:
            display_video_info('invalid_video.mp4')
            
            # Verify error message is printed
            mock_print.assert_called_with("Error: Could not open video file invalid_video.mp4")
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('cv2.VideoCapture')
    @patch('subprocess.run')
    def test_get_video_info(self, mock_run, mock_video_capture, mock_getsize, mock_exists):
        """Test getting detailed video information."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 104857600  # 100 MB
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 9000.0,
            cv2.CAP_PROP_FOURCC: 875967048.0,  # ASCII for 'H264'
        }.get(prop, 0.0)
        
        mock_video_capture.return_value = mock_cap
        
        # Mock FFprobe result
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080
                }
            ],
            "format": {
                "duration": "300.0"
            }
        })
        mock_run.return_value = ffprobe_result
        
        # Call the function
        result = get_video_info('test_video.mp4')
        
        # Verify result fields
        assert result["path"] == 'test_video.mp4'
        assert result["exists"] == True
        assert result["size_mb"] == 100.0
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["fps"] == 30.0
        assert result["frame_count"] == 9000
        assert result["duration"] == 300.0
        assert result["codec"] == "H264"
        assert "ffprobe" in result
        
    @patch('cv2.VideoCapture')
    def test_try_alternative_decoder_success(self, mock_video_capture):
        """Test alternative decoder with successful frame read."""
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())  # Mock successful frame read
        mock_video_capture.return_value = mock_cap
        
        # Call the function
        result = try_alternative_decoder('test_video.mp4')
        
        # Verify result
        assert result == True
        mock_video_capture.assert_called_once_with('test_video.mp4', cv2.CAP_FFMPEG)
        
    @patch('cv2.VideoCapture')
    def test_try_alternative_decoder_failure(self, mock_video_capture):
        """Test alternative decoder with failed frame read."""
        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Mock failed frame read
        mock_video_capture.return_value = mock_cap
        
        # Call the function
        result = try_alternative_decoder('test_video.mp4')
        
        # Verify result
        assert result == False
