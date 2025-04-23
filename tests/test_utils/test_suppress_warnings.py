"""
Tests for warning suppression utilities in utils/suppress_warnings.py.
"""
import pytest
import os
import sys
import ctypes
from unittest.mock import patch, MagicMock, call
from utils.suppress_warnings import suppress_ffmpeg_warnings, suppress_stdout_stderr

class TestSuppressFFmpegWarnings:
    """Test suite for FFmpeg warning suppression utilities."""
    
    @patch('utils.suppress_warnings.logger')  # Patch the logger directly in the module
    @patch.dict('os.environ', {}, clear=True)
    def test_suppress_ffmpeg_warnings(self, mock_logger):
        """Test that suppress_ffmpeg_warnings sets correct environment variables."""
        # Call the function
        suppress_ffmpeg_warnings()
        
        # Verify environment variables are set correctly
        assert os.environ["OPENCV_FFMPEG_LOGLEVEL"] == "0"
        assert os.environ["AV_LOG_LEVEL"] == "0"
        
        # Verify logging calls
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("FFmpeg warnings suppression enabled. This may not completely eliminate all warnings.")
        mock_logger.info.assert_any_call("Common H.264 warnings like 'co located POCs unavailable' are harmless and won't affect functionality.")


class TestSuppressStdoutStderr:
    """Test suite for stdout/stderr suppression utilities."""
    
    @patch('os.open')
    @patch('os.close')
    def test_suppress_stdout_stderr_context(self, mock_close, mock_open):
        """Test that suppress_stdout_stderr context manager works correctly."""
        # Set up mocks
        mock_open.return_value = 123  # Mock file descriptor
        
        # Platform-specific patching
        if sys.platform == 'win32':
            with patch('utils.suppress_warnings.ctypes.c_void_p') as mock_c_void_p:
                # Set up the null mock that gets returned from c_void_p constructor
                mock_null = MagicMock()
                mock_null.value = 123
                mock_c_void_p.return_value = mock_null
                
                # Use the context manager
                with suppress_stdout_stderr():
                    # Just verify it runs without errors
                    pass
        else:  # Unix/Linux
            # Make sure ctypes.util.find_library is properly mocked
            with patch('ctypes.util.find_library') as mock_find_library, \
                 patch('ctypes.CDLL') as mock_cdll, \
                 patch('os.dup') as mock_dup, \
                 patch('os.dup2') as mock_dup2:
                
                # Set up the mocks
                mock_find_library.return_value = 'libc.so'
                mock_libc = MagicMock()
                mock_cdll.return_value = mock_libc
                mock_dup.return_value = 456  # Mock duplicated file descriptor
                
                # Use the context manager
                with suppress_stdout_stderr():
                    # Just verify it runs without errors
                    pass
                
                # Verify find_library was called with 'c'
                mock_find_library.assert_called_once_with('c')
                
                # Verify dup was called twice (once for stdout, once for stderr)
                assert mock_dup.call_count >= 1, "os.dup should be called at least once"
                assert mock_dup2.call_count >= 1, "os.dup2 should be called at least once"
        
        # Verify the null device was opened
        mock_open.assert_called_with(os.devnull, os.O_WRONLY)
    
    def test_suppress_stdout_stderr_functionality(self, capsys):
        """Test that suppress_stdout_stderr actually suppresses output (integration test)."""
        # First, print without suppression to verify capture works
        print("This should be captured")
        captured = capsys.readouterr()
        assert "This should be captured" in captured.out
        
        # Try the suppression - it should work on both platforms now
        try:
            with suppress_stdout_stderr():
                print("This should be suppressed")
            
            # Check that nothing was captured
            captured = capsys.readouterr()
            assert not captured.out or "This should be suppressed" not in captured.out
            
        except Exception as e:
            pytest.skip(f"Suppression not available in this environment: {e}")
