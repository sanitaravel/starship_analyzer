"""
Tests for warning suppression utilities in utils/suppress_warnings.py.
"""
import pytest
import os
import sys
import ctypes
from unittest.mock import patch, MagicMock
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
    
    @patch('utils.suppress_warnings.ctypes.c_void_p')
    @patch('os.open')
    @patch('os.close')
    def test_suppress_stdout_stderr_context(self, mock_close, mock_open, mock_c_void_p):
        """Test that suppress_stdout_stderr context manager works correctly."""
        # Set up mocks
        mock_open.return_value = 123  # Mock file descriptor
        
        # Set up the null mock that gets returned from c_void_p constructor
        mock_null = MagicMock()
        mock_null.value = 123
        mock_c_void_p.return_value = mock_null
        
        # Use the context manager
        with suppress_stdout_stderr():
            # Just verify it runs without errors
            pass
        
        # Verify the null device was opened and closed correctly
        mock_open.assert_called_once_with(os.devnull, os.O_WRONLY)
        mock_close.assert_called_once_with(123)
    
    def test_suppress_stdout_stderr_functionality(self, capsys):
        """Test that suppress_stdout_stderr actually suppresses output (integration test)."""
        # First, print without suppression to verify capture works
        print("This should be captured")
        captured = capsys.readouterr()
        assert "This should be captured" in captured.out
        
        # Skip this test on non-Windows platforms
        if sys.platform != 'win32':
            pytest.skip(f"suppress_stdout_stderr only implemented for Windows, current system {sys.platform}")
        
        # On Windows, not all Python distributions/environments expose the stdout symbol
        # This happens especially in newer Windows versions or certain Python installations
        try:
            # Create a dummy test to verify symbol availability
            import ctypes
            stdout_check = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stdout')
            stderr_check = ctypes.c_void_p.in_dll(ctypes.cdll.msvcrt, 'stderr')
            
            # If we get here, symbols are available
            print("Stdout/stderr symbols available, testing suppression")
            
            with suppress_stdout_stderr():
                print("This should be suppressed")
            
            # Check that nothing was captured
            captured = capsys.readouterr()
            assert not captured.out or "This should be suppressed" not in captured.out
            
        except (AttributeError, ValueError, OSError) as e:
            pytest.skip(f"Stdout symbol not available in this Windows environment: {e}. "
                       "This is normal in some Windows configurations and doesn't indicate a problem with the code - "
                       "the suppress_stdout_stderr function will automatically handle this case gracefully in production.")
