"""
Tests for terminal utility functions in utils/terminal.py.
"""
import pytest
from unittest.mock import patch
import platform
from utils.terminal import clear_screen

class TestTerminal:
    """Test suite for terminal utility functions."""
    
    @patch('os.system')
    @patch('platform.system')
    def test_clear_screen_windows(self, mock_platform, mock_system):
        """Test that clear_screen uses 'cls' command on Windows."""
        # Mock platform.system() to return "Windows"
        mock_platform.return_value = "Windows"
        
        # Call the function
        clear_screen()
        
        # Verify the correct command was used
        mock_system.assert_called_once_with('cls')
    
    @patch('os.system')
    @patch('platform.system')
    def test_clear_screen_linux(self, mock_platform, mock_system):
        """Test that clear_screen uses 'clear' command on Linux."""
        # Mock platform.system() to return "Linux"
        mock_platform.return_value = "Linux"
        
        # Call the function
        clear_screen()
        
        # Verify the correct command was used
        mock_system.assert_called_once_with('clear')
    
    @patch('os.system')
    @patch('platform.system')
    def test_clear_screen_darwin(self, mock_platform, mock_system):
        """Test that clear_screen uses 'clear' command on macOS."""
        # Mock platform.system() to return "Darwin" (macOS)
        mock_platform.return_value = "Darwin"
        
        # Call the function
        clear_screen()
        
        # Verify the correct command was used
        mock_system.assert_called_once_with('clear')
