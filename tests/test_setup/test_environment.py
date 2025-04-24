"""
Tests for setup/environment.py functions.
"""
import os
import sys
import platform
import shutil
import subprocess
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from setup.environment import (
    try_force_remove_venv,
    create_virtual_environment,
    create_required_directories
)

class TestTryForceRemoveVenv:
    """Test suite for try_force_remove_venv function."""
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_force_remove_windows(self, mock_system, mock_run):
        """Test force removal of venv on Windows."""
        # Setup mocks
        mock_system.return_value = "Windows"
        
        # Call the function
        try_force_remove_venv("venv", debug=False)
        
        # Verify results
        mock_system.assert_called_once()
        mock_run.assert_called_once()
        # On Windows, it should use cmd with rmdir
        args, kwargs = mock_run.call_args
        assert args[0][0] == "cmd"
        assert args[0][1] == "/c"
        assert "rmdir /s /q venv" in args[0][2]
        assert kwargs.get('check') is False
        assert kwargs.get('capture_output') is True
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_force_remove_unix(self, mock_system, mock_run):
        """Test force removal of venv on Unix systems."""
        # Setup mocks
        mock_system.return_value = "Linux"
        
        # Call the function
        try_force_remove_venv("venv", debug=False)
        
        # Verify results
        mock_system.assert_called_once()
        mock_run.assert_called_once()
        # On Unix, it should use rm -rf directly
        args, kwargs = mock_run.call_args
        assert args[0][0] == "rm"
        assert args[0][1] == "-rf"
        assert args[0][2] == "venv"
        assert kwargs.get('check') is False
        assert kwargs.get('capture_output') is True
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_force_remove_with_debug(self, mock_system, mock_run):
        """Test force removal with debug enabled."""
        # Setup mocks
        mock_system.return_value = "Windows"
        
        # Call the function with debug
        try_force_remove_venv("venv", debug=True)
        
        # Verify results
        mock_run.assert_called_once()
        # With debug=True, capture_output should not be present
        args, kwargs = mock_run.call_args
        assert kwargs.get('check') is False
        assert 'capture_output' not in kwargs
    
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('setup.environment.print_warning')
    def test_force_remove_exception(self, mock_print_warning, mock_system, mock_run):
        """Test handling of exceptions during force removal."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_run.side_effect = Exception("Simulated error")
        
        # Call the function
        try_force_remove_venv("venv", debug=True)
        
        # Verify results
        mock_print_warning.assert_called()
        # Check that the mock was called with the correct message
        # The function prints the main error message first, then debug details
        assert mock_print_warning.call_count >= 1
        # Check if any call contains our expected message
        called_with_expected_message = False
        for call_args in mock_print_warning.call_args_list:
            if "Force removal method also failed" in call_args[0][0]:
                called_with_expected_message = True
                break
        assert called_with_expected_message, "Expected warning message not found"


class TestCreateVirtualEnvironment:
    """Test suite for create_virtual_environment function."""
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('setup.environment.print_success')
    @patch('setup.environment.print_info')
    def test_create_new_environment(self, mock_print_info, mock_print_success, mock_run, mock_exists):
        """Test creating a new virtual environment when none exists."""
        # Setup mocks
        mock_exists.return_value = False  # venv doesn't exist
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        result = create_virtual_environment()
        
        # Verify results
        assert result is True
        mock_exists.assert_called_once_with("venv")
        mock_run.assert_called_once()
        mock_print_info.assert_called_with("Creating new virtual environment...")
        mock_print_success.assert_called()
    
    @patch('os.path.exists')
    @patch('setup.environment.print_warning')
    @patch('builtins.input')
    def test_keep_existing_environment(self, mock_input, mock_print_warning, mock_exists):
        """Test keeping an existing virtual environment when user declines recreation."""
        # Setup mocks
        mock_exists.return_value = True  # venv exists
        mock_input.return_value = "n"  # User doesn't want to recreate
        
        # Call the function
        result = create_virtual_environment()
        
        # Verify results
        assert result is True
        mock_exists.assert_called_once_with("venv")
        mock_print_warning.assert_any_call("Virtual environment already exists at 'venv'")
        mock_print_warning.assert_any_call("Using existing virtual environment")
        mock_input.assert_called_once()
    
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('subprocess.run')
    @patch('setup.environment.print_warning')
    @patch('builtins.input')
    def test_recreate_environment(self, mock_input, mock_print_warning, mock_run, 
                                mock_rmtree, mock_exists):
        """Test recreating an existing virtual environment when user confirms."""
        # Setup mocks
        mock_exists.return_value = True  # venv exists
        mock_input.return_value = "y"  # User wants to recreate
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        result = create_virtual_environment()
        
        # Verify results
        assert result is True
        mock_exists.assert_called_with("venv")
        mock_print_warning.assert_any_call("Virtual environment already exists at 'venv'")
        mock_input.assert_called_once()
        mock_rmtree.assert_called_once_with("venv")
        mock_run.assert_called_once()
    
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('platform.system')
    @patch('subprocess.run')
    @patch('time.sleep')
    @patch('setup.environment.print_warning')
    @patch('setup.environment.print_error')
    @patch('builtins.input')
    def test_windows_permission_error_retry_success(self, mock_input, mock_print_error, 
                                                 mock_print_warning, mock_sleep, mock_run,
                                                 mock_system, mock_rmtree, mock_exists):
        """Test handling of permission error on Windows with successful retry."""
        # Setup mocks
        mock_exists.return_value = True  # venv exists
        mock_input.side_effect = ["y", "y"]  # User wants to recreate, then retry
        mock_system.return_value = "Windows"
        mock_rmtree.side_effect = [
            PermissionError("Access is denied"),  # First attempt fails
            None  # Second attempt succeeds
        ]
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        result = create_virtual_environment()
        
        # Verify results
        assert result is True
        assert mock_rmtree.call_count == 2
        mock_print_warning.assert_any_call("Access denied error detected. This usually happens when:")
        mock_input.assert_any_call("Try again after closing applications? (y/n): ")
        mock_sleep.assert_called_once_with(2)
        mock_run.assert_called_once()
    
    @patch('os.path.exists')
    @patch('setup.environment.print_warning')
    def test_unattended_mode_keep_default(self, mock_print_warning, mock_exists):
        """Test unattended mode with default behavior (keep existing environment)."""
        # Setup mocks
        mock_exists.return_value = True  # venv exists
        
        # Call the function in unattended mode
        result = create_virtual_environment(unattended=True)
        
        # Verify results
        assert result is True
        mock_exists.assert_called_once_with("venv")
        mock_print_warning.assert_any_call("Virtual environment already exists at 'venv'")
        mock_print_warning.assert_any_call("Unattended mode: Using existing virtual environment (default)")
        
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    @patch('subprocess.run')
    @patch('setup.environment.print_warning')
    def test_unattended_mode_recreate(self, mock_print_warning, mock_run, mock_rmtree, mock_exists):
        """Test unattended mode with recreate option."""
        # Setup mocks
        mock_exists.return_value = True  # venv exists
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function in unattended mode with recreate
        result = create_virtual_environment(unattended=True, recreate=True)
        
        # Verify results
        assert result is True
        mock_exists.assert_called_once_with("venv")
        mock_print_warning.assert_any_call("Virtual environment already exists at 'venv'")
        mock_print_warning.assert_any_call("Unattended mode: Recreating virtual environment")
        mock_rmtree.assert_called_once_with("venv")
        mock_run.assert_called_once()
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('setup.environment.print_error')
    def test_subprocess_error(self, mock_print_error, mock_run, mock_exists):
        """Test handling of subprocess error during environment creation."""
        # Setup mocks
        mock_exists.return_value = False  # venv doesn't exist, so no input prompt
        mock_run.side_effect = subprocess.CalledProcessError(1, "venv")
        
        # Call the function
        result = create_virtual_environment()
        
        # Verify results
        assert result is False
        mock_print_error.assert_any_call("Failed to create virtual environment")


class TestCreateRequiredDirectories:
    """Test suite for create_required_directories function."""
    
    @patch('pathlib.Path.mkdir')
    @patch('setup.environment.print_success')
    def test_create_directories_success(self, mock_print_success, mock_mkdir):
        """Test successful creation of required directories."""
        # Call the function
        create_required_directories()
        
        # Verify results
        assert mock_mkdir.call_count == 4  # 4 directories should be created
        mock_mkdir.assert_has_calls([
            call(exist_ok=True),  # flight_recordings
            call(exist_ok=True),  # results
            call(exist_ok=True),  # .tmp
            call(exist_ok=True),  # logs
        ])
        assert mock_print_success.call_count == 4
    
    @patch('pathlib.Path.mkdir')
    @patch('setup.environment.print_success')
    @patch('setup.environment.print_error')
    def test_create_directories_error(self, mock_print_error, mock_print_success, mock_mkdir):
        """Test handling of errors during directory creation."""
        # Setup mock to raise exception for the second directory
        mock_mkdir.side_effect = [
            None,  # First directory succeeds
            Exception("Permission denied"),  # Second directory fails
            None,  # Third directory succeeds
            None   # Fourth directory succeeds
        ]
        
        # Call the function
        create_required_directories()
        
        # Verify results
        assert mock_mkdir.call_count == 4
        assert mock_print_success.call_count == 3  # Only 3 successful directories
        assert mock_print_error.call_count == 1    # 1 error
        mock_print_error.assert_called_with("Failed to create directory 'results': Permission denied")
