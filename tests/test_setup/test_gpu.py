"""
Tests for setup/gpu.py functions.
"""
import os
import re
import platform
import subprocess
from unittest.mock import patch, MagicMock, call

from setup.gpu import check_cuda_version, install_nvidia_drivers, install_cuda_toolkit

class TestCheckCudaVersion:
    """Test suite for check_cuda_version function."""
    
    @patch('subprocess.run')
    def test_nvidia_smi_detection(self, mock_run):
        """Test CUDA version detection using nvidia-smi."""
        # Setup mock nvidia-smi response
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = """
        Wed Jan 26 10:20:55 2023
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.4     |
        |-------------------------------+----------------------+----------------------+
        """
        mock_run.return_value = mock_process
        
        # Call the function
        result = check_cuda_version()
        
        # Verify results
        assert result == "12.4"
        mock_run.assert_called_once_with(["nvidia-smi"], capture_output=True, text=True, check=False)
    
    @patch('subprocess.run')
    def test_nvidia_smi_not_found(self, mock_run):
        """Test when nvidia-smi command fails."""
        # Setup mock for nvidia-smi failure
        mock_process = MagicMock()
        mock_process.returncode = 1  # Command failed
        mock_run.return_value = mock_process
        
        with patch('platform.system', return_value='Linux'):
            # Additional patches needed for Linux checks
            with patch('os.environ.get', return_value=None), \
                 patch('os.path.islink', return_value=False), \
                 patch('os.path.isdir', return_value=False):
                
                # Call the function
                result = check_cuda_version()
        
        # Verify results
        assert result is None
        mock_run.assert_called_once_with(["nvidia-smi"], capture_output=True, text=True, check=False)
    
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('builtins.__import__')
    def test_windows_registry_detection(self, mock_import, mock_system, mock_run):
        """Test CUDA version detection from Windows registry."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_run.return_value = MagicMock(returncode=1)  # nvidia-smi fails
        
        # Create a mock winreg module
        mock_winreg = MagicMock()
        mock_key = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key
        mock_winreg.QueryValueEx.return_value = ("11.8", 1)
        mock_winreg.HKEY_LOCAL_MACHINE = 1
        
        # Make __import__ return our mock winreg when requested
        def mock_import_function(name, *args, **kwargs):
            if name == 'winreg':
                return mock_winreg
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = mock_import_function
        
        # Call the function
        result = check_cuda_version()
        
        # Verify results
        assert result == "11.8"
        mock_system.assert_called_once()
        # Check that import was called with 'winreg' as the first argument
        assert any(args[0] == 'winreg' for args, _ in mock_import.call_args_list)
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_windows_path_detection(self, mock_system, mock_run):
        """Test CUDA version detection from Windows filesystem paths."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_run.return_value = MagicMock(returncode=1)  # nvidia-smi fails
        
        # Important: We need to provide a mock winreg that allows the import to succeed
        # but have the OpenKey call fail, so the function continues to filesystem checks
        mock_winreg = MagicMock()
        mock_winreg.OpenKey.side_effect = FileNotFoundError("Registry key not found")
        mock_winreg.HKEY_LOCAL_MACHINE = 1
        
        cuda_base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        home_path = r"C:\Users\testuser"
        secondary_cuda_path = os.path.join(home_path, "NVIDIA GPU Computing Toolkit", "CUDA")
        
        print("\n============= TEST DEBUG INFO =============")
        print(f"Primary CUDA Path: {cuda_base_path}")
        print(f"Home Path: {home_path}")
        print(f"Secondary CUDA Path: {secondary_cuda_path}")
        print("==========================================\n")
        
        with patch('builtins.__import__') as mock_import, \
             patch('os.path.exists') as mock_exists, \
             patch('os.listdir') as mock_listdir:
            
            # Simply return the mock winreg module
            def import_mock(name, *args, **kwargs):
                print(f"Import attempt: {name}")
                if name == 'winreg':
                    print("Returning mock winreg")
                    return mock_winreg
                return __import__(name, *args, **kwargs)
            mock_import.side_effect = import_mock
            
            # Make os.path.exists return True for the CUDA path
            def exists_mock(path):
                result = path == cuda_base_path
                print(f"Checking if exists: {path} -> {result}")
                return result
            mock_exists.side_effect = exists_mock
            
            # Return the version directories
            def listdir_mock(path):
                if path == cuda_base_path:
                    # This is the key fix: return directories that start with 'v'
                    result = ["v11.8", "v10.2", "somefile.txt"]
                    print(f"Listing directory: {path} -> {result}")
                    return result
                return []
            mock_listdir.side_effect = listdir_mock
            
            # The important part: patch isdir to return True for versions
            with patch('os.path.isdir', return_value=True), \
                 patch('os.path.expanduser', return_value=home_path), \
                 patch('os.path.join', side_effect=os.path.join):
                
                # Call the function with debug enabled
                result = check_cuda_version(debug=True)
                
        # Verify results
        assert result == "11.8", f"Expected '11.8' but got {result}"
        mock_system.assert_called_once()
        assert mock_exists.called
        assert mock_listdir.called
    
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('os.environ')
    @patch('os.path.islink')
    @patch('os.readlink')
    def test_linux_env_var_detection(self, mock_readlink, mock_islink, mock_environ, mock_system, mock_run):
        """Test CUDA version detection from Linux environment variables."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=1)  # nvidia-smi fails
        
        # Mock Linux environment variable
        mock_environ.__contains__.return_value = True  # CUDA_PATH exists
        mock_environ.__getitem__.return_value = "/usr/local/cuda-11.2"
        mock_islink.return_value = False
        
        # Call the function
        result = check_cuda_version()
        
        # Verify results
        assert result == "11.2"
        # Function calls platform.system() twice:
        # - Once to check if it's Windows (returns False)
        # - Once to check if it's Linux (returns True)
        assert mock_system.call_count == 2
        mock_environ.__contains__.assert_called_with("CUDA_PATH")
        mock_environ.__getitem__.assert_called_with("CUDA_PATH")
    
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('os.environ')
    @patch('os.path.islink')
    @patch('os.readlink')
    def test_linux_symlink_detection(self, mock_readlink, mock_islink, mock_environ, mock_system, mock_run):
        """Test CUDA version detection from Linux symbolic links."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=1)  # nvidia-smi fails
        
        # Mock Linux environment variable check (fails)
        mock_environ.__contains__.return_value = False
        
        # Mock Linux symlink check
        mock_islink.return_value = True
        mock_readlink.return_value = "/usr/local/cuda-11.8"
        
        # Call the function
        result = check_cuda_version()
        
        # Verify results
        assert result == "11.8"
        # Function calls platform.system() twice:
        # - Once to check if it's Windows (returns False)
        # - Once to check if it's Linux (returns True)
        assert mock_system.call_count == 2
        assert mock_islink.called
        mock_readlink.assert_called_once()
    
    @patch('subprocess.run')
    @patch('platform.system')
    def test_no_cuda_detected(self, mock_system, mock_run):
        """Test when no CUDA is detected on the system."""
        # Setup mocks
        mock_system.return_value = "Linux"
        
        # nvidia-smi fails
        mock_run.return_value = MagicMock(returncode=1)
        
        # Additional patches to make sure all detection methods fail
        with patch('os.environ.get', return_value=None), \
             patch('os.path.islink', return_value=False), \
             patch('os.path.isdir', return_value=False), \
             patch('os.path.exists', return_value=False):
            
            # Call the function
            result = check_cuda_version()
        
        # Verify results
        assert result is None
        # Function calls platform.system() twice:
        # - Once to check if it's Windows (returns False)
        # - Once to check if it's Linux (returns True)
        assert mock_system.call_count == 2
        mock_run.assert_called_once()


class TestInstallNvidiaDrivers:
    """Test suite for install_nvidia_drivers function."""
    
    @patch('platform.system')
    @patch('setup.gpu.print_warning')
    def test_windows_driver_guidance(self, mock_print_warning, mock_system):
        """Test NVIDIA driver installation guidance on Windows."""
        # Setup mock
        mock_system.return_value = "Windows"
        
        # Call the function
        install_nvidia_drivers()
        
        # Verify results
        # For Windows, platform.system() is called once due to short-circuit evaluation
        # (if block passes, so elif is never evaluated)
        assert mock_system.call_count == 1
        mock_print_warning.assert_any_call("Downloading and installing NVIDIA drivers for Windows...")
        # Check that URL is included in one of the print statements
        any_has_url = False
        for call_args in mock_print_warning.call_args_list:
            if "nvidia.com/Download" in str(call_args):
                any_has_url = True
                break
        assert any_has_url, "URL for driver download not found in print statements"
    
    @patch('platform.system')
    @patch('subprocess.run')
    @patch('setup.gpu.print_warning')
    @patch('setup.gpu.print_success')
    def test_linux_driver_installation(self, mock_print_success, mock_print_warning, mock_run, mock_system):
        """Test NVIDIA driver installation on Linux."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        install_nvidia_drivers()
        
        # Verify results
        # For Linux, platform.system() is called twice:
        # First for 'if platform.system() == "Windows"' (returns False)
        # Then for 'elif platform.system() == "Linux"' (returns True)
        assert mock_system.call_count == 2
        assert mock_run.call_count == 2  # apt update + driver install
        mock_print_warning.assert_any_call("Installing NVIDIA drivers for Linux...")
        mock_print_success.assert_called_with("NVIDIA drivers installed successfully")
    
    @patch('platform.system')
    @patch('subprocess.run')
    @patch('setup.gpu.print_error')
    def test_linux_driver_installation_error(self, mock_print_error, mock_run, mock_system):
        """Test error handling during Linux driver installation."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.side_effect = subprocess.CalledProcessError(1, "apt-get")
        
        # Call the function
        install_nvidia_drivers()
        
        # Verify results
        # Function calls platform.system() twice - once to check Windows, once to check Linux
        assert mock_system.call_count == 2
        assert mock_run.call_count == 1  # First subprocess call fails
        mock_print_error.assert_called_with("Failed to install NVIDIA drivers")


class TestInstallCudaToolkit:
    """Test suite for install_cuda_toolkit function."""
    
    @patch('platform.system')
    @patch('setup.gpu.print_warning')
    def test_windows_toolkit_guidance(self, mock_print_warning, mock_system):
        """Test CUDA toolkit installation guidance on Windows."""
        # Setup mock
        mock_system.return_value = "Windows"
        
        # Call the function
        install_cuda_toolkit()
        
        # Verify results
        # When platform.system() returns "Windows", it's only called once
        # because it doesn't need to check for Linux in the elif branch
        assert mock_system.call_count == 1
        mock_print_warning.assert_any_call("Downloading and installing CUDA Toolkit for Windows...")
        # Check that URL is included in one of the print statements
        any_has_url = False
        for call_args in mock_print_warning.call_args_list:
            if "developer.nvidia.com/cuda-downloads" in str(call_args):
                any_has_url = True
                break
        assert any_has_url, "URL for CUDA toolkit download not found in print statements"
    
    @patch('platform.system')
    @patch('subprocess.run')
    @patch('setup.gpu.print_warning')
    @patch('setup.gpu.print_success')
    def test_linux_toolkit_installation(self, mock_print_success, mock_print_warning, mock_run, mock_system):
        """Test CUDA toolkit installation on Linux."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        install_cuda_toolkit()
        
        # Verify results
        # Function calls platform.system() twice - once to check Windows, once to check Linux
        assert mock_system.call_count == 2
        assert mock_run.call_count == 2  # apt update + toolkit install
        mock_print_warning.assert_any_call("Installing CUDA Toolkit for Linux...")
        mock_print_success.assert_called_with("CUDA Toolkit installed successfully")
    
    @patch('platform.system')
    @patch('subprocess.run')
    @patch('setup.gpu.print_error')
    def test_linux_toolkit_installation_error(self, mock_print_error, mock_run, mock_system):
        """Test error handling during Linux toolkit installation."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_run.side_effect = subprocess.CalledProcessError(1, "apt-get")
        
        # Call the function
        install_cuda_toolkit()
        
        # Verify results
        # Function calls platform.system() twice - once to check Windows, once to check Linux
        assert mock_system.call_count == 2
        assert mock_run.call_count == 1  # First subprocess call fails
        mock_print_error.assert_called_with("Failed to install CUDA Toolkit")
    
    @patch('platform.system')
    @patch('setup.gpu.print_warning')
    def test_unsupported_platform(self, mock_print_warning, mock_system):
        """Test graceful handling of unsupported platforms."""
        # Setup mock for unsupported OS
        mock_system.return_value = "Darwin"  # macOS
        
        # Call the function
        install_cuda_toolkit()
        
        # Verify results
        # The function calls platform.system() twice - once to check for Windows, once for Linux
        assert mock_system.call_count == 2
        mock_print_warning.assert_called_with("CUDA Toolkit installation is not supported on this platform.")
