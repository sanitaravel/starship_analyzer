"""
Tests for setup/dependencies.py functions.
"""
import os
import platform
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call

from setup.dependencies import install_torch_with_cuda, install_dependencies

class TestInstallTorchWithCuda:
    """Test suite for install_torch_with_cuda function."""
    
    @patch('subprocess.run')
    def test_install_with_cuda_support(self, mock_run):
        """Test PyTorch installation with CUDA support."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "12.4"
        
        # Mock subprocess.run to return success and CUDA availability
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "CUDA Available:True, Version:2.0.1+cu124"
        mock_run.return_value = mock_process
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Verify results
        assert result is True
        # Check that correct CUDA URL was used
        assert mock_run.call_count >= 2
        install_call = mock_run.call_args_list[0]
        assert install_call[0][0] == [pip_path, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu124']
    
    @patch('subprocess.run')
    def test_install_with_older_cuda_version(self, mock_run):
        """Test PyTorch installation with older CUDA version."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "11.8"
        
        # Mock subprocess.run to return success and CUDA availability
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "CUDA Available:True, Version:2.0.1+cu118"
        mock_run.return_value = mock_process
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Verify results
        assert result is True
        # Check that correct CUDA URL was used for older version
        assert mock_run.call_count >= 2
        install_call = mock_run.call_args_list[0]
        assert install_call[0][0] == [pip_path, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu118']
    
    @patch('subprocess.run')
    def test_install_newest_cuda_version(self, mock_run):
        """Test PyTorch installation with newest CUDA version."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "12.6"
        
        # Mock subprocess.run to return success and CUDA availability
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "CUDA Available:True, Version:2.0.1+cu126"
        mock_run.return_value = mock_process
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Verify results
        assert result is True
        # Check that correct CUDA URL was used for newest version
        assert mock_run.call_count >= 2
        install_call = mock_run.call_args_list[0]
        assert install_call[0][0] == [pip_path, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu126']
    
    @patch('subprocess.run')
    def test_install_with_cpu_only(self, mock_run):
        """Test PyTorch installation with CPU-only support."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = None  # No CUDA
        
        # Mock subprocess.run to return success
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Verify results
        assert result is True
        # Check that CPU-only URL was used
        assert mock_run.call_count >= 1
        install_call = mock_run.call_args_list[0]
        assert install_call[0][0] == [pip_path, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu']
    
    @patch('subprocess.run')
    def test_cuda_install_failure_fallback_to_cpu(self, mock_run):
        """Test fallback to CPU when CUDA install fails."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "12.4"
        
        # Mock CUDA install failure, then CPU install success
        def side_effect(*args, **kwargs):
            # First call (CUDA install) fails
            if args[0][5] == 'https://download.pytorch.org/whl/cu124':
                raise subprocess.CalledProcessError(1, args[0])
                
            # Second call (CPU install) succeeds
            return MagicMock(returncode=0, stdout="")
        mock_run.side_effect = side_effect
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Verify results - should still succeed with CPU fallback
        assert result is True
        assert mock_run.call_count >= 2
        cpu_install_call = mock_run.call_args_list[1]
        assert cpu_install_call[0][0] == [pip_path, 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu']
    
    @patch('subprocess.run')
    def test_cuda_available_verification_failed(self, mock_run):
        """Test when CUDA is installed but verification fails."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "12.4"
        
        # Mock successful installation but CUDA not available in verification
        def side_effect(*args, **kwargs):
            if len(args[0]) == 6 and args[0][5] == 'https://download.pytorch.org/whl/cu124':
                # PyTorch install succeeds
                return MagicMock(returncode=0)
            else:
                # But verification shows CPU-only
                return MagicMock(returncode=0, stdout="CUDA Available:False, Version:2.0.1+cpu")
        mock_run.side_effect = side_effect
        
        # Call the function
        result = install_torch_with_cuda(pip_path, cuda_version)
        
        # Despite CUDA not working, function should return True as installation completed
        assert result is True
        assert mock_run.call_count >= 2
    
    @patch('subprocess.run')
    def test_debug_mode_enabled(self, mock_run):
        """Test installation with debug mode enabled."""
        # Setup mocks
        pip_path = "venv/bin/pip"
        cuda_version = "12.4"
        mock_run.return_value = MagicMock(returncode=0, stdout="CUDA Available:True, Version:2.0.1+cu124")
        
        # Call the function with debug=True
        result = install_torch_with_cuda(pip_path, cuda_version, debug=True)
        
        # Verify that capture_output is not used in debug mode
        assert result is True
        install_call = mock_run.call_args_list[0]
        assert 'capture_output' not in install_call[1]


class TestInstallDependencies:
    """Test suite for install_dependencies function."""
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('setup.dependencies.install_torch_with_cuda')
    def test_install_dependencies_windows(self, mock_install_torch, mock_system, mock_run, mock_exists):
        """Test dependencies installation on Windows."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_exists.return_value = True  # requirements.txt exists
        mock_run.return_value = MagicMock(returncode=0)
        mock_install_torch.return_value = True
        
        # Expected pip path for Windows
        expected_pip_path = os.path.join("venv", "Scripts", "pip.exe")
        
        # Mock reading requirements.txt
        with patch('builtins.open', mock_open(read_data="""
        numpy==1.24.3
        opencv-python==4.8.0
        # Comment line
        pywin32==306
        """)):
            # Call the function
            result = install_dependencies("12.4")
        
        # Verify results
        assert result is True
        # Check correct pip path for Windows
        assert mock_run.call_count >= 3  # pip upgrade + 2 packages
        # Check PyTorch installation was called
        mock_install_torch.assert_called_once_with(expected_pip_path, "12.4", debug=False)
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('setup.dependencies.install_torch_with_cuda')
    @patch('os.path.join')
    def test_install_dependencies_linux(self, mock_join, mock_install_torch, mock_system, mock_run, mock_exists):
        """Test dependencies installation on Linux."""
        # Setup mocks
        mock_system.return_value = "Linux"
        mock_exists.return_value = True  # requirements.txt exists
        mock_run.return_value = MagicMock(returncode=0)
        mock_install_torch.return_value = True
        
        # Make os.path.join return Linux-style paths
        def linux_path_join(*args):
            return '/'.join(args)
        mock_join.side_effect = linux_path_join
        
        # Mock reading requirements.txt
        with patch('builtins.open', mock_open(read_data="""
        numpy==1.24.3
        opencv-python==4.8.0
        # Comment line
        pywin32==306  # Should be skipped on Linux
        """)):
            # Call the function
            result = install_dependencies("11.8")
        
        # Verify results
        assert result is True
        
        # Check Linux-specific installs were performed
        # Look for the sudo apt-get install command for python3-tk
        linux_install_calls = [call for call in mock_run.call_args_list 
                             if len(call[0][0]) >= 3 and call[0][0][0] == "sudo" and 
                                call[0][0][1] == "apt-get" and call[0][0][2] == "install"]
        
        assert len(linux_install_calls) > 0, "No Linux package installation calls found"
        
        # Check that python3-tk was one of the packages installed
        python_tk_installed = any("python3-tk" in str(call) for call in linux_install_calls)
        assert python_tk_installed, "python3-tk package was not installed"
        
        # Check PyTorch installation was called with Linux pip path
        mock_install_torch.assert_called_once_with("venv/bin/pip", "11.8", debug=False)
    
    @patch('os.path.exists')
    def test_missing_requirements_file(self, mock_exists):
        """Test behavior when requirements.txt is missing."""
        # Setup mock
        mock_exists.return_value = False  # requirements.txt doesn't exist
        
        # Call the function
        result = install_dependencies("12.4")
        
        # Verify result
        assert result is False
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('platform.system')
    def test_pip_upgrade_failure(self, mock_system, mock_run, mock_exists):
        """Test handling of pip upgrade failure."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_exists.return_value = True  # requirements.txt exists
        
        # Mock pip upgrade failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
        
        # Mock empty requirements file
        with patch('builtins.open', mock_open(read_data="")):
            # Call the function
            result = install_dependencies("12.4")
        
        # Verify results
        assert result is False
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('setup.dependencies.install_torch_with_cuda')
    def test_force_cpu_installation(self, mock_install_torch, mock_system, mock_run, mock_exists):
        """Test force CPU-only installation."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_exists.return_value = True  # requirements.txt exists
        mock_run.return_value = MagicMock(returncode=0)
        mock_install_torch.return_value = True
        
        # Expected pip path for Windows
        expected_pip_path = os.path.join("venv", "Scripts", "pip.exe")
        
        # Mock empty requirements file
        with patch('builtins.open', mock_open(read_data="")):
            # Call the function with force_cpu=True
            result = install_dependencies("12.4", force_cpu=True)
        
        # Verify results
        assert result is True
        # Check that PyTorch was installed with None instead of CUDA version
        mock_install_torch.assert_called_once_with(expected_pip_path, None, debug=False)
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('platform.system')
    @patch('setup.dependencies.install_torch_with_cuda')
    def test_debug_mode(self, mock_install_torch, mock_system, mock_run, mock_exists):
        """Test installation with debug mode enabled."""
        # Setup mocks
        mock_system.return_value = "Windows"
        mock_exists.return_value = True  # requirements.txt exists
        mock_run.return_value = MagicMock(returncode=0)
        mock_install_torch.return_value = True
        
        # Expected pip path for Windows
        expected_pip_path = os.path.join("venv", "Scripts", "pip.exe")
        
        # Mock empty requirements file
        with patch('builtins.open', mock_open(read_data="")):
            # Call the function with debug=True
            result = install_dependencies("12.4", debug=True)
        
        # Verify results
        assert result is True
        # Check that PyTorch was installed with debug=True
        mock_install_torch.assert_called_once_with(expected_pip_path, "12.4", debug=True)
        # Verify subprocess.run was called without capture_output
        for call_args in mock_run.call_args_list:
            assert 'capture_output' not in call_args[1]
