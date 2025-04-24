"""
Tests for setup/verification.py functions.
"""
import subprocess
from unittest.mock import patch, MagicMock

from setup.verification import verify_installations

class TestVerification:
    """Test suite for verification module functions."""
    
    @patch('subprocess.run')
    def test_all_dependencies_with_gpu(self, mock_run):
        """Test verification with all dependencies installed and GPU available."""
        # Based on the debug output, we need to properly mock each subprocess call
        # in the exact order they're made in the verification function
        mock_responses = [
            # numpy import check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            # numpy version
            MagicMock(returncode=0, stdout="1.24.3", stderr=""),
            
            # cv2 import check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            # cv2 version 
            MagicMock(returncode=0, stdout="4.8.0", stderr=""),
            
            # torch import check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            # torch version check (using getattr)
            MagicMock(returncode=0, stdout="2.0.1", stderr=""),
            # ADDITIONAL torch version check (direct __version__)
            MagicMock(returncode=0, stdout="2.0.1+cu118", stderr=""),
            
            # easyocr import check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            # easyocr version check
            MagicMock(returncode=0, stdout="1.7.0", stderr=""),
            
            # GPU availability check
            MagicMock(returncode=0, stdout="True", stderr=""),
            # GPU device name
            MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080", stderr=""),
            # CUDA version
            MagicMock(returncode=0, stdout="11.8", stderr="")
        ]
        
        mock_run.side_effect = mock_responses
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # For debugging, get the actual calls made
        calls = mock_run.call_args_list
        commands = [call[0][0][2] for call in calls]
        
        # Verify results
        assert success is True, f"Expected success=True but got False. Commands: {commands}"
        assert gpu_available is True
        assert mock_run.call_count == 12
    
    @patch('subprocess.run')
    def test_all_dependencies_without_gpu(self, mock_run):
        """Test verification with all dependencies installed but no GPU."""
        # Create mock responses
        mock_responses = [
            # numpy import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.24.3", stderr=""),
            
            # cv2 import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="4.8.0", stderr=""),
            
            # torch import, version, and special torch.__version__ check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1+cpu", stderr=""),
            
            # easyocr import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.7.0", stderr=""),
            
            # GPU availability check - returns "False" for CPU-only
            MagicMock(returncode=0, stdout="False", stderr=""),
            # Additional CUDA info check
            MagicMock(
                returncode=0, 
                stdout="CUDA Available: False, PyTorch Version: 2.0.1+cpu, CUDA Version: Not available", 
                stderr=""
            )
        ]
        
        # Assign the mock responses
        mock_run.side_effect = mock_responses
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # Verify results
        assert success is True
        assert gpu_available is False
        assert mock_run.call_count == 11
    
    @patch('subprocess.run')
    def test_missing_dependency(self, mock_run):
        """Test verification with a missing dependency."""
        # Create mock responses with one failed import
        mock_responses = [
            # numpy import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.24.3", stderr=""),
            
            # cv2 import FAILS
            MagicMock(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'cv2'"),
            
            # torch import, version, and special torch.__version__ check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1+cu118", stderr=""),
            
            # easyocr import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.7.0", stderr=""),
            
            # GPU availability and CUDA checks
            MagicMock(returncode=0, stdout="True", stderr=""),
            MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080", stderr=""),
            MagicMock(returncode=0, stdout="11.8", stderr="")
        ]
        
        # Assign the mock responses
        mock_run.side_effect = mock_responses
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # Verify results
        assert success is False
        assert gpu_available is True
        # Updated call count - the verification code still makes same number of calls even when one import fails
        assert mock_run.call_count == 11
    
    @patch('subprocess.run')
    def test_easyocr_missing(self, mock_run):
        """Test verification with EasyOCR missing."""
        # Create mock responses with EasyOCR import failing
        mock_responses = [
            # numpy import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.24.3", stderr=""),
            
            # cv2 import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="4.8.0", stderr=""),
            
            # torch import, version, and special torch.__version__ check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1+cu118", stderr=""),
            
            # easyocr import FAILS
            MagicMock(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'easyocr'"),
            
            # GPU availability and CUDA checks
            MagicMock(returncode=0, stdout="True", stderr=""),
            MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080", stderr=""),
            MagicMock(returncode=0, stdout="11.8", stderr="")
        ]
        
        # Assign the mock responses
        mock_run.side_effect = mock_responses
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # Verify results
        assert success is False
        assert gpu_available is True
        # The verification code still makes the same number of subprocess calls
        assert mock_run.call_count == 11
    
    @patch('subprocess.run')
    def test_exception_during_verification(self, mock_run):
        """Test handling of exceptions during verification."""
        # Make subprocess.run raise an exception
        mock_run.side_effect = Exception("Simulated error")
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # Verify results
        assert success is False
        assert gpu_available is False
        assert mock_run.call_count >= 1
    
    @patch('subprocess.run')
    def test_gpu_check_exception(self, mock_run):
        """Test handling of exception during GPU check."""
        # Create mock responses with successful imports but GPU check fails
        mock_responses = [
            # numpy import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.24.3", stderr=""),
            
            # cv2 import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="4.8.0", stderr=""),
            
            # torch import, version, and special torch.__version__ check
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1", stderr=""),
            MagicMock(returncode=0, stdout="2.0.1+cu118", stderr=""),
            
            # easyocr import and version
            MagicMock(returncode=0, stdout="Success", stderr=""),
            MagicMock(returncode=0, stdout="1.7.0", stderr=""),
            
            # GPU check raises exception
            Exception("GPU check error")
        ]
        
        # Assign the mock responses
        mock_run.side_effect = mock_responses
        
        # Call the function
        success, gpu_available = verify_installations("python")
        
        # Verify results
        assert success is True  # Dependencies are still considered installed
        assert gpu_available is False  # GPU not available due to error
        assert mock_run.call_count == 10
    
    @patch('subprocess.run')
    @patch('setup.verification.print_debug')
    def test_debug_mode_enabled(self, mock_print_debug, mock_run):
        """Test verification with debug mode enabled."""
        # Create enough successful mock responses
        mock_responses = []
        
        # For each dependency (4 dependencies with import, version, plus extra torch check)
        # plus 3 GPU-related checks
        for _ in range(12):
            mock_responses.append(MagicMock(returncode=0, stdout="Success", stderr=""))
            
        mock_run.side_effect = mock_responses
        
        # Call the function with debug=True
        verify_installations("python", debug=True)
        
        # Verify print_debug was called with debug=True
        assert mock_print_debug.called
        assert any(call[0][1] == True for call in mock_print_debug.call_args_list)
