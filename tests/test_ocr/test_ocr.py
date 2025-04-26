import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os
import torch
import threading

from ocr.ocr import (
    get_reader,
    extract_values_from_roi,
    extract_single_value,
    extract_time
)

@pytest.fixture
def mock_easyocr_reader():
    """Mock the EasyOCR reader."""
    with patch('ocr.ocr.easyocr.Reader') as mock_reader:
        # Configure the mock reader instance
        reader_instance = MagicMock()
        mock_reader.return_value = reader_instance
        
        # Configure readtext to return different results based on input
        def mock_readtext(img, detail=0, allowlist=None):
            if img.shape == (25, 83, 3):  # Speed ROI
                return ["100"]
            elif img.shape == (25, 50, 3):  # Altitude ROI
                return ["5000"]
            elif img.shape == (44, 197, 3):  # Time ROI
                return ["+01:30:00"]
            else:
                return []
        
        reader_instance.readtext.side_effect = mock_readtext
        yield mock_reader

@pytest.fixture
def test_rois():
    """Create test ROIs with different shapes for different data types."""    
    speed_roi = np.zeros((25, 83, 3), dtype=np.uint8)
    altitude_roi = np.zeros((25, 50, 3), dtype=np.uint8)
    time_roi = np.zeros((44, 197, 3), dtype=np.uint8)
    empty_roi = np.zeros((0, 0, 3), dtype=np.uint8)
    return speed_roi, altitude_roi, time_roi, empty_roi


class TestGetReader:
    """Tests for get_reader function."""
    
    @patch('ocr.ocr.torch.cuda.is_available')
    @patch('ocr.ocr.easyocr.Reader')
    def test_get_reader_gpu(self, mock_reader, mock_cuda_available):
        """Test get_reader with GPU available."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Configure the mock
        reader_instance = MagicMock()
        mock_reader.return_value = reader_instance
        
        # Create a fresh thread local storage for testing
        test_thread_local = threading.local()
        
        # Use patch to replace the _thread_local in the ocr module with our test one
        with patch('ocr.ocr._thread_local', test_thread_local), \
             patch('ocr.ocr.torch.cuda.set_device') as mock_set_device, \
             patch('ocr.ocr.torch.cuda.get_device_name') as mock_get_device_name, \
             patch('ocr.ocr.torch.cuda.get_device_properties') as mock_get_device_props:
            
            # Configure mocks for successful GPU setup
            mock_get_device_name.return_value = "NVIDIA Test GPU"
            mock_device_props = MagicMock()
            mock_device_props.total_memory = 8 * 1024**3  # 8GB of memory
            mock_get_device_props.return_value = mock_device_props
            
            # Call the function
            result = get_reader()
            
            # Verify reader was created with GPU=True
            mock_reader.assert_called_once_with(['en'], gpu=True, verbose=False)
            assert result == reader_instance
    
    @patch('ocr.ocr.torch.cuda.is_available')
    @patch('ocr.ocr.easyocr.Reader')
    def test_get_reader_cpu(self, mock_reader, mock_cuda_available):
        """Test get_reader with GPU not available."""
        # Mock CUDA unavailability
        mock_cuda_available.return_value = False
        
        # Configure the mock
        reader_instance = MagicMock()
        mock_reader.return_value = reader_instance
        
        # Create a fresh thread local storage for testing
        test_thread_local = threading.local()
            
        # Use patch to replace the _thread_local in the ocr module with our test one
        with patch('ocr.ocr._thread_local', test_thread_local):
            # Call the function
            result = get_reader()
            
            # Verify reader was created with GPU=False
            mock_reader.assert_called_once_with(['en'], gpu=False, verbose=False)
            assert result == reader_instance
    
    @patch('ocr.ocr.torch.cuda.is_available')
    @patch('ocr.ocr.easyocr.Reader')
    def test_get_reader_cuda_error(self, mock_reader, mock_cuda_available):
        """Test get_reader with CUDA error."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Configure the reader mock
        reader_instance = MagicMock()
        mock_reader.return_value = reader_instance
        
        # Create a fresh thread local storage for testing
        test_thread_local = threading.local()
        
        # Use patch to replace the _thread_local in the ocr module
        with patch('ocr.ocr._thread_local', test_thread_local):
            # Mock CUDA functions - set_device raises exception
            with patch('ocr.ocr.torch.cuda.set_device') as mock_set_device, \
                 patch('ocr.ocr.logger') as mock_logger:
                
                # Configure set_device to raise an exception
                mock_set_device.side_effect = RuntimeError("CUDA error")
                
                # Call the function
                result = get_reader()
                
                # Verify error was logged
                mock_logger.error.assert_called_once_with(f"Process {os.getpid()}: Error setting CUDA device: CUDA error")
                
                # Verify reader was created with GPU=False due to error
                mock_reader.assert_called_once_with(['en'], gpu=False, verbose=False)
                assert result == reader_instance
    
    @patch('ocr.ocr.torch.cuda.is_available')
    @patch('ocr.ocr.easyocr.Reader')
    def test_get_reader_reuse(self, mock_reader, mock_cuda_available):
        """Test get_reader reuses existing reader."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Configure the mock
        reader_instance = MagicMock()
        mock_reader.return_value = reader_instance
        
        # Create a fresh thread local storage for testing
        test_thread_local = threading.local()
        
        # Use patch to replace the _thread_local in the ocr module
        with patch('ocr.ocr._thread_local', test_thread_local), \
             patch('ocr.ocr.torch.cuda.set_device') as mock_set_device, \
             patch('ocr.ocr.torch.cuda.get_device_name') as mock_get_device_name, \
             patch('ocr.ocr.torch.cuda.get_device_properties') as mock_get_device_props:
            
            # Configure mocks for successful GPU setup
            mock_get_device_name.return_value = "NVIDIA Test GPU"
            mock_device_props = MagicMock()
            mock_device_props.total_memory = 8 * 1024**3  # 8GB of memory
            mock_get_device_props.return_value = mock_device_props
            
            # Call the function twice
            result1 = get_reader()
            result2 = get_reader()
            
            # Verify reader was created only once
            mock_reader.assert_called_once_with(['en'], gpu=True, verbose=False)
            assert result1 == result2 == reader_instance


class TestExtractValuesFromROI:
    """Tests for extract_values_from_roi function."""    
    
    @patch('ocr.ocr.get_reader')
    def test_extract_speed(self, mock_get_reader, test_rois):
        """Test extracting speed value."""        
        speed_roi, _, _, _ = test_rois
        
        # Setup mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["100"]
        mock_get_reader.return_value = mock_reader
        
        # Call the function
        result = extract_values_from_roi(speed_roi, mode="speed")
        
        # Verify results
        assert result == {"value": 100}
        mock_reader.readtext.assert_called_once_with(speed_roi, detail=0, allowlist='0123456789')
    
    @patch('ocr.ocr.get_reader')
    def test_extract_altitude(self, mock_get_reader, test_rois):
        """Test extracting altitude value."""        
        _, altitude_roi, _, _ = test_rois
        
        # Setup mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["5000"]
        mock_get_reader.return_value = mock_reader
        
        # Call the function
        result = extract_values_from_roi(altitude_roi, mode="altitude")
        
        # Verify results
        assert result == {"value": 5000}
        mock_reader.readtext.assert_called_once_with(altitude_roi, detail=0, allowlist='0123456789')
    
    @patch('ocr.ocr.get_reader')
    def test_extract_time(self, mock_get_reader, test_rois):
        """Test extracting time value."""        
        _, _, time_roi, _ = test_rois
        
        # Setup mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["+01:30:00"]
        mock_get_reader.return_value = mock_reader
        
        # Call the function
        result = extract_values_from_roi(time_roi, mode="time")
        
        # Verify results
        assert result == {"sign": "+", "hours": 1, "minutes": 30, "seconds": 0}
        mock_reader.readtext.assert_called_once_with(time_roi, detail=0, allowlist='0123456789T+-:')
    
    @patch('ocr.ocr.get_reader')
    def test_invalid_roi(self, mock_get_reader, test_rois):
        """Test handling of invalid ROI."""        
        _, _, _, empty_roi = test_rois
        
        # Call the function with empty ROI
        with patch('ocr.ocr.logger') as mock_logger:
            result = extract_values_from_roi(empty_roi, mode="speed", debug=True)
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            
            # Verify empty result
            assert result == {}
            
            # Verify reader was not called
            mock_get_reader.assert_called_once()
            mock_get_reader.return_value.readtext.assert_not_called()
    
    @patch('ocr.ocr.get_reader')
    def test_unknown_mode(self, mock_get_reader, test_rois):
        """Test handling of unknown mode."""        
        speed_roi, _, _, _ = test_rois
        
        # Setup mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["100"]
        mock_get_reader.return_value = mock_reader
        
        # Call the function with unknown mode
        with patch('ocr.ocr.logger') as mock_logger:
            result = extract_values_from_roi(speed_roi, mode="unknown", debug=True)
            
            # Verify debug was logged
            mock_logger.debug.assert_any_call("Unknown mode: unknown")
            
            # Verify empty result
            assert result == {}
    
    @patch('ocr.ocr.get_reader')
    def test_cuda_out_of_memory(self, mock_get_reader, test_rois):
        """Test handling of CUDA out of memory error."""        
        speed_roi, _, _, _ = test_rois
        
        # Setup mock reader to raise CUDA OOM error then succeed
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = [
            RuntimeError("CUDA out of memory"),  # First call fails
            ["100"]  # Second call succeeds
        ]
        mock_get_reader.return_value = mock_reader
        
        # Call the function
        with patch('ocr.ocr.torch.cuda.empty_cache') as mock_empty_cache, \
             patch('ocr.ocr.easyocr.Reader') as mock_reader_class, \
             patch('ocr.ocr.logger') as mock_logger:
            
            # Set up new CPU reader
            cpu_reader = MagicMock()
            cpu_reader.readtext.return_value = ["100"]
            mock_reader_class.return_value = cpu_reader
            
            # Call the function
            result = extract_values_from_roi(speed_roi, mode="speed")
            
            # Verify CUDA memory was cleared
            mock_empty_cache.assert_called_once()
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            
            # Verify new reader was created with GPU=False
            mock_reader_class.assert_called_once_with(['en'], gpu=False, verbose=False)
            
            # Verify result from CPU fallback
            assert result == {"value": 100}


class TestExtractSingleValue:
    """Tests for extract_single_value function."""    
    
    def test_valid_input(self):
        """Test extraction with valid input."""        
        assert extract_single_value("100") == 100
        assert extract_single_value("speed is 100") == 100
        assert extract_single_value("100 km/h") == 100
    
    def test_multiple_numbers(self):
        """Test extraction with multiple numbers - should take first one."""        
        assert extract_single_value("100 200 300") == 100
    
    def test_no_numbers(self):
        """Test extraction with no numbers."""        
        with patch('ocr.ocr.logger') as mock_logger:
            assert extract_single_value("no numbers here") is None
            mock_logger.debug.assert_called_once()


class TestExtractTime:
    """Tests for extract_time function."""    
    
    def test_valid_time(self):
        """Test extraction with valid time format."""        
        assert extract_time("+01:30:00") == {"sign": "+", "hours": 1, "minutes": 30, "seconds": 0}
        assert extract_time("-00:05:15") == {"sign": "-", "hours": 0, "minutes": 5, "seconds": 15}
    
    def test_time_in_text(self):
        """Test extraction with time embedded in text."""        
        assert extract_time("Time is +01:30:00 now") == {"sign": "+", "hours": 1, "minutes": 30, "seconds": 0}
    
    def test_invalid_format(self):
        """Test extraction with invalid time format."""        
        with patch('ocr.ocr.logger') as mock_logger:
            assert extract_time("01:30:00") is None  # Missing sign
            assert extract_time("+1:30:0") is None  # Wrong format
            assert extract_time("no time here") is None
            assert mock_logger.debug.call_count == 3
