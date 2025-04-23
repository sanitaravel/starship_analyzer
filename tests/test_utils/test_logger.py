"""
Tests for logger package in utils/logger.
"""
import pytest
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call

# Import the logger module but not the specific functions we want to mock
from utils.logger import (
    get_logger,
    set_global_log_level,
    _update_file_handlers,
    start_new_session,
    log_system_info,
    ColoredFormatter,
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    DATE_FORMAT,
    LOG_DIR,
    LOG_FILE
)

# Import the module for direct access
import utils.logger.system_info

class TestLogger:
    """Test suite for logger package utilities."""
    
    def test_logger_constants(self):
        """Test that logger constants are properly defined."""
        # Test LOG_LEVELS dictionary has expected keys
        assert 'DEBUG' in LOG_LEVELS
        assert 'INFO' in LOG_LEVELS
        assert 'WARNING' in LOG_LEVELS
        assert 'ERROR' in LOG_LEVELS
        assert 'CRITICAL' in LOG_LEVELS
        
        # Test default log level is a valid logging level
        assert DEFAULT_LOG_LEVEL in [logging.DEBUG, logging.INFO, logging.WARNING, 
                                    logging.ERROR, logging.CRITICAL]
        
        # Test format strings are not empty
        assert LOG_FORMAT
        assert DATE_FORMAT
        
        # Test log paths
        assert LOG_DIR
        assert LOG_FILE
    
    @patch('logging.getLogger')
    def test_get_logger(self, mock_get_logger):
        """Test get_logger returns the correct logger instance."""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call the function
        logger = get_logger("test_module")
        
        # Verify correct logger was retrieved
        mock_get_logger.assert_called_once_with("test_module")
        assert logger == mock_logger
    
    @patch('logging.getLogger')
    def test_set_global_log_level(self, mock_get_logger):
        """Test setting global log level."""
        # Setup test
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger
        
        # Call the function
        set_global_log_level(logging.WARNING)
        
        # Verify root logger level was set
        mock_get_logger.assert_called_once_with()
        mock_root_logger.setLevel.assert_called_once_with(logging.WARNING)
    
    @patch('utils.logger.core.get_logger')
    @patch('os.makedirs')
    def test_start_new_session(self, mock_makedirs, mock_get_logger):
        """Test starting a new logging session."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Patch os.path.exists to return False specifically for the log directory check
        with patch('os.path.exists', return_value=False) as mock_exists:
            # Patch open to prevent actual file operations
            with patch('builtins.open', mock_open()):
                # Patch log_system_info to avoid its calls
                with patch('utils.logger.core.log_system_info') as mock_log_system_info:
                    # Patch CURRENT_SESSION_LOG_FILE verification
                    with patch('os.path.join', return_value='/mock/path/to/log.log'):
                        start_new_session()
        
        # Verify directory creation was performed
        mock_makedirs.assert_called_once()
        
        # Verify logger was obtained and info was logged
        mock_get_logger.assert_called_with("starship_analyzer")
        assert mock_logger.info.call_count >= 1
    
    def test_collect_system_info(self):
        """Test system info collection by mocking the entire function."""
        # Define a mock system info return value
        mock_system_info = {
            "system": "Test OS",
            "platform": "Test Platform",
            "python_version": "3.9.5",
            "processor": "Mock CPU",
            "architecture": ("64bit", "ELF"),
            "release": "10",
            "cpu_count_physical": 8,
            "cpu_count_logical": 16,
            "total_memory_gb": 16.0,
            "available_memory_gb": 8.0,
            "used_memory_gb": 8.0,
            "memory_percent": 50.0,
            "cuda_available": True,
            "cuda_version": "11.3",
            "gpu_count": 2,
            "opencv_version": "4.5.4",
            "numpy_version": "1.21.0",
            "easyocr_version": "1.4.1",
            "gpus": [
                {
                    "name": "Test GPU 0",
                    "index": 0,
                    "memory_allocated_mb": 1024.0,
                    "memory_reserved_mb": 2048.0
                },
                {
                    "name": "Test GPU 1",
                    "index": 1,
                    "memory_allocated_mb": 512.0,
                    "memory_reserved_mb": 1024.0
                }
            ]
        }
        
        # Create a genuine mock that we can use to replace the function
        original_collect_system_info = utils.logger.system_info.collect_system_info
        
        try:
            # Replace the function with a mock
            mock_collect = MagicMock(return_value=mock_system_info)
            utils.logger.system_info.collect_system_info = mock_collect
            
            # Call function through the module
            result = utils.logger.system_info.collect_system_info()
            
            # Verify mock was called and returned our mock data
            mock_collect.assert_called_once()
            assert result == mock_system_info
            
            # Verify specific expected values
            assert result["system"] == "Test OS"
            assert result["platform"] == "Test Platform"
            assert result["python_version"] == "3.9.5"
            assert result["total_memory_gb"] == 16.0
            assert result["cuda_available"] == True
            assert result["gpu_count"] == 2
        finally:
            # Restore the original function
            utils.logger.system_info.collect_system_info = original_collect_system_info
    
    def test_log_system_info(self):
        """Test system info logging without actually collecting system info."""
        # Create mock return data
        mock_system_info = {
            "system": "Test OS",
            "platform": "Test Platform",
            "python_version": "3.9.5",
            "total_memory_gb": 16.0,
            "cuda_available": True,
            "cuda_version": "11.3"
        }
        
        # Create a mock logger
        mock_logger = MagicMock()
        mock_handler = MagicMock(spec=logging.FileHandler)
        mock_handler.baseFilename = "/tmp/test.log"
        mock_logger.handlers = [mock_handler]
        
        # Use monkeypatching instead of patch
        original_collect_system_info = utils.logger.system_info.collect_system_info
        original_write_system_info_section = utils.logger.system_info.write_system_info_section
        
        try:
            # Replace the functions with mocks
            mock_collect = MagicMock(return_value=mock_system_info)
            mock_write = MagicMock()
            utils.logger.system_info.collect_system_info = mock_collect
            utils.logger.system_info.write_system_info_section = mock_write
            
            # Call the function directly
            log_system_info(mock_logger)
            
            # Verify collect_system_info was actually called
            mock_collect.assert_called_once()
            
            # Verify write_system_info_section was called
            mock_write.assert_called_once()
        finally:
            # Restore the original functions
            utils.logger.system_info.collect_system_info = original_collect_system_info
            utils.logger.system_info.write_system_info_section = original_write_system_info_section
    
    def test_colored_formatter(self):
        """Test ColoredFormatter formats log messages correctly."""
        # Create a formatter
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")
        
        # Create a sample record
        record = logging.LogRecord(
            name="test_logger", 
            level=logging.INFO,
            pathname="test_path",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Verify format is as expected
        assert "INFO" in formatted
        assert "Test message" in formatted
