"""
Tests for download utility functions in download/utils.py.
"""
import pytest
from unittest.mock import patch, MagicMock
import json
import requests

from download.utils import get_launch_data, get_downloaded_launches, FLIGHTS_URL

class TestGetLaunchData:
    """Test suite for get_launch_data function."""
    
    @patch('download.utils.requests.get')
    def test_get_launch_data_success(self, mock_get):
        """Test successful retrieval of launch data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [{"flight": 1, "name": "Test"}, {"flight": 2, "name": "Test2"}]
        mock_get.return_value = mock_response
        
        # Call the function
        result = get_launch_data()
        
        # Assert results
        assert result is not None
        assert len(result) == 2
        assert result[0]["flight"] == 1
        assert result[1]["name"] == "Test2"
        mock_get.assert_called_once_with(FLIGHTS_URL, timeout=10)
    
    @patch('download.utils.requests.get')
    def test_get_launch_data_request_error(self, mock_get):
        """Test handling of request exceptions."""
        # Setup mock to raise exception
        mock_get.side_effect = requests.RequestException("Connection error")
        
        # Call the function with mocked print
        with patch('builtins.print') as mock_print:
            result = get_launch_data()
            
            # Assert results
            assert result is None
            mock_print.assert_called_with("Error fetching flight data: Connection error")
    
    @patch('download.utils.requests.get')
    def test_get_launch_data_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Setup mock to raise exception
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
        mock_get.return_value = mock_response
        
        # Call the function with mocked print
        with patch('builtins.print') as mock_print:
            result = get_launch_data()
            
            # Assert results
            assert result is None
            mock_print.assert_called_with("Error fetching flight data: 404 Client Error")
    
    @patch('download.utils.requests.get')
    def test_get_launch_data_json_error(self, mock_get):
        """Test handling of JSON decoding errors."""
        # Setup mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        # Call the function with mocked print
        with patch('builtins.print') as mock_print:
            result = get_launch_data()
            
            # Assert results
            assert result is None
            mock_print.assert_called_with("Error parsing flight data: Invalid JSON: line 1 column 1 (char 0)")


class TestGetDownloadedLaunches:
    """Test suite for get_downloaded_launches function."""
    
    @patch('os.path.exists')
    def test_get_downloaded_launches_path_not_exists(self, mock_exists):
        """Test when output path does not exist."""
        # Setup mock
        mock_exists.return_value = False
        
        # Call the function
        result = get_downloaded_launches()
        
        # Assert results
        assert result == []
        mock_exists.assert_called_once_with("flight_recordings")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_downloaded_launches_empty_dir(self, mock_listdir, mock_exists):
        """Test when output directory is empty."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        # Call the function
        result = get_downloaded_launches()
        
        # Assert results
        assert result == []
        mock_exists.assert_called_once_with("flight_recordings")
        mock_listdir.assert_called_once_with("flight_recordings")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_downloaded_launches_with_files(self, mock_listdir, mock_exists):
        """Test when output directory contains flight files."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "flight_1.mp4", 
            "flight_2.mp4", 
            "flight_5.mp4",
            "other_file.mp4",
            "not_a_flight.txt"
        ]
        
        # Call the function
        result = get_downloaded_launches()
        
        # Assert results
        assert sorted(result) == [1, 2, 5]
        mock_exists.assert_called_once_with("flight_recordings")
        mock_listdir.assert_called_once_with("flight_recordings")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_downloaded_launches_invalid_filenames(self, mock_listdir, mock_exists):
        """Test handling of invalid filenames."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "flight_.mp4",  # Missing number
            "flight_abc.mp4",  # Non-numeric
            "flight_1",  # Missing extension
            "flight_2.mp4.part"  # Multiple extensions
        ]
        
        # Call the function
        result = get_downloaded_launches()
        
        # Assert results - should only get valid ones
        assert result == [1, 2]
        mock_exists.assert_called_once_with("flight_recordings")
        mock_listdir.assert_called_once_with("flight_recordings")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_downloaded_launches_custom_path(self, mock_listdir, mock_exists):
        """Test using a custom output path."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["flight_1.mp4", "flight_2.mp4"]
        custom_path = "custom/path"
        
        # Call the function
        result = get_downloaded_launches(output_path=custom_path)
        
        # Assert results
        assert sorted(result) == [1, 2]
        mock_exists.assert_called_once_with(custom_path)
        mock_listdir.assert_called_once_with(custom_path)
