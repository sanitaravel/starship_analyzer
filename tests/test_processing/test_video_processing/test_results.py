import pytest
import os
import json
import tempfile
from unittest.mock import patch, mock_open, ANY

from processing.video_processing.results import calculate_real_times, save_results


@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    return [
        {
            "frame_number": 100,
            "error": "Test error"
        },
        {
            "frame_number": 200,
            "time": {
                "sign": "+",
                "hours": 0,
                "minutes": 1,
                "seconds": 30
            },
            "superheavy": {"speed": 100},
            "starship": {"altitude": 5000}
        },
        {
            "frame_number": 300,
            "time": {
                "sign": "+",
                "hours": 0,
                "minutes": 2,
                "seconds": 0
            },
            "superheavy": {"speed": 150},
            "starship": {"altitude": 10000}
        }
    ]


class TestCalculateRealTimes:
    
    def test_with_no_zero_time_frame(self, sample_results):
        """Test when no zero time frame is provided."""
        result = calculate_real_times(sample_results, None, 30.0)
        # Should return the original results unchanged
        assert result == sample_results
    
    def test_with_zero_time_frame(self, sample_results):
        """Test calculation with a valid zero time frame."""
        result = calculate_real_times(sample_results, 200, 30.0)
        
        # Check real_time_seconds calculation
        assert "real_time_seconds" not in result[0]  # Has error, skipped
        assert result[1]["real_time_seconds"] == 0.0  # Zero frame
        assert result[2]["real_time_seconds"] == (300 - 200) / 30.0  # 3.33 seconds
        
        # Check real_time components
        assert result[1]["real_time"] == {"hours": 0, "minutes": 0, "seconds": 0, "milliseconds": 0}
        assert result[2]["real_time"]["seconds"] == 3
        assert result[2]["real_time"]["milliseconds"] == 333  # Approximately
    
    def test_with_negative_time(self, sample_results):
        """Test calculation with frames before zero time."""
        # Use the last frame as zero time
        result = calculate_real_times(sample_results, 300, 30.0)
        
        # Second frame should now have negative time
        assert result[1]["real_time_seconds"] < 0
        assert result[1]["real_time_seconds"] == (200 - 300) / 30.0  # -3.33 seconds


class TestSaveResults:
    
    def test_basic_save(self, sample_results, tmp_path):
        """Test basic save functionality with a temporary directory."""
        with patch('processing.video_processing.results.os.makedirs') as mock_makedirs:
            with patch('builtins.open', new_callable=mock_open) as mock_file:
                with patch('processing.video_processing.results.json.dump') as mock_json_dump:
                    save_results(sample_results, 42)
                    
                    # Check directory creation
                    mock_makedirs.assert_called_once_with(os.path.join("results", "launch_42"), exist_ok=True)
                    
                    # Check file opening
                    mock_file.assert_called_once_with(os.path.join("results", "launch_42", "results.json"), "w")
                    
                    # Check json.dump was called with correct parameters
                    mock_json_dump.assert_called_once()
                    args, kwargs = mock_json_dump.call_args
                    assert args[0] == sample_results  # First arg should be the results data
                    assert kwargs.get('indent') == 4  # Should use pretty formatting
    
    def test_save_error_handling(self, sample_results):
        """Test error handling during save."""
        with patch('processing.video_processing.results.os.makedirs') as mock_makedirs:
            # Configure open to raise an error for the primary save, but return a mock file object for backup
            mock_file = mock_open()
            with patch('builtins.open', side_effect=[IOError("Test error"), mock_file()]) as mock_open_patch:
                with patch('processing.video_processing.results.logger') as mock_logger:
                    save_results(sample_results, 42)
                    
                    # Verify primary error was logged
                    mock_logger.error.assert_called_once_with('Error saving results: Test error')
                    
                    # Verify backup path was used
                    assert mock_open_patch.call_count == 2
                    backup_call = mock_open_patch.call_args_list[1]
                    assert "backup_results_42.json" in backup_call[0][0]
    
    def test_save_backup_error(self, sample_results):
        """Test error handling when both primary and backup saves fail."""
        with patch('processing.video_processing.results.os.makedirs') as mock_makedirs:
            with patch('builtins.open', side_effect=[IOError("Primary error"), IOError("Backup error")]) as mock_open:
                with patch('processing.video_processing.results.logger') as mock_logger:
                    save_results(sample_results, 42)
                    
                    # Verify both errors were logged
                    assert mock_logger.error.call_count == 2
    
    @pytest.mark.integration
    def test_real_save(self, sample_results):
        """Integration test with real file saving."""
        # Create a temp directory for our test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the os.path.join calls in the save_results function
            original_join = os.path.join
            
            def mock_join(*args):
                if args[0] == "results":
                    # Replace the base directory for result saving
                    return original_join(temp_dir, *args[1:])
                # For all other calls, use the original join
                return original_join(*args)
            
            with patch('processing.video_processing.results.os.path.join', side_effect=mock_join):
                save_results(sample_results, 42)
                
                # Check that the file was created
                result_path = os.path.join(temp_dir, "launch_42", "results.json")
                assert os.path.exists(result_path)
                
                # Verify content
                with open(result_path, 'r') as f:
                    loaded_data = json.load(f)
                    assert len(loaded_data) == 3
                    assert loaded_data[1]["frame_number"] == 200
