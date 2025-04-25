"""
Performance tests for video processing functions.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import cv2
from typing import List

from processing.video_processing.batch_processing import create_batches, process_batch
from processing.video_processing.frame_processing import process_frame
from processing.video_processing.validation import get_video_properties


@pytest.fixture
def mock_video_capture():
    """Create a mock video capture object that returns synthetic frames."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock capture object
        cap_instance = MagicMock()
        mock_cap.return_value = cap_instance
        
        # Configure basic video properties
        cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        
        # Configure read to return synthetic frames
        def mock_read():
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Add some content to the frame
            frame[400:600, 800:1000] = [100, 100, 100]  # Gray box
            frame[300:350, 1400:1600] = [255, 255, 255]  # White area for text
            return True, frame
            
        cap_instance.read.side_effect = mock_read
        cap_instance.isOpened.return_value = True
        
        yield mock_cap


@pytest.fixture
def sample_batches():
    """Create sample batches of frame numbers with different sizes."""
    return [
        [0, 10, 20, 30, 40],           # Small batch 
        list(range(0, 100, 10)),       # Medium batch
        list(range(0, 300, 10))        # Large batch
    ]


@pytest.mark.performance
def test_create_batches_performance(benchmark):
    """Test performance of batch creation with different parameters."""
    # Test with a large number of frames
    result = benchmark(create_batches, 10000, 30, sample_rate=2)
    
    # Basic validation
    assert isinstance(result, list)
    assert all(isinstance(batch, list) for batch in result)
    assert len(result) > 0


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [10, 30, 100])
def test_create_batches_scaling(benchmark, batch_size):
    """Test how batch creation scales with different batch sizes."""
    # Keep frame count constant but vary batch size
    frame_count = 10000
    
    result = benchmark(create_batches, frame_count, batch_size)
    
    # Validate number of batches is correct
    expected_batch_count = (frame_count + batch_size - 1) // batch_size  # Ceiling division
    assert len(result) == expected_batch_count


@pytest.mark.performance
@patch('processing.video_processing.batch_processing.process_frame')
def test_process_batch_performance(mock_process_frame, benchmark, mock_video_capture):
    """Test performance of batch processing with mocked frame processing."""
    # Configure mock to return quickly
    mock_process_frame.return_value = {
        "frame_number": 0,
        "superheavy": {"speed": 100},
        "starship": {"altitude": 5000},
        "time": {"sign": "+", "hours": 0, "minutes": 0, "seconds": 30}
    }
    
    # Create a medium-sized batch for testing
    batch = list(range(0, 50, 5))  # [0, 5, 10, ..., 45]
    
    # Benchmark batch processing
    result = benchmark(process_batch, batch, "dummy_video.mp4", False, False, False)
    
    # Basic validation
    assert isinstance(result, list)
    assert len(result) == len(batch)


@pytest.mark.performance
@patch('ocr.extract_data.extract_values_from_roi')
@patch('ocr.extract_data.detect_engine_status')
@patch('ocr.extract_data.extract_fuel_levels')
def test_process_frame_performance(mock_fuel, mock_engines, mock_extract, benchmark):
    """Test performance of the frame processing function."""
    # Configure mocks to return test data quickly
    mock_extract.return_value = {"value": 100}
    mock_engines.return_value = {
        "superheavy": {"inner": [True, False], "outer": [True]},
        "starship": {"raptor": [True, False]}
    }
    mock_fuel.return_value = {
        "superheavy": {"lox": {"fullness": 85}, "ch4": {"fullness": 90}},
        "starship": {"lox": {"fullness": 75}, "ch4": {"fullness": 80}}
    }
    
    # Create a test frame
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Benchmark frame processing
    result = benchmark(process_frame, 100, test_frame, False, False, False)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "frame_number" in result
    assert "superheavy" in result
    assert "starship" in result
    assert "time" in result


@pytest.mark.performance
@patch('cv2.VideoCapture')
def test_get_video_properties_performance(mock_cap, benchmark):
    """Test performance of video property extraction."""
    # Configure the mock
    cap_instance = MagicMock()
    mock_cap.return_value = cap_instance
    cap_instance.isOpened.return_value = True
    cap_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 10000,
        cv2.CAP_PROP_FPS: 30.0
    }.get(prop, 0)
    
    # Benchmark property extraction
    frame_count, fps = benchmark(get_video_properties, "dummy_video.mp4")
    
    # Basic validation
    assert frame_count == 10000
    assert fps == 30.0


@pytest.mark.performance
def test_batch_size_impact(benchmark):
    """Test the impact of batch size on processing performance."""
    # This tests the create_batches function with varying batch sizes
    def create_multiple_batch_sizes():
        results = []
        for batch_size in [10, 50, 100, 500]:
            results.append(create_batches(10000, batch_size))
        return results
    
    results = benchmark(create_multiple_batch_sizes)
    
    # Basic validation
    assert len(results) == 4
    assert len(results[0]) > len(results[1]) > len(results[2]) > len(results[3])
