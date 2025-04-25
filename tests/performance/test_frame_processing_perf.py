"""
Performance tests for frame processing functions.
"""
import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from processing.video_processing.frame_processing import (
    process_frame,
    process_single_frame,
)

from ocr.extract_data import extract_data
from ocr import extract_values_from_roi


@pytest.fixture(params=[(640, 360), (1280, 720), (1920, 1080), (3840, 2160)])
def test_frame(request):
    """
    Create test frames of different resolutions for performance testing.
    """
    width, height = request.param
    # Create a test frame with specific regions to simulate telemetry data areas
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Scale factors for different resolutions
    scale_x = width / 1920
    scale_y = height / 1080
    
    # Add some simulated regions of interest (ROIs)
    # SuperHeavy speed ROI
    x1, y1 = int(359 * scale_x), int(913 * scale_y)
    x2, y2 = int(442 * scale_x), int(938 * scale_y)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # Gray box for speed
    
    # SuperHeavy altitude ROI
    x1, y1 = int(392 * scale_x), int(948 * scale_y)
    x2, y2 = int(442 * scale_x), int(973 * scale_y)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # Gray box for altitude
    
    # Starship speed ROI
    x1, y1 = int(1539 * scale_x), int(913 * scale_y)
    x2, y2 = int(1622 * scale_x), int(938 * scale_y)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # Gray box for speed
    
    # Starship altitude ROI
    x1, y1 = int(1572 * scale_x), int(948 * scale_y)
    x2, y2 = int(1622 * scale_x), int(973 * scale_y)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # Gray box for altitude
    
    # Time ROI
    x1, y1 = int(860 * scale_x), int(940 * scale_y)
    x2, y2 = int(1057 * scale_x), int(984 * scale_y)
    frame[y1:y2, x1:x2] = [200, 200, 200]  # Gray box for time
    
    # Add simulated engine regions
    # SH engines
    engine_center_x = int(width / 2)
    engine_center_y = int(height * 0.8)
    engine_radius = int(5 * scale_x)
    
    # Add a grid of "engines"
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = engine_center_x + i * int(20 * scale_x)
            y = engine_center_y + j * int(20 * scale_y)
            if 0 <= x < width and 0 <= y < height:
                # Some engines are "on" (bright), some are "off" (dim)
                brightness = 255 if (i + j) % 2 == 0 else 50
                cv2.circle(frame, (x, y), engine_radius, [0, 0, brightness], -1)
    
    # Add some simulated fuel level gauges
    gauge_width = int(100 * scale_x)
    gauge_height = int(10 * scale_y)
    
    # LOX gauge
    x1, y1 = int(200 * scale_x), int(500 * scale_y)
    cv2.rectangle(frame, (x1, y1), (x1 + gauge_width, y1 + gauge_height), [150, 150, 255], -1)
    
    # CH4 gauge
    x1, y1 = int(200 * scale_x), int(520 * scale_y)
    cv2.rectangle(frame, (x1, y1), (x1 + int(gauge_width * 0.75), y1 + gauge_height), [150, 255, 150], -1)
    
    # Add some text to the frame to simulate numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = scale_x * 0.5
    cv2.putText(frame, "100 km/h", (x1, y1 - 10), font, font_scale, [255, 255, 255], 1)
    cv2.putText(frame, "5000 m", (x1, y1 + 30), font, font_scale, [255, 255, 255], 1)
    
    return frame


@pytest.fixture
def mock_extract_data():
    """Mock extract_data for isolated component testing."""
    with patch('processing.video_processing.frame_processing.extract_data') as mock:
        mock.return_value = (
            # SuperHeavy data
            {"speed": 100, "altitude": 5000, "engines": {"central": [True, False, True], "inner": [True] * 5 + [False] * 5, "outer": [True] * 15 + [False] * 5}},
            # Starship data
            {"speed": 200, "altitude": 10000, "engines": {"raptor": [True, True, False]}},
            # Time data
            {"sign": "+", "hours": 0, "minutes": 1, "seconds": 30}
        )
        yield mock


@pytest.mark.performance
def test_process_frame_performance(benchmark, test_frame):
    """Test the performance of the process_frame function with different sized frames."""
    # Benchmark the function
    result = benchmark(process_frame, 1000, test_frame, False, False, False)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "frame_number" in result
    assert result["frame_number"] == 1000


@pytest.mark.performance
def test_process_single_frame_performance(benchmark, test_frame):
    """Test the performance of the process_single_frame function."""
    # Benchmark the function
    result = benchmark(process_single_frame, 1000, test_frame, False, False, False)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "frame_number" in result
    assert result["frame_number"] == 1000


@pytest.mark.performance
def test_extract_data_performance(benchmark, test_frame):
    """Test the performance of the extract_data function directly."""
    # Benchmark the function
    result = benchmark(extract_data, test_frame, False, False)
    
    # Basic validation
    assert isinstance(result, tuple)
    assert len(result) == 3


@pytest.mark.performance
@patch('ocr.extract_data.detect_engine_status')
@patch('ocr.extract_data.extract_fuel_levels')
def test_extract_data_without_engines_fuel(mock_fuel, mock_engines, benchmark, test_frame):
    """Test extract_data performance without engine detection and fuel extraction."""
    # Mock engine detection and fuel extraction to isolate text extraction performance
    mock_engines.return_value = {"superheavy": {}, "starship": {}}
    mock_fuel.return_value = {"superheavy": {}, "starship": {}}
    
    # Benchmark the function
    result = benchmark(extract_data, test_frame, False, False)
    
    # Basic validation
    assert isinstance(result, tuple)
    assert len(result) == 3


@pytest.mark.performance
@patch('ocr.extract_values_from_roi')
def test_extract_data_preprocessing_only(mock_extract_values, benchmark, test_frame):
    """Test the performance of just the image preprocessing part of extract_data."""
    # Configure mock to do nothing
    mock_extract_values.return_value = {"value": 0}
    
    # Create a test function that just does preprocessing
    def preprocess_only(frame):
        from ocr.extract_data import preprocess_image
        return preprocess_image(frame)
    
    # Benchmark the function
    rois = benchmark(preprocess_only, test_frame)
    
    # Basic validation
    assert len(rois) == 5


@pytest.mark.performance
def test_extract_values_from_roi_performance(benchmark):
    """Test the performance of extract_values_from_roi function."""
    # Create a small ROI with a number in it
    roi = np.zeros((25, 83, 3), dtype=np.uint8)
    # Add some text to the ROI
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(roi, "100", (10, 20), font, 0.5, [255, 255, 255], 1)
    
    # Benchmark the function for speed extraction
    result = benchmark(extract_values_from_roi, roi, mode="speed")
    
    # Basic validation
    assert isinstance(result, dict)


@pytest.mark.performance
def test_complete_frame_pipeline_performance(benchmark, test_frame):
    """Test the performance of the complete frame processing pipeline."""
    # Define a complete pipeline function
    def process_complete_pipeline():
        # 1. Process the frame
        frame_result = process_frame(1000, test_frame, False, False, False)
        return frame_result
    
    # Benchmark the complete pipeline
    result = benchmark(process_complete_pipeline)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "frame_number" in result
    assert result["frame_number"] == 1000


@pytest.mark.performance
@pytest.mark.parametrize("debug_mode", [False, True])
def test_debug_mode_impact(benchmark, test_frame, debug_mode):
    """Test the performance impact of debug mode."""
    # Benchmark with and without debug mode
    result = benchmark(process_frame, 1000, test_frame, False, debug_mode, False)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "frame_number" in result


@pytest.mark.performance
@pytest.mark.parametrize("display_rois", [False, True])
def test_display_rois_impact(benchmark, test_frame, display_rois):
    """Test the performance impact of displaying ROIs."""
    with patch('cv2.imshow'), patch('cv2.waitKey'):
        # Benchmark with and without ROI display
        result = benchmark(process_frame, 1000, test_frame, display_rois, False, False)
        
        # Basic validation
        assert isinstance(result, dict)
        assert "frame_number" in result
