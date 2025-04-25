"""
Performance tests for OCR text extraction functions.
"""
import pytest
import numpy as np
from ocr.ocr import extract_values_from_roi, extract_single_value, extract_time

# Define different ROI sizes for performance testing
ROI_SIZES = [
    (50, 25),     # Small ROI
    (100, 50),    # Medium ROI
    (200, 100)    # Large ROI
]

TEXT_TYPES = [
    "speed",      # Numeric value
    "altitude",   # Numeric value
    "time"        # Time format
]


@pytest.fixture(params=ROI_SIZES)
def test_rois(request):
    """Create test ROIs of different sizes with simulated text content."""
    height, width = request.param
    
    # Create several test ROIs with different types of content
    rois = []
    
    # Speed ROI with digits
    speed_roi = np.zeros((height, width), dtype=np.uint8)
    # Draw white text-like pattern in the middle
    text_height = height // 2
    text_width = width // 2
    y_offset = (height - text_height) // 2
    x_offset = (width - text_width) // 2
    speed_roi[y_offset:y_offset+text_height, x_offset:x_offset+text_width] = 255
    rois.append(speed_roi)
    
    # Altitude ROI
    altitude_roi = np.zeros((height, width), dtype=np.uint8)
    altitude_roi[y_offset:y_offset+text_height, x_offset:x_offset+text_width] = 255
    rois.append(altitude_roi)
    
    # Time ROI with time format
    time_roi = np.zeros((height, width), dtype=np.uint8)
    time_roi[y_offset:y_offset+text_height, x_offset:x_offset+text_width] = 255
    rois.append(time_roi)
    
    # Convert to 3-channel for compatibility
    return [np.dstack([roi, roi, roi]) for roi in rois]


@pytest.mark.performance
def test_extract_values_performance(benchmark, test_rois):
    """Test performance of extract_values_from_roi function with different modes."""
    # Use the first ROI for testing speed extraction
    speed_roi = test_rois[0]
    
    # Benchmark the function
    result = benchmark(extract_values_from_roi, speed_roi, mode="speed", debug=False)
    
    # Basic validation
    assert isinstance(result, dict)


@pytest.mark.performance
@pytest.mark.parametrize("mode", TEXT_TYPES)
def test_extract_values_different_modes(benchmark, test_rois, mode):
    """Test how performance varies with different extraction modes."""
    # Use different ROIs based on the mode
    mode_idx = {"speed": 0, "altitude": 1, "time": 2}
    roi = test_rois[mode_idx.get(mode, 0)]
    
    # Benchmark the function
    result = benchmark(extract_values_from_roi, roi, mode=mode, debug=False)
    
    # Basic validation
    assert isinstance(result, dict)


@pytest.mark.performance
def test_extract_single_value_performance(benchmark):
    """Test performance of extract_single_value function."""
    # Create test data with varied text complexity
    test_texts = [
        "100",
        "The speed is 100 km/h",
        "Multiple numbers 100, 200, 300"
    ]
    
    # Measure performance for each text sample and return the average
    def extract_all_values():
        results = []
        for text in test_texts:
            results.append(extract_single_value(text))
        return results
    
    results = benchmark(extract_all_values)
    
    # Basic validation - should have one result per input
    assert len(results) == len(test_texts)


@pytest.mark.performance
def test_extract_time_performance(benchmark):
    """Test performance of extract_time function."""
    # Create test data with varied time complexity
    test_times = [
        "+00:01:30",
        "The time is +00:01:30",
        "-01:30:00 remaining"
    ]
    
    # Measure performance for each time format and return the average
    def extract_all_times():
        results = []
        for time in test_times:
            results.append(extract_time(time))
        return results
    
    results = benchmark(extract_all_times)
    
    # Basic validation - should have one result per input
    assert len(results) == len(test_times)
