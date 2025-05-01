"""
Performance tests for engine detection functions.
"""
import pytest
import numpy as np
from ocr.engine_detection import check_engines, check_engines_numba, detect_engine_status

# Define image sizes for performance testing
IMAGE_SIZES = [
    (480, 270),   # 480p (reduced)
    (1280, 720),  # 720p
    (1920, 1080)  # 1080p
]


@pytest.fixture(params=IMAGE_SIZES)
def test_images(request):
    """Create test images of different resolutions with simulated engine patterns."""
    height, width = request.param
    
    # Create base image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create bright spots to simulate engine flames
    # Distribute across the image proportionally to resolution
    engine_locations = [
        # Center engines
        (width // 2, height // 2),
        (width // 2 - width // 10, height // 2),
        (width // 2 + width // 10, height // 2),
        
        # Ring of engines
        (width // 2, height // 3),
        (width // 3, height // 2),
        (width // 2, height * 2 // 3),
        (width * 2 // 3, height // 2)
    ]
    
    # Create bright spots for engines (some on, some off)
    for i, (x, y) in enumerate(engine_locations):
        # Make some engines bright (on) and some dim (off)
        brightness = 255 if i % 2 == 0 else 100
        
        # Create a flame pattern
        flame_radius = max(3, min(width, height) // 50)
        y_start = max(0, y - flame_radius)
        y_end = min(height, y + flame_radius)
        x_start = max(0, x - flame_radius)
        x_end = min(width, x + flame_radius)
        
        image[y_start:y_end, x_start:x_end] = [0, 0, brightness]  # Blue channel for visibility
    
    # Add some noise
    noise = np.random.randint(0, 30, size=image.shape, dtype=np.uint8)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


@pytest.fixture
def engine_coordinates():
    """Generate test engine coordinates."""
    # Create a dictionary of engine coordinates similar to the actual engine layout
    return {
        'central': np.array([(100, 100), (200, 100), (300, 100)]),
        'inner': np.array([(100, 200), (200, 200), (300, 200), (400, 200)]),
        'outer': np.array([(100, 300), (200, 300), (300, 300), (400, 300), (500, 300)])
    }


@pytest.mark.performance
def test_check_engines_numba_performance(benchmark, test_images, engine_coordinates):
    """Test performance of the check_engines_numba function."""
    # Extract coordinates for testing
    coords = engine_coordinates['central']
    
    # Set a threshold that should result in some engine detections
    threshold = 150
    
    # Benchmark the function
    result = benchmark(check_engines_numba, test_images, coords, threshold)
    
    # Basic validation
    assert isinstance(result, list)
    assert all(isinstance(x, bool) for x in result)


@pytest.mark.performance
def test_check_engines_performance(benchmark, test_images, engine_coordinates):
    """Test performance of the check_engines function."""
    # Benchmark the function with various parameters
    result = benchmark(check_engines, test_images, engine_coordinates, False, "Test")
    
    # Basic validation
    assert isinstance(result, dict)
    assert "central" in result
    assert "inner" in result
    assert "outer" in result


@pytest.mark.performance
def test_detect_engine_status_performance(benchmark, test_images):
    """Test performance of the detect_engine_status function."""
    # Benchmark the complete engine detection pipeline
    result = benchmark(detect_engine_status, test_images, debug=False)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "superheavy" in result
    assert "starship" in result


@pytest.mark.performance
def test_engine_detection_scaling(benchmark, test_images):
    """Test how engine detection performance scales with image size."""
    def detect_all_engines():
        # Full detection with both vehicles
        return detect_engine_status(test_images, debug=False)
    
    # Benchmark the complete process
    result = benchmark(detect_all_engines)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "superheavy" in result
    assert "starship" in result
