"""
Performance tests for fuel level extraction functions.
"""
import pytest
import numpy as np
from ocr.fuel_level_extraction import extract_fuel_levels, process_strip

# Define image sizes for performance testing
IMAGE_SIZES = [
    (480, 270),   # 480p (reduced)
    (1280, 720),  # 720p
    (1920, 1080)  # 1080p
]


@pytest.fixture(params=IMAGE_SIZES)
def test_images(request):
    """Create test images of different resolutions with simulated fuel level bars."""
    height, width = request.param
    
    # Create base image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add simulated fuel level bars in different positions
    # These are vertical bright regions that simulate fuel level displays
    bar_height = height // 4
    bar_width = width // 20
    
    # Place bars at positions where they would be detected for each strip
    bars_x = [width // 5, width // 2, 3 * width // 4, 4 * width // 5]
    bars_y = [height // 2, height // 2, height // 2, height // 2]
    
    # Add bars with different lengths (simulating different fuel levels)
    for i, (x, y) in enumerate(zip(bars_x, bars_y)):
        # Calculate bar length based on position (50%-80% full)
        fullness = 0.5 + (i * 0.1)  # 50%, 60%, 70%, 80%
        bar_length = int(bar_height * fullness)
        
        # Make the bar region bright
        image[y-bar_height//2:y-bar_height//2+bar_length, x:x+bar_width] = 230
    
    # Add some background variation
    noise = np.random.randint(0, 30, size=(height, width), dtype=np.uint8)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


@pytest.mark.performance
def test_extract_fuel_levels_performance(benchmark, test_images):
    """Test performance of the extract_fuel_levels function."""
    result = benchmark(extract_fuel_levels, test_images)
    
    # Basic validation
    assert isinstance(result, dict)
    assert "superheavy" in result
    assert "starship" in result
    assert "lox" in result["superheavy"]
    assert "ch4" in result["superheavy"]
    assert "lox" in result["starship"]
    assert "ch4" in result["starship"]


@pytest.mark.performance
def test_process_strip_performance(benchmark, test_images):
    """Test performance of the process_strip function."""
    # Test the first strip (index 0)
    result = benchmark(process_strip, test_images, 0)
    
    # Basic validation - check keys instead of type since Numba returns a special dict type
    assert "fullness" in result
    assert "length" in result
    assert "ref_diff" in result
    assert isinstance(result["fullness"], float)
    assert 0 <= result["fullness"] <= 100


@pytest.mark.performance
def test_strip_processing_scaling(benchmark, test_images):
    """Test how strip processing scales with different image sizes."""
    # Process all four strips to measure overall scaling
    def process_all_strips():
        results = []
        for i in range(4):  # There are 4 strips to process
            results.append(process_strip(test_images, i))
        return results
    
    results = benchmark(process_all_strips)
    
    # Basic validation - check contents not exact type
    assert len(results) == 4
    assert all("fullness" in r for r in results)
