"""
Tests for constants defined in utils/constants.py.
"""
import pytest
from utils.constants import (
    # Engine Detection Constants
    SUPERHEAVY_ENGINES, STARSHIP_ENGINES, 
    # OCR Constants
    WHITE_THRESHOLD,
    # Data Processing Constants
    G_FORCE_CONVERSION,
    # Visualization Constants 
    FIGURE_SIZE, TITLE_FONT_SIZE, LINE_WIDTH,
    ENGINE_TIMELINE_PARAMS, ANALYZE_RESULTS_PLOT_PARAMS,
    FUEL_LEVEL_PLOT_PARAMS, COMPARE_FUEL_LEVEL_PARAMS,
    PLOT_MULTIPLE_LAUNCHES_PARAMS, ENGINE_PERFORMANCE_PARAMS
)

class TestConstants:
    """Test suite for application constants."""
    
    def test_engine_detection_constants(self):
        """Test engine detection coordinate constants."""
        # Check SUPERHEAVY_ENGINES structure
        assert isinstance(SUPERHEAVY_ENGINES, dict)
        assert "central_stack" in SUPERHEAVY_ENGINES
        assert "inner_ring" in SUPERHEAVY_ENGINES
        assert "outer_ring" in SUPERHEAVY_ENGINES
        
        # Check coordinate formats (tuples of two integers)
        for group, coordinates in SUPERHEAVY_ENGINES.items():
            assert isinstance(coordinates, list)
            for coord in coordinates:
                assert isinstance(coord, tuple)
                assert len(coord) == 2
                assert all(isinstance(val, int) for val in coord)
        
        # Verify expected number of coordinates
        assert len(SUPERHEAVY_ENGINES["central_stack"]) == 3
        assert len(SUPERHEAVY_ENGINES["inner_ring"]) == 10
        assert len(SUPERHEAVY_ENGINES["outer_ring"]) == 20
        
        # Check STARSHIP_ENGINES structure
        assert isinstance(STARSHIP_ENGINES, dict)
        assert "rearth" in STARSHIP_ENGINES
        assert "rvac" in STARSHIP_ENGINES
        assert len(STARSHIP_ENGINES["rearth"]) == 3
        assert len(STARSHIP_ENGINES["rvac"]) == 3
    
    def test_ocr_constants(self):
        """Test OCR threshold constants."""
        assert isinstance(WHITE_THRESHOLD, int)
        assert 0 <= WHITE_THRESHOLD <= 255  # Valid color threshold range
    
    def test_data_processing_constants(self):
        """Test data processing constants."""
        assert isinstance(G_FORCE_CONVERSION, (int, float))
        assert G_FORCE_CONVERSION > 0
    
    def test_visualization_constants(self):
        """Test visualization configuration constants."""
        # Check figure layout constants
        assert isinstance(FIGURE_SIZE, tuple)
        assert len(FIGURE_SIZE) == 2
        assert all(isinstance(val, (int, float)) for val in FIGURE_SIZE)
        
        # Check font sizes
        assert isinstance(TITLE_FONT_SIZE, (int, float))
        assert TITLE_FONT_SIZE > 0
        
        # Check line styling
        assert isinstance(LINE_WIDTH, (int, float))
        assert LINE_WIDTH > 0
    
    def test_engine_timeline_params(self):
        """Test engine timeline parameters structure."""
        assert isinstance(ENGINE_TIMELINE_PARAMS, dict)
        assert "superheavy" in ENGINE_TIMELINE_PARAMS
        assert "starship" in ENGINE_TIMELINE_PARAMS
        assert "xlabel" in ENGINE_TIMELINE_PARAMS
        
        # Check superheavy section
        sh_params = ENGINE_TIMELINE_PARAMS["superheavy"]
        assert "title" in sh_params
        assert "ylabel" in sh_params
        assert "ylim" in sh_params
        assert "groups" in sh_params
        assert isinstance(sh_params["groups"], list)
        
        # Check a group entry
        group = sh_params["groups"][0]
        assert "column" in group
        assert "label" in group
        assert "color" in group
    
    def test_plot_parameters(self):
        """Test plot parameter structures."""
        # Check analysis results plot params
        assert isinstance(ANALYZE_RESULTS_PLOT_PARAMS, list)
        for plot_params in ANALYZE_RESULTS_PLOT_PARAMS:
            assert isinstance(plot_params, tuple)
            assert len(plot_params) == 7
            
        # Check fuel level plot params
        assert isinstance(FUEL_LEVEL_PLOT_PARAMS, list)
        for plot_params in FUEL_LEVEL_PLOT_PARAMS:
            assert isinstance(plot_params, tuple)
            assert len(plot_params) == 7
            
        # Check multi-launch comparison params
        assert isinstance(COMPARE_FUEL_LEVEL_PARAMS, list)
        for plot_params in COMPARE_FUEL_LEVEL_PARAMS:
            assert isinstance(plot_params, tuple)
            assert len(plot_params) == 6
            
        # Check engine performance params
        assert isinstance(ENGINE_PERFORMANCE_PARAMS, dict)
        assert "superheavy" in ENGINE_PERFORMANCE_PARAMS
        assert "starship" in ENGINE_PERFORMANCE_PARAMS
        
        engine_params = ENGINE_PERFORMANCE_PARAMS["superheavy"]
        assert "x_col" in engine_params
        assert "y_col" in engine_params
        assert "color_col" in engine_params
        assert "title" in engine_params
        assert "x_label" in engine_params
        assert "y_label" in engine_params
        assert "filename" in engine_params
