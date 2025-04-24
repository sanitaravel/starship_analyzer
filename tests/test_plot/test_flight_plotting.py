"""
Tests for the flight_plotting module.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, call, ANY
import matplotlib
# Use non-interactive backend for testing to avoid Tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

from plot.flight_plotting import (
    maximize_figure_window,
    create_engine_group_plot,
    create_engine_timeline_plot,
    create_scatter_plot,
    create_engine_performance_correlation,
    create_fuel_level_plot,
    plot_flight_data
)
from utils.constants import (
    ENGINE_TIMELINE_PARAMS,
    ENGINE_PERFORMANCE_PARAMS,
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE
)


@pytest.fixture
def test_df():
    """Fixture to create a test DataFrame for flight data."""
    return pd.DataFrame({
        "real_time_seconds": np.linspace(0, 600, 100),
        "starship.speed": np.linspace(0, 5000, 100),
        "superheavy.speed": np.linspace(0, 2500, 100),
        "starship.altitude": np.linspace(0, 150, 100),
        "superheavy.altitude": np.linspace(0, 75, 100),
        "superheavy_central_active": [3] * 100,
        "superheavy_central_total": [3] * 100,
        "superheavy_inner_active": [10] * 100,
        "superheavy_inner_total": [10] * 100,
        "superheavy_outer_active": [20] * 100,
        "superheavy_outer_total": [20] * 100,
        "superheavy_all_active": [33] * 100,  
        "superheavy_all_total": [33] * 100,
        "starship_rearth_active": [3] * 100,
        "starship_rearth_total": [3] * 100,
        "starship_rvac_active": [3] * 100,
        "starship_rvac_total": [3] * 100,
        "starship_all_active": [6] * 100,
        "starship_all_total": [6] * 100,
        "starship_acceleration": np.random.normal(10, 2, 100),
        "superheavy_acceleration": np.random.normal(8, 1.5, 100),
        "starship_g_force": np.random.normal(1, 0.2, 100),
        "superheavy_g_force": np.random.normal(0.8, 0.15, 100),
        "superheavy.fuel.lox.fullness": np.linspace(100, 0, 100),
        "superheavy.fuel.ch4.fullness": np.linspace(100, 5, 100),
        "starship.fuel.lox.fullness": np.linspace(100, 20, 100),
        "starship.fuel.ch4.fullness": np.linspace(100, 25, 100)
    })


class TestWindowManagement:
    """Tests for window management functions."""
    
    def test_maximize_figure_window_qt_backend(self):
        """Test maximize_figure_window with Qt backend."""
        mock_manager = MagicMock()
        mock_window = MagicMock()
        mock_manager.window = mock_window
        
        with patch('matplotlib.pyplot.get_current_fig_manager', return_value=mock_manager):
            maximize_figure_window()
            mock_window.showMaximized.assert_called_once()
    
    def test_maximize_figure_window_tk_backend(self):
        """Test maximize_figure_window with TkAgg backend."""
        mock_manager = MagicMock()
        mock_window = MagicMock()
        mock_window.state = MagicMock()
        mock_window.tk = MagicMock()  # Add this to match condition
        mock_manager.window = mock_window
        
        # Remove showMaximized to force TkAgg path
        del mock_window.showMaximized
        
        with patch('matplotlib.pyplot.get_current_fig_manager', return_value=mock_manager):
            maximize_figure_window()
            mock_window.state.assert_called_once_with('zoomed')
    
    def test_maximize_figure_window_error_handling(self):
        """Test maximize_figure_window error handling."""
        with patch('matplotlib.pyplot.get_current_fig_manager', side_effect=Exception("Test error")):
            # Should not raise exception
            maximize_figure_window()


class TestEnginePlots:
    """Tests for engine plotting functions."""
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.lineplot')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.tick_params')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    @patch('plot.flight_plotting.maximize_figure_window')
    def test_create_engine_group_plot(self, mock_maximize, mock_makedirs, mock_savefig, mock_tight_layout, 
                                     mock_legend, mock_tick_params, mock_ylim, mock_ylabel, mock_xlabel, 
                                     mock_title, mock_lineplot, mock_figure, test_df):
        """Test creating engine group plot for a vehicle."""
        # Setup
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call function with Superheavy
        create_engine_group_plot(
            test_df, 
            "superheavy", 
            "test_folder", 
            "5",
            show_figures=False
        )
        
        # Verify the correct calls were made
        mock_figure.assert_called_once_with(figsize=FIGURE_SIZE)
        
        # Verify lineplot calls
        assert mock_lineplot.call_count == len(ENGINE_TIMELINE_PARAMS["superheavy"]["groups"])
        
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_ylim.assert_called_once()
        mock_tick_params.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_makedirs.assert_called_once_with("test_folder", exist_ok=True)
        mock_savefig.assert_called_once()
        
        # Make sure maximize_figure_window was not called (since show_figures=False)
        mock_maximize.assert_not_called()
    
    @patch('plot.flight_plotting.create_engine_group_plot')
    def test_create_engine_timeline_plot(self, mock_create_engine_group, test_df):
        """Test creating engine timeline plots for both vehicles."""
        # Call the function
        create_engine_timeline_plot(test_df, "test_folder", "5", show_figures=False)
        
        # Should call create_engine_group_plot twice (once for each vehicle)
        assert mock_create_engine_group.call_count == 2
        
        # Check each call individually for better debugging
        call_args_list = mock_create_engine_group.call_args_list
        
        first_call_args, first_call_kwargs = call_args_list[0]
        second_call_args, second_call_kwargs = call_args_list[1]
        
        # First call should be for superheavy
        assert first_call_args[1] == "superheavy"
        assert first_call_args[2] == "test_folder"
        assert first_call_args[3] == "5"
        # Check show_figures is correctly passed (might be positional or keyword)
        if 'show_figures' in first_call_kwargs:
            assert first_call_kwargs['show_figures'] is False
        else:
            # If it's a positional argument, it will be in args[4]
            if len(first_call_args) > 4:
                assert first_call_args[4] is False
        
        # Second call should be for starship
        assert second_call_args[1] == "starship"
        assert second_call_args[2] == "test_folder"
        assert second_call_args[3] == "5"
        # Check show_figures is correctly passed (might be positional or keyword)
        if 'show_figures' in second_call_kwargs:
            assert second_call_kwargs['show_figures'] is False
        else:
            # If it's a positional argument, it will be in args[4]
            if len(second_call_args) > 4:
                assert second_call_args[4] is False


class TestScatterPlot:
    """Tests for scatter plot function."""
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.scatterplot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tick_params')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    @patch('plot.flight_plotting.maximize_figure_window')
    def test_create_scatter_plot_basic(self, mock_maximize, mock_makedirs, mock_savefig, mock_legend, 
                                     mock_tick_params, mock_title, mock_ylabel, mock_xlabel, 
                                     mock_scatterplot, mock_figure, test_df):
        """Test creating a basic scatter plot."""
        # Setup
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call function
        create_scatter_plot(
            test_df, 
            "real_time_seconds", 
            "starship.speed", 
            "Test Plot", 
            "test_plot.png", 
            "Starship Speed", 
            "Time (s)", 
            "Speed (km/h)", 
            "test_folder", 
            "5", 
            show_figures=False
        )
        
        # Verify the correct calls were made
        mock_figure.assert_called_once_with(figsize=FIGURE_SIZE)
        mock_scatterplot.assert_called_once()
        mock_xlabel.assert_called_once_with("Time (s)", fontsize=LABEL_FONT_SIZE)
        mock_ylabel.assert_called_once_with("Speed (km/h)", fontsize=LABEL_FONT_SIZE)
        mock_title.assert_called_once_with("Launch 5 - Test Plot", fontsize=TITLE_FONT_SIZE)
        mock_tick_params.assert_called_once()
        mock_legend.assert_called_once()
        mock_makedirs.assert_called_once_with("test_folder", exist_ok=True)
        mock_savefig.assert_called_once()
        
        # Make sure maximize_figure_window was not called (since show_figures=False)
        mock_maximize.assert_not_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.scatterplot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.plot')  # For trendline
    @patch('matplotlib.pyplot.tick_params')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    def test_create_scatter_plot_with_trendline(self, mock_makedirs, mock_savefig, mock_legend, 
                                              mock_tick_params, mock_plot, mock_title, mock_ylabel, 
                                              mock_xlabel, mock_scatterplot, mock_figure, test_df):
        """Test creating a scatter plot with acceleration data (should add trendline)."""
        # Setup
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call function with acceleration data (should trigger trendline)
        create_scatter_plot(
            test_df, 
            "real_time_seconds", 
            "starship_acceleration", 
            "Acceleration Plot", 
            "acceleration_plot.png", 
            "Starship Acceleration", 
            "Time (s)", 
            "Acceleration (m/s²)", 
            "test_folder", 
            "5", 
            show_figures=False
        )
        
        # Verify the scatterplot was created
        mock_scatterplot.assert_called_once()
        
        # Verify the trendline was added
        mock_plot.assert_called_once()
        
        # Update this to use TITLE_FONT_SIZE rather than hardcoded 16
        mock_title.assert_called_once_with("Launch 5 - Acceleration Plot", fontsize=TITLE_FONT_SIZE)


class TestEnginePerformanceCorrelation:
    """Tests for engine performance correlation plots."""
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.scatterplot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tick_params')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    @patch('plot.flight_plotting.maximize_figure_window')
    def test_create_engine_performance_correlation(self, mock_maximize, mock_makedirs, mock_savefig, 
                                                 mock_legend, mock_tick_params, mock_title, mock_ylabel, 
                                                 mock_xlabel, mock_scatterplot, mock_figure, test_df):
        """Test creating an engine performance correlation plot."""
        # Setup
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Setup legend mock return
        legend_mock = MagicMock()
        mock_legend.return_value = legend_mock
        
        # Call function
        create_engine_performance_correlation(
            test_df, 
            "superheavy", 
            "test_folder", 
            "5", 
            show_figures=False
        )
        
        # Verify the correct calls were made
        mock_figure.assert_called_once_with(figsize=FIGURE_SIZE)
        mock_scatterplot.assert_called_once()
        
        # Verify that setp was called on the legend title
        params = ENGINE_PERFORMANCE_PARAMS["superheavy"]
        mock_xlabel.assert_called_once_with(params["x_label"], fontsize=LABEL_FONT_SIZE)
        mock_ylabel.assert_called_once_with(params["y_label"], fontsize=LABEL_FONT_SIZE)
        
        # Verify the title included the launch number
        mock_title.assert_called_once_with(f"Launch 5 - {params['title']}", fontsize=TITLE_FONT_SIZE)
        
        # Verify file was saved
        mock_makedirs.assert_called_once_with("test_folder", exist_ok=True)
        mock_savefig.assert_called_once_with(f"test_folder/{params['filename']}", dpi=300, bbox_inches='tight')


class TestFuelLevelPlot:
    """Tests for fuel level plot function."""
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.lineplot')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tick_params')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    @patch('plot.flight_plotting.maximize_figure_window')
    def test_create_fuel_level_plot(self, mock_maximize, mock_makedirs, mock_savefig, mock_tight_layout,
                                  mock_legend, mock_tick_params, mock_title, mock_ylabel, mock_xlabel,
                                  mock_grid, mock_ylim, mock_lineplot, mock_figure, test_df):
        """Test creating a fuel level plot."""
        # Setup
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call function
        create_fuel_level_plot(
            test_df,
            "real_time_seconds",
            ["superheavy.fuel.lox.fullness", "superheavy.fuel.ch4.fullness"],
            "Superheavy Fuel Levels",
            "superheavy_fuel.png",
            ["LOX", "CH4"],
            "Time (s)",
            "Fuel Level (%)",
            "test_folder",
            "5",
            show_figures=False
        )
        
        # Verify the correct calls were made
        mock_figure.assert_called_once_with(figsize=FIGURE_SIZE)
        
        # Should call lineplot once for each fuel type
        assert mock_lineplot.call_count == 2
        mock_ylim.assert_called_once_with(0, 100)
        mock_grid.assert_called_once_with(True, alpha=0.3)
        
        mock_xlabel.assert_called_once_with("Time (s)", fontsize=LABEL_FONT_SIZE)
        mock_ylabel.assert_called_once_with("Fuel Level (%)", fontsize=LABEL_FONT_SIZE)
        mock_title.assert_called_once_with("Launch 5 - Superheavy Fuel Levels", fontsize=TITLE_FONT_SIZE)
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        
        mock_makedirs.assert_called_once_with("test_folder", exist_ok=True)
        mock_savefig.assert_called_once_with("test_folder/superheavy_fuel.png", dpi=300, bbox_inches='tight')


class TestPlotFlightData:
    """Tests for the main flight data plotting function."""
    
    @patch('plot.flight_plotting.load_and_clean_data')
    @patch('plot.flight_plotting.extract_launch_number')
    @patch('plot.flight_plotting.compute_acceleration')
    @patch('plot.flight_plotting.compute_g_force')
    @patch('plot.flight_plotting.prepare_fuel_data_columns')
    @patch('plot.flight_plotting.create_fuel_level_plot')
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.lineplot')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    @patch('plot.flight_plotting.create_engine_performance_correlation')
    @patch('plot.flight_plotting.create_scatter_plot')
    @patch('plot.interactive_viewer.show_plots_interactively')
    def test_plot_flight_data(self, mock_viewer, mock_scatter, mock_correlation,
                             mock_makedirs, mock_savefig, mock_lineplot, mock_figure,
                             mock_fuel_plot, mock_prepare_fuel, mock_g_force,
                             mock_acceleration, mock_extract, mock_load_data, test_df):
        """Test the main flight data plotting function."""
        # Setup mocks
        mock_load_data.return_value = test_df
        mock_extract.return_value = "5"
        mock_prepare_fuel.return_value = test_df
        
        mock_viewer_instance = MagicMock()
        mock_viewer.return_value = mock_viewer_instance
        
        # Call function
        plot_flight_data("test_json.json", 0, 100, show_figures=True)
        
        # Verify the core data processing was done
        mock_load_data.assert_called_once_with("test_json.json")
        mock_extract.assert_called_once()
        assert mock_acceleration.call_count == 2  # Once for each vehicle
        assert mock_g_force.call_count == 2  # Once for each vehicle
        mock_prepare_fuel.assert_called_once()
        
        # Verify plots were created
        mock_figure.assert_called()  # Multiple figures created
        mock_savefig.assert_called()  # Multiple saves
        mock_makedirs.assert_called()  # Directories created
        
        # Verify interactive viewer
        mock_viewer.assert_called_once()
        mock_viewer_instance.show.assert_called_once()
    
    @patch('plot.flight_plotting.load_and_clean_data')
    def test_plot_flight_data_empty_df(self, mock_load_data):
        """Test plotting with an empty DataFrame."""
        # Setup mock to return empty DataFrame
        mock_load_data.return_value = pd.DataFrame()
        
        # Call function
        plot_flight_data("test_json.json")
        
        # Verify early exit occurred
        mock_load_data.assert_called_once_with("test_json.json")
