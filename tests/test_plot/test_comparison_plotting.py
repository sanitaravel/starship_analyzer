"""
Tests for the comparison_plotting module.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, call
import matplotlib
# Use non-interactive backend for testing to avoid Tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

from plot.comparison_plotting import (
    maximize_figure_window,
    plot_multiple_launches,
    compare_multiple_launches
)
from plot.data_processing import (
    load_and_clean_data,
    compute_acceleration,
    compute_g_force
)
from utils.constants import (LABEL_FONT_SIZE, TITLE_FONT_SIZE, TICK_FONT_SIZE, 
                           LEGEND_FONT_SIZE, MARKER_SIZE, MARKER_ALPHA, LINE_WIDTH)


@pytest.fixture
def test_dataframes():
    """Fixture to create test dataframes representing different launches."""
    # Create mock dataframes with minimal data needed for comparison tests
    df1 = pd.DataFrame({
        'real_time_seconds': np.arange(0, 100, 10),
        'starship_speed': np.arange(0, 1000, 100),
        'superheavy_speed': np.arange(0, 500, 50),
        'starship.altitude': np.arange(0, 2000, 200),
        'superheavy.altitude': np.arange(0, 1000, 100),
        'lox_level': np.linspace(100, 0, 10),
        'methane_level': np.linspace(100, 5, 10)
    })
    
    # Add some variation for the second dataframe
    df2 = pd.DataFrame({
        'real_time_seconds': np.arange(0, 100, 10),
        'starship_speed': np.arange(50, 1050, 100),
        'superheavy_speed': np.arange(25, 525, 50),
        'starship.altitude': np.arange(100, 2100, 200),
        'superheavy.altitude': np.arange(50, 1050, 100),
        'lox_level': np.linspace(100, 5, 10),
        'methane_level': np.linspace(100, 0, 10)
    })
    
    # Calculate acceleration and g-force for both dataframes
    df1['superheavy_acceleration'] = compute_acceleration(df1, 'superheavy_speed')
    df1['starship_acceleration'] = compute_acceleration(df1, 'starship_speed')
    df1['superheavy_g_force'] = compute_g_force(df1['superheavy_acceleration'])
    df1['starship_g_force'] = compute_g_force(df1['starship_acceleration'])
    
    df2['superheavy_acceleration'] = compute_acceleration(df2, 'superheavy_speed')
    df2['starship_acceleration'] = compute_acceleration(df2, 'starship_speed')
    df2['superheavy_g_force'] = compute_g_force(df2['superheavy_acceleration'])
    df2['starship_g_force'] = compute_g_force(df2['starship_acceleration'])
    
    return [df1, df2]


@pytest.fixture
def mock_json_paths():
    """Fixture to create mock JSON paths."""
    return ["results/launch_4/results.json", "results/launch_5/results.json"]


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


class TestPlotMultipleLaunches:
    """Tests for the plot_multiple_launches function."""
    
    def test_plot_multiple_launches_basic(self, test_dataframes):
        """Test basic functionality of plot_multiple_launches."""
        df_list = test_dataframes
        labels = ["Launch 1", "Launch 2"]
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('seaborn.scatterplot') as mock_scatter, \
             patch('matplotlib.pyplot.plot') as mock_plot, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tick_params') as mock_tick_params, \
             patch('matplotlib.pyplot.legend') as mock_legend, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('os.makedirs') as mock_makedirs, \
             patch('plot.comparison_plotting.maximize_figure_window') as mock_maximize:
            
            # Create a mock figure
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            
            # Call the function
            plot_multiple_launches(
                df_list,
                'real_time_seconds',
                'starship_speed',
                'Test Plot',
                'test_plot.png',
                'test_folder',
                labels,
                'Time (s)',
                'Speed (m/s)',
                show_figures=False  # Don't show figures to avoid UI interactions
            )
            
            # Verify function calls with actual font sizes from constants
            mock_figure.assert_called_once()
            assert mock_scatter.call_count == 2  # Once for each dataframe
            mock_xlabel.assert_called_once_with('Time (s)', fontsize=LABEL_FONT_SIZE)
            mock_ylabel.assert_called_once_with('Speed (m/s)', fontsize=LABEL_FONT_SIZE)
            mock_title.assert_called_once_with('Test Plot', fontsize=TITLE_FONT_SIZE)
            mock_tick_params.assert_called_once_with(labelsize=TICK_FONT_SIZE)
            mock_legend.assert_called_once_with(frameon=True, fontsize=LEGEND_FONT_SIZE)
            mock_makedirs.assert_called_once_with('test_folder', exist_ok=True)
            mock_savefig.assert_called_once()
            # maximize_figure_window should not be called when show_figures=False
            mock_maximize.assert_not_called()

    def test_plot_multiple_launches_with_trendline(self, test_dataframes):
        """Test plot_multiple_launches with trendlines for acceleration data."""
        df_list = test_dataframes
        labels = ["Launch 1", "Launch 2"]
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('seaborn.scatterplot') as mock_scatter, \
             patch('matplotlib.pyplot.plot') as mock_plot, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('os.makedirs') as mock_makedirs:
            
            # Create a mock figure
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            
            # Patch the condition that checks for minimum data points
            # This ensures the trendline plotting code is always executed
            with patch('plot.comparison_plotting.len', side_effect=lambda x: 40):  # Always return 40 (> 30)
                # Call the function with acceleration data to trigger trendline
                plot_multiple_launches(
                    df_list,
                    'real_time_seconds',
                    'starship_acceleration',  # This should trigger trendline
                    'Acceleration Test',
                    'acceleration_test.png',
                    'test_folder',
                    labels,
                    'Time (s)',
                    'Acceleration (m/s²)',
                    show_figures=False
                )
                
                # Verify scatter and plot calls (plot is for trendlines)
                assert mock_scatter.call_count == 2  # Once for each dataframe
                assert mock_plot.call_count == 2  # Once for each trendline
                mock_savefig.assert_called_once()

    def test_plot_multiple_launches_interactive_mode(self, test_dataframes):
        """Test plot_multiple_launches in interactive mode."""
        df_list = test_dataframes
        labels = ["Launch 1", "Launch 2"]
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('seaborn.scatterplot') as mock_scatter, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('os.makedirs') as mock_makedirs, \
             patch('plot.comparison_plotting.maximize_figure_window') as mock_maximize, \
             patch('matplotlib.pyplot.show') as mock_show:
            
            # Create a mock figure
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            
            # Call the function with show_figures=True
            plot_multiple_launches(
                df_list,
                'real_time_seconds',
                'starship_speed',
                'Interactive Test',
                'interactive_test.png',
                'test_folder',
                labels,
                'Time (s)',
                'Speed (m/s)',
                show_figures=True
            )
            
            # Verify maximize and show were called
            mock_maximize.assert_called_once()
            mock_show.assert_called_once()


class TestCompareMultipleLaunches:
    """Tests for the compare_multiple_launches function."""
    
    @patch('plot.comparison_plotting.load_and_clean_data')
    @patch('plot.comparison_plotting.plot_multiple_launches')
    def test_compare_multiple_launches_basic(self, mock_plot, mock_load_data, test_dataframes, mock_json_paths):
        """Test basic functionality of compare_multiple_launches."""
        # Setup the mock to return our test dataframes
        mock_load_data.side_effect = test_dataframes
        
        with patch('os.path.join', return_value='test/output/folder'):
            # Call the function
            compare_multiple_launches(0, 100, *mock_json_paths, show_figures=False)
            
            # Verify load_and_clean_data called for each JSON path
            assert mock_load_data.call_count == 2
            mock_load_data.assert_any_call(mock_json_paths[0])
            mock_load_data.assert_any_call(mock_json_paths[1])
            
            # Verify plot_multiple_launches was called multiple times
            assert mock_plot.call_count > 0
    
    @patch('plot.comparison_plotting.load_and_clean_data')
    @patch('plot.comparison_plotting.plot_multiple_launches')
    def test_compare_multiple_launches_empty_data(self, mock_plot, mock_load_data, mock_json_paths):
        """Test compare_multiple_launches with empty dataframes."""
        # Setup the mock to return empty dataframes
        mock_load_data.return_value = pd.DataFrame()
        
        # Call the function
        compare_multiple_launches(0, 100, *mock_json_paths, show_figures=False)
        
        # Verify plot_multiple_launches was not called
        mock_plot.assert_not_called()
    
    @patch('plot.comparison_plotting.load_and_clean_data')
    @patch('plot.comparison_plotting.plot_multiple_launches')
    @patch('plot.interactive_viewer.show_plots_interactively')
    def test_compare_multiple_launches_interactive(self, mock_show_viewer, mock_plot, 
                                                mock_load_data, test_dataframes, mock_json_paths):
        """Test compare_multiple_launches in interactive mode."""
        # Setup the mock to return our test dataframes
        mock_load_data.side_effect = test_dataframes
        
        # Mock the interactive viewer
        mock_viewer = MagicMock()
        mock_show_viewer.return_value = mock_viewer
        
        # Call the function with show_figures=True
        compare_multiple_launches(0, 100, *mock_json_paths, show_figures=True)
        
        # Verify interactive viewer was created and shown
        mock_show_viewer.assert_called_once_with("Multiple Launches Comparison")
        mock_viewer.show.assert_called_once()
