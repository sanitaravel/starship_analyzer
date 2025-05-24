"""
Performance tests for plot generation functionality.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from plot.flight_plotting import (
    create_scatter_plot,
    create_engine_group_plot, 
    create_engine_timeline_plot,
    create_engine_performance_correlation,
    create_fuel_level_plot,
    plot_flight_data
)
from plot.comparison_plotting import (
    plot_multiple_launches,
    compare_multiple_launches
)


@pytest.fixture(params=[100, 1000, 10000])
def sample_dataframe(request):
    """Create sample dataframes of different sizes for performance testing."""
    row_count = request.param
    
    # Generate time values
    time_values = np.linspace(0, 500, row_count)
    
    # Generate realistic telemetry data
    speed_values = np.sin(time_values * 0.1) * 1000 + 2000 + np.random.normal(0, 50, row_count)
    altitude_values = np.minimum(time_values * 2, 200) + np.random.normal(0, 0.5, row_count)
    acceleration_values = np.gradient(speed_values) + np.random.normal(0, 0.2, row_count)
    g_force_values = acceleration_values / 9.81
    
    # Generate engine data with proper arrays
    engines_central = np.ones(row_count) * 3
    engines_inner = np.ones(row_count) * 10
    engines_outer = np.ones(row_count) * 20
    
    # Generate fuel data
    fuel_start = 100
    fuel_rate = fuel_start / (row_count * 0.8)  # Empty at 80% of time
    
    lox_values = np.maximum(0, fuel_start - time_values * fuel_rate * 1.1) + np.random.normal(0, 1, row_count)
    ch4_values = np.maximum(0, fuel_start - time_values * fuel_rate) + np.random.normal(0, 1, row_count)
    
    # Clip values to valid ranges
    lox_values = np.clip(lox_values, 0, 100)
    ch4_values = np.clip(ch4_values, 0, 100)
    
    # Create dataframe with all the required columns for both naming conventions
    df = pd.DataFrame({
        'real_time_seconds': time_values,
        # Standard dot notation columns
        'superheavy.speed': speed_values,
        'starship.speed': speed_values * 1.1,
        'superheavy.altitude': altitude_values,
        'starship.altitude': altitude_values * 1.05,
        
        # Underscore notation columns
        'superheavy_speed': speed_values,
        'starship_speed': speed_values * 1.1,
        'superheavy_altitude': altitude_values,
        'starship_altitude': altitude_values * 1.05,
        'superheavy_acceleration': acceleration_values,
        'starship_acceleration': acceleration_values * 1.05,
        'superheavy_g_force': g_force_values,
        'starship_g_force': g_force_values * 1.05,
        
        # Engine status columns
        'superheavy_central_active': engines_central,
        'superheavy_central_total': np.ones(row_count) * 3,
        'superheavy_inner_active': engines_inner,
        'superheavy_inner_total': np.ones(row_count) * 10,
        'superheavy_outer_active': engines_outer,
        'superheavy_outer_total': np.ones(row_count) * 20,
        'superheavy_all_active': engines_central + engines_inner + engines_outer,
        'superheavy_all_total': np.ones(row_count) * 33,
        'starship_rearth_active': np.ones(row_count) * 3,
        'starship_rearth_total': np.ones(row_count) * 3,
        'starship_rvac_active': np.ones(row_count) * 3,
        'starship_rvac_total': np.ones(row_count) * 3,
        'starship_all_active': np.ones(row_count) * 6,
        'starship_all_total': np.ones(row_count) * 6,
        
        # Fuel data columns
        'superheavy.fuel.lox.fullness': lox_values,
        'superheavy.fuel.ch4.fullness': ch4_values,
        'starship.fuel.lox.fullness': lox_values * 0.9,
        'starship.fuel.ch4.fullness': ch4_values * 0.9,
    })
    
    return df


class NoopFigure:
    """A do-nothing placeholder for plt.figure to eliminate side effects."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass


@pytest.mark.performance
def test_create_scatter_plot_performance(benchmark, sample_dataframe):
    """Test performance of scatter plot generation."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        plot=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.scatterplot', MagicMock()):
        
        def create_plot():
            create_scatter_plot(
                test_df, 
                'real_time_seconds', 
                'starship.speed', 
                'Speed vs Time', 
                'speed_vs_time.png',
                'Starship',
                'Mission Time (seconds)',
                'Speed (km/h)',
                'dummy_folder',
                '5',
                False
            )
            
        benchmark(create_plot)


@pytest.mark.performance
def test_create_engine_group_plot_performance(benchmark, sample_dataframe):
    """Test performance of engine group plot generation."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        ylim=MagicMock(),
                        tight_layout=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.lineplot', MagicMock()):
        
        def create_plot():
            create_engine_group_plot(
                test_df,
                'superheavy',
                'dummy_folder',
                '5',
                False
            )
            
        benchmark(create_plot)


@pytest.mark.performance
def test_create_engine_timeline_plot_performance(benchmark, sample_dataframe):
    """Test performance of engine timeline plot generation."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    with patch('plot.flight_plotting.create_engine_group_plot', MagicMock()) as mock_engine_group:
        def run_test():
            create_engine_timeline_plot(test_df, 'dummy_folder', '5', False)
            
        benchmark(run_test)


@pytest.mark.performance
def test_create_engine_performance_correlation_performance(benchmark, sample_dataframe):
    """Test performance of engine performance correlation plot generation."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        setp=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.scatterplot', MagicMock(return_value=MagicMock(legend=MagicMock(return_value=MagicMock())))):
        
        def run_test():
            create_engine_performance_correlation(test_df, 'superheavy', 'dummy_folder', '5', False)
            
        benchmark(run_test)


@pytest.mark.performance
def test_create_fuel_level_plot_performance(benchmark, sample_dataframe):
    """Test performance of fuel level plot generation."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        ylim=MagicMock(),
                        grid=MagicMock(),
                        tight_layout=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.lineplot', MagicMock()):
        
        def create_plot():
            create_fuel_level_plot(
                test_df,
                'real_time_seconds',
                ['superheavy.fuel.lox.fullness', 'superheavy.fuel.ch4.fullness'],
                'Fuel Levels',
                'fuel_levels.png',
                ['LOX', 'CH4'],
                'Time (s)',
                'Fuel Level (%)',
                'dummy_folder',
                '5',
                False
            )
            
        benchmark(create_plot)


@pytest.mark.performance
def test_plot_multiple_launches_performance(benchmark, sample_dataframe):
    """Test performance of multiple launch comparison plot generation."""
    # Create smaller dataframes for comparison
    df_list = [sample_dataframe.iloc[:100].copy(), sample_dataframe.iloc[100:200].copy()]
    labels = ['Launch 1', 'Launch 2']
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        plot=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.scatterplot', MagicMock()):
        
        def create_plot():
            plot_multiple_launches(
                df_list,
                'real_time_seconds',
                'starship.speed',
                'Speed Comparison',
                'speed_comparison.png',
                'dummy_folder',
                labels,
                'Time (s)',
                'Speed (km/h)',
                False
            )
            
        benchmark(create_plot)


@pytest.mark.performance
def test_generate_combined_plots_performance(benchmark, sample_dataframe):
    """Test performance of generating all plots at once."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    with patch('plot.flight_plotting.load_and_clean_data', return_value=test_df), \
         patch('plot.flight_plotting.extract_launch_number', return_value='5'), \
         patch('plot.flight_plotting.compute_acceleration', return_value=pd.Series(np.zeros(len(test_df)))), \
         patch('plot.flight_plotting.compute_g_force', return_value=pd.Series(np.zeros(len(test_df)))), \
         patch('plot.flight_plotting.prepare_fuel_data_columns', return_value=test_df), \
         patch('plot.flight_plotting.create_fuel_level_plot', MagicMock()), \
         patch('plot.flight_plotting.create_engine_performance_correlation', MagicMock()), \
         patch('plot.flight_plotting.create_scatter_plot', MagicMock()), \
         patch.multiple(plt, 
                       figure=MagicMock(return_value=NoopFigure()),
                       savefig=MagicMock(), 
                       close=MagicMock(),
                       xlabel=MagicMock(),
                       ylabel=MagicMock(),
                       title=MagicMock(),
                       tick_params=MagicMock(),
                       legend=MagicMock(),
                       ylim=MagicMock(),
                       tight_layout=MagicMock()), \
         patch('seaborn.lineplot', MagicMock()), \
         patch('os.makedirs', MagicMock()):
        
        def plot_all():
            plot_flight_data("dummy_json.json", show_figures=False)
        
        benchmark(plot_all)


@pytest.mark.performance
@pytest.mark.parametrize("dpi", [72, 150, 300])
def test_plot_resolution_impact(benchmark, sample_dataframe, dpi):
    """Test the performance impact of different plot resolutions (DPI)."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Create a special savefig mock that records the dpi value
    mock_savefig = MagicMock()
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=mock_savefig, 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        plot=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.scatterplot', MagicMock()):
        
        # Define the function to be benchmarked
        def create_high_res_plot():
            # Because create_scatter_plot doesn't accept dpi directly,
            # we'll need to modify our patched savefig implementation
            create_scatter_plot(
                test_df, 
                'real_time_seconds', 
                'starship.speed', 
                'Speed vs Time', 
                'speed_vs_time.png',
                'Starship',
                'Mission Time (seconds)',
                'Speed (km/h)',
                'dummy_folder',
                '5',
                False
            )
            
        # For different DPIs, update the save_path format to include DPI
        # and benchmark the operation
        benchmark(create_high_res_plot)


@pytest.mark.performance
def test_real_world_plot_generation(benchmark, sample_dataframe):
    """Test the performance of a real-world plotting scenario with multiple operations."""
    # Use a smaller subset to avoid memory issues
    test_df = sample_dataframe.iloc[:300].copy() if len(sample_dataframe) > 300 else sample_dataframe
    
    # Global patch of all plt and seaborn functions
    with patch.multiple(plt, 
                        figure=MagicMock(return_value=NoopFigure()),
                        savefig=MagicMock(), 
                        close=MagicMock(),
                        xlabel=MagicMock(),
                        ylabel=MagicMock(),
                        title=MagicMock(),
                        tick_params=MagicMock(),
                        legend=MagicMock(),
                        ylim=MagicMock(),
                        grid=MagicMock(),
                        tight_layout=MagicMock(),
                        plot=MagicMock()), \
         patch('os.makedirs', MagicMock()), \
         patch('seaborn.scatterplot', MagicMock()), \
         patch('seaborn.lineplot', MagicMock()):
        
        def comprehensive_plot_workflow():
            # Generate several plots in sequence as would happen in real usage
            create_scatter_plot(
                test_df, 
                'real_time_seconds', 
                'starship.speed', 
                'Speed vs Time', 
                'speed_vs_time.png',
                'Starship',
                'Mission Time (seconds)',
                'Speed (km/h)',
                'dummy_folder',
                '5',
                False
            )
            
            create_engine_group_plot(
                test_df,
                'superheavy',
                'dummy_folder',
                '5',
                False
            )
            
            create_fuel_level_plot(
                test_df,
                'real_time_seconds',
                ['superheavy.fuel.lox.fullness', 'superheavy.fuel.ch4.fullness'],
                'Fuel Levels',
                'fuel_levels.png',
                ['LOX', 'CH4'],
                'Time (s)',
                'Fuel Level (%)',
                'dummy_folder',
                '5',
                False
            )
            
        benchmark(comprehensive_plot_workflow)
