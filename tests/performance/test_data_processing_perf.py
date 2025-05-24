"""
Performance tests for data processing functions.
"""
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, mock_open
import os

from plot.data_processing import (
    clean_dataframe,
    process_engine_data,
    prepare_fuel_data_columns,
    normalize_fuel_levels,
    load_and_clean_data,
    compute_acceleration,
    compute_g_force
)
from utils.constants import G_FORCE_CONVERSION


@pytest.fixture(params=[100, 1000, 10000])
def sample_dataframe(request):
    """Create a sample dataframe of different sizes for performance testing."""
    row_count = request.param
    
    # Generate time values
    time_values = np.linspace(0, 500, row_count)
    
    # Generate realistic speed and altitude values with some noise
    base_speed = np.linspace(0, 3000, row_count)  # 0 to 3000 km/h
    speed_noise = np.random.normal(0, 50, row_count)
    
    base_altitude = np.linspace(0, 200, row_count)  # 0 to 200 km
    altitude_noise = np.random.normal(0, 2, row_count)
    
    # Add a small percentage of outliers (5%)
    outlier_indices = np.random.choice(row_count, int(row_count * 0.05), replace=False)
    
    speed_outliers = np.random.uniform(5000, 30000, len(outlier_indices))
    altitude_outliers = np.random.uniform(300, 1000, len(outlier_indices))
    
    superheavy_speed = base_speed + speed_noise
    superheavy_speed[outlier_indices] = speed_outliers
    
    starship_speed = base_speed * 1.2 + np.random.normal(0, 100, row_count)
    starship_altitude = base_altitude + altitude_noise
    starship_altitude[outlier_indices] = altitude_outliers
    
    return pd.DataFrame({
        "real_time_seconds": time_values,
        "superheavy.speed": superheavy_speed,
        "superheavy.altitude": base_altitude * 0.5 + np.random.normal(0, 1, row_count),
        "starship.speed": starship_speed,
        "starship.altitude": starship_altitude
    })


@pytest.fixture(params=[100, 1000, 10000])
def sample_engine_dataframe(request):
    """Create a sample dataframe with engine data for performance testing."""
    row_count = request.param
    
    # Create more complex dataframe with nested engine data
    df = pd.DataFrame({
        "real_time_seconds": np.linspace(0, 500, row_count)
    })
    
    # Generate random engine states for each row
    sh_central = []
    sh_inner = []
    sh_outer = []
    ss_rearth = []
    ss_rvac = []
    
    for i in range(row_count):
        # More engines activate as time progresses
        progress = min(1.0, i / (row_count * 0.8))
        
        # Create random engine states with increasing probability of being active
        sh_central.append([np.random.random() < (0.3 + 0.7 * progress) for _ in range(3)])
        sh_inner.append([np.random.random() < (0.2 + 0.8 * progress) for _ in range(10)])
        sh_outer.append([np.random.random() < (0.1 + 0.9 * progress) for _ in range(20)])
        ss_rearth.append([np.random.random() < (0.4 + 0.6 * progress) for _ in range(3)])
        ss_rvac.append([np.random.random() < (0.3 + 0.7 * progress) for _ in range(3)])
    
    # Add engine data to the dataframe
    df['superheavy.engines.central_stack'] = sh_central
    df['superheavy.engines.inner_ring'] = sh_inner
    df['superheavy.engines.outer_ring'] = sh_outer
    df['starship.engines.rearth'] = ss_rearth
    df['starship.engines.rvac'] = ss_rvac
    
    return df


@pytest.fixture(params=[100, 1000, 10000])
def sample_fuel_dataframe(request):
    """Create a sample dataframe with fuel data for performance testing."""
    row_count = request.param
    
    # Generate time values
    time_values = np.linspace(0, 500, row_count)
    
    # Generate decreasing fuel levels (100% to 0%)
    base_fuel_level = np.linspace(100, 0, row_count)
    
    # Add some noise and variations between fuel types
    sh_lox = np.clip(base_fuel_level + np.random.normal(0, 3, row_count), 0, 100)
    sh_ch4 = np.clip(base_fuel_level * 0.9 + np.random.normal(0, 3, row_count), 0, 100)
    ss_lox = np.clip(base_fuel_level * 1.1 + np.random.normal(0, 3, row_count), 0, 100)
    ss_ch4 = np.clip(base_fuel_level * 0.95 + np.random.normal(0, 3, row_count), 0, 100)
    
    return pd.DataFrame({
        "real_time_seconds": time_values,
        "superheavy.fuel.lox.fullness": sh_lox,
        "superheavy.fuel.ch4.fullness": sh_ch4,
        "starship.fuel.lox.fullness": ss_lox,
        "starship.fuel.ch4.fullness": ss_ch4
    })


@pytest.fixture
def mock_json_data(sample_dataframe, sample_engine_dataframe, sample_fuel_dataframe):
    """Create mock JSON data for load_and_clean_data testing."""
    # Combine all dataframes
    row_count = len(sample_dataframe)
    
    # Create JSON-like structure from dataframes
    json_data = []
    
    for i in range(row_count):
        # Basic frame data
        frame_data = {
            "frame_number": i,
            "superheavy": {
                "speed": sample_dataframe["superheavy.speed"].iloc[i],
                "altitude": sample_dataframe["superheavy.altitude"].iloc[i],
            },
            "starship": {
                "speed": sample_dataframe["starship.speed"].iloc[i],
                "altitude": sample_dataframe["starship.altitude"].iloc[i],
            },
            "real_time_seconds": sample_dataframe["real_time_seconds"].iloc[i]
        }
        
        # Add engine data if available
        if i < len(sample_engine_dataframe):
            frame_data["superheavy"]["engines"] = {
                "central_stack": sample_engine_dataframe["superheavy.engines.central_stack"].iloc[i],
                "inner_ring": sample_engine_dataframe["superheavy.engines.inner_ring"].iloc[i],
                "outer_ring": sample_engine_dataframe["superheavy.engines.outer_ring"].iloc[i]
            }
            frame_data["starship"]["engines"] = {
                "rearth": sample_engine_dataframe["starship.engines.rearth"].iloc[i],
                "rvac": sample_engine_dataframe["starship.engines.rvac"].iloc[i]
            }
        
        # Add fuel data if available
        if i < len(sample_fuel_dataframe):
            frame_data["superheavy"]["fuel"] = {
                "lox": {"fullness": sample_fuel_dataframe["superheavy.fuel.lox.fullness"].iloc[i]},
                "ch4": {"fullness": sample_fuel_dataframe["superheavy.fuel.ch4.fullness"].iloc[i]}
            }
            frame_data["starship"]["fuel"] = {
                "lox": {"fullness": sample_fuel_dataframe["starship.fuel.lox.fullness"].iloc[i]},
                "ch4": {"fullness": sample_fuel_dataframe["starship.fuel.ch4.fullness"].iloc[i]}
            }
        
        json_data.append(frame_data)
    
    return json_data


@pytest.mark.performance
def test_clean_dataframe_performance(benchmark, sample_dataframe):
    """Test performance of the clean_dataframe function with different sized datasets."""
    # Benchmark the function
    result = benchmark(clean_dataframe, sample_dataframe.copy())
    
    # Basic validation
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_dataframe)
    assert 'starship.speed_diff' in result.columns


@pytest.mark.performance
def test_process_engine_data_performance(benchmark, sample_engine_dataframe):
    """Test performance of the process_engine_data function."""
    # Benchmark the function
    result = benchmark(process_engine_data, sample_engine_dataframe.copy())
    
    # Basic validation
    assert isinstance(result, pd.DataFrame)
    assert 'superheavy_central_active' in result.columns
    assert 'starship_all_active' in result.columns


@pytest.mark.performance
def test_prepare_fuel_data_columns_performance(benchmark, sample_fuel_dataframe):
    """Test performance of the prepare_fuel_data_columns function."""
    # Remove some columns to test column creation
    test_df = sample_fuel_dataframe.copy()
    
    # Benchmark the function
    result = benchmark(prepare_fuel_data_columns, test_df)
    
    # Basic validation
    assert isinstance(result, pd.DataFrame)
    assert 'superheavy.fuel.lox.fullness' in result.columns
    assert 'starship.fuel.ch4.fullness' in result.columns


@pytest.mark.performance
def test_normalize_fuel_levels_performance(benchmark, sample_fuel_dataframe):
    """Test performance of the normalize_fuel_levels function."""
    # Benchmark the function
    result = benchmark(normalize_fuel_levels, sample_fuel_dataframe.copy())
    
    # Basic validation
    assert isinstance(result, pd.DataFrame)
    # Check for the presence of the original fuel columns
    assert 'superheavy.fuel.lox.fullness' in result.columns
    assert 'superheavy.fuel.ch4.fullness' in result.columns
    assert 'starship.fuel.lox.fullness' in result.columns
    assert 'starship.fuel.ch4.fullness' in result.columns


@pytest.mark.performance
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
def test_load_and_clean_data_performance(mock_json_load, mock_open, benchmark, mock_json_data):
    """Test performance of the complete data loading and cleaning pipeline."""
    # Configure mock to return our test data
    mock_json_load.return_value = mock_json_data
    
    # Benchmark the function
    result = benchmark(load_and_clean_data, "dummy.json")
    
    # Basic validation
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@pytest.mark.performance
@pytest.mark.parametrize("frame_distance", [1, 10, 30])
def test_compute_acceleration_performance(benchmark, sample_dataframe, frame_distance):
    """Test performance of acceleration computation with different frame distances."""
    # Benchmark the function
    result = benchmark(compute_acceleration, sample_dataframe, "starship.speed", frame_distance)
    
    # Basic validation
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_dataframe)


@pytest.mark.performance
def test_compute_g_force_performance(benchmark, sample_dataframe):
    """Test performance of G-force computation."""
    # First compute acceleration to get input for G-force
    acceleration = compute_acceleration(sample_dataframe, "starship.speed")
    
    # Benchmark the G-force computation
    result = benchmark(compute_g_force, acceleration)
    
    # Basic validation
    assert isinstance(result, pd.Series)
    assert len(result) == len(acceleration)


@pytest.mark.performance
def test_data_processing_pipeline_scaling(benchmark, mock_json_data):
    """Test how the complete data processing pipeline scales with input size."""
    with patch('builtins.open', new_callable=mock_open):
        with patch('json.load', return_value=mock_json_data):
            
            # Define a function that runs the complete pipeline
            def run_complete_pipeline():
                df = load_and_clean_data("dummy.json")
                df['starship.acceleration'] = compute_acceleration(df, "starship.speed")
                df['starship.g_force'] = compute_g_force(df['starship.acceleration'])
                return df
            
            # Benchmark the complete pipeline
            result = benchmark(run_complete_pipeline)
            
            # Basic validation
            assert isinstance(result, pd.DataFrame)
            assert 'starship.g_force' in result.columns
