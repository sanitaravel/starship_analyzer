"""
Tests for the data_processing module.
"""
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock, mock_open
import os

from plot.data_processing import (
    validate_json, 
    clean_dataframe, 
    process_engine_data, 
    prepare_fuel_data_columns, 
    normalize_fuel_levels,
    load_and_clean_data, 
    compute_acceleration, 
    compute_g_force
)
from utils.constants import G_FORCE_CONVERSION


@pytest.fixture
def sample_json_data():
    """Fixture to create sample JSON data for testing."""
    return [
        {
            "frame_number": 0,
            "superheavy": {"speed": 0, "altitude": 0},
            "starship": {"speed": 0, "altitude": 0},
            "time": "00:00:00",
            "real_time_seconds": 0
        },
        {
            "frame_number": 1,
            "superheavy": {"speed": 10, "altitude": 0.1},
            "starship": {"speed": 10, "altitude": 0.1},
            "time": "00:00:01",
            "real_time_seconds": 1
        }
    ]


@pytest.fixture
def sample_engine_data():
    """Fixture to create sample engine data for testing."""
    return pd.DataFrame({
        "superheavy.engines.central_stack": [[True, True, False], [True, True, True]],
        "superheavy.engines.inner_ring": [[True] * 5 + [False] * 5, [True] * 8 + [False] * 2],
        "superheavy.engines.outer_ring": [[True] * 15 + [False] * 5, [True] * 18 + [False] * 2],
        "starship.engines.rearth": [[True, True, False], [True, True, True]],
        "starship.engines.rvac": [[True, True, True], [True, True, True]],
        "real_time_seconds": [0, 1]
    })


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    return pd.DataFrame({
        "real_time_seconds": np.arange(10),
        "starship.speed": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "superheavy.speed": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "starship.altitude": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "superheavy.altitude": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        # Add some outliers for cleaning
        "starship.speed_diff": [0, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        "superheavy.speed_diff": [0, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    })


@pytest.fixture
def sample_fuel_dataframe():
    """Fixture to create a sample DataFrame with fuel data for testing."""
    return pd.DataFrame({
        "real_time_seconds": [0, 50, 100, 150, 200, 250, 300],
        "superheavy.fuel.lox.fullness": [100, 90, 80, 70, 60, 50, 40],
        "superheavy.fuel.ch4.fullness": [100, 85, 70, 55, 20, 10, 5],  # Increased difference at index 4+
        "starship.fuel.lox.fullness": [100, 95, 90, 85, 80, 75, 70],
        "starship.fuel.ch4.fullness": [100, 90, 80, 70, 45, 35, 25]  # 35% difference at index 4+
    })


class TestValidateJSON:
    """Tests for validate_json function."""
    
    def test_valid_json(self, sample_json_data):
        """Test validation with valid JSON data."""
        result, invalid_entry = validate_json(sample_json_data)
        assert result is True
        assert invalid_entry is None
    
    def test_invalid_json_missing_key(self):
        """Test validation with invalid JSON data (missing key)."""
        invalid_data = [
            {
                "frame_number": 0,
                "superheavy": {"speed": 0, "altitude": 0},
                # Missing "starship" key
                "time": "00:00:00",
                "real_time_seconds": 0
            }
        ]
        result, invalid_entry = validate_json(invalid_data)
        assert result is False
        assert invalid_entry is not None
        assert "starship" not in invalid_entry
    
    def test_invalid_json_empty_list(self):
        """Test validation with empty JSON list."""
        result, invalid_entry = validate_json([])
        assert result is True  # Empty list technically valid
        assert invalid_entry is None


class TestCleanDataframe:
    """Tests for clean_dataframe function."""
    
    def test_clean_dataframe(self, sample_dataframe):
        """Test basic dataframe cleaning functionality."""
        original_cols = set(sample_dataframe.columns)
        df_cleaned = clean_dataframe(sample_dataframe.copy())
        
        # Check that numeric conversion happened
        assert df_cleaned['starship.speed'].dtype == float
        assert df_cleaned['superheavy.speed'].dtype == float
        
        # Check that original columns are preserved (the function adds _diff columns)
        for col in original_cols:
            assert col in df_cleaned.columns
            
        # Check that the diff columns are added
        assert 'starship.speed_diff' in df_cleaned.columns
        assert 'superheavy.speed_diff' in df_cleaned.columns
        assert 'starship.altitude_diff' in df_cleaned.columns
        assert 'superheavy.altitude_diff' in df_cleaned.columns
        
        # Check that row count remains the same
        assert len(df_cleaned) == len(sample_dataframe)
    
    def test_clean_dataframe_with_outliers(self):
        """Test cleaning with outliers to clip."""
        df = pd.DataFrame({
            "starship.speed": [0, 30000, 20, -10],  # Values outside range
            "superheavy.speed": [0, 10, 7000, 20],  # Values outside range
            "starship.altitude": [0, 0.1, 250, 0.3],  # Values outside range
            "superheavy.altitude": [0, 0.05, 0.1, 150],  # Values outside range
        })
        
        df_cleaned = clean_dataframe(df)
        
        # Check that values were clipped properly
        assert df_cleaned['starship.speed'].max() <= 28000
        assert df_cleaned['starship.speed'].min() >= 0
        assert df_cleaned['superheavy.speed'].max() <= 6000
        assert df_cleaned['superheavy.speed'].min() >= 0
        assert df_cleaned['starship.altitude'].max() <= 200
        assert df_cleaned['starship.altitude'].min() >= 0
        assert df_cleaned['superheavy.altitude'].max() <= 100
        assert df_cleaned['superheavy.altitude'].min() >= 0
    
    def test_clean_dataframe_abrupt_changes(self):
        """Test cleaning with abrupt changes."""
        df = pd.DataFrame({
            "starship.speed": [10, 12, 15, 100, 20, 22],  # Abrupt change
            "superheavy.speed": [5, 8, 70, 10, 12, 15],   # Abrupt change
            "starship.altitude": [0.1, 0.2, 10, 0.4, 0.5, 0.6],  # Abrupt change
            "superheavy.altitude": [0.05, 0.1, 0.15, 5, 0.25, 0.3],  # Abrupt change
            "real_time_seconds": [0, 1, 2, 3, 4, 5]
        })
        
        df_cleaned = clean_dataframe(df)
        
        # Check that abrupt changes were detected and nullified
        # starship.speed_diff at index 3 would be 85, which is > 50
        assert pd.isna(df_cleaned['starship.speed'].iloc[3])
        # superheavy.speed_diff at index 2 would be 62, which is > 50
        assert pd.isna(df_cleaned['superheavy.speed'].iloc[2])
        # starship.altitude_diff at index 2 would be 9.8, which is > 1
        assert pd.isna(df_cleaned['starship.altitude'].iloc[2])
        # superheavy.altitude_diff at index 3 would be 4.85, which is > 1
        assert pd.isna(df_cleaned['superheavy.altitude'].iloc[3])


class TestProcessEngineData:
    """Tests for process_engine_data function."""
    
    def test_process_engine_data(self, sample_engine_data):
        """Test basic engine data processing."""
        df_processed = process_engine_data(sample_engine_data.copy())
        
        # Check that engine count columns are created
        assert 'superheavy_central_active' in df_processed.columns
        assert 'superheavy_all_active' in df_processed.columns
        assert 'starship_rearth_active' in df_processed.columns
        assert 'starship_all_active' in df_processed.columns
        
        # Check values calculated correctly
        assert df_processed['superheavy_central_active'].iloc[0] == 2  # [True, True, False]
        assert df_processed['superheavy_central_active'].iloc[1] == 3  # [True, True, True]
        
        assert df_processed['superheavy_inner_active'].iloc[0] == 5  # [True] * 5 + [False] * 5
        assert df_processed['superheavy_inner_active'].iloc[1] == 8  # [True] * 8 + [False] * 2
        
        assert df_processed['superheavy_all_active'].iloc[0] == 22  # 2 + 5 + 15
        assert df_processed['superheavy_all_active'].iloc[1] == 29  # 3 + 8 + 18
        
        assert df_processed['starship_all_active'].iloc[0] == 5  # 2 + 3
        assert df_processed['starship_all_active'].iloc[1] == 6  # 3 + 3
        
        # Check that original engine columns were dropped
        assert 'superheavy.engines.central_stack' not in df_processed.columns
        assert 'starship.engines.rvac' not in df_processed.columns
    
    def test_process_engine_data_no_engine_columns(self):
        """Test with DataFrame missing engine columns."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1],
            "starship.speed": [0, 10]
        })
        
        df_processed = process_engine_data(df.copy())
        
        # Check that engine count columns were created with zeros
        assert df_processed['superheavy_central_active'].iloc[0] == 0
        assert df_processed['starship_all_active'].iloc[0] == 0
        
        # Check that original data is preserved
        assert df_processed['real_time_seconds'].iloc[1] == 1
        assert df_processed['starship.speed'].iloc[1] == 10


class TestPrepareFuelDataColumns:
    """Tests for prepare_fuel_data_columns function."""
    
    def test_prepare_fuel_data_no_columns(self):
        """Test preparing fuel columns when no columns exist."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1],
            "starship.speed": [0, 10]
        })
        
        df_prepared = prepare_fuel_data_columns(df.copy())
        
        # Check that columns were created
        assert 'superheavy.fuel.lox.fullness' in df_prepared.columns
        assert 'superheavy.fuel.ch4.fullness' in df_prepared.columns
        assert 'starship.fuel.lox.fullness' in df_prepared.columns
        assert 'starship.fuel.ch4.fullness' in df_prepared.columns
        
        # Check that values were set to 0
        assert df_prepared['superheavy.fuel.lox.fullness'].iloc[0] == 0
        assert df_prepared['starship.fuel.ch4.fullness'].iloc[0] == 0
    
    def test_prepare_fuel_data_alternative_names(self):
        """Test preparing fuel columns with alternative column names."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1],
            "superheavy_lox_fullness": [100, 90],
            "starship.ch4_fullness": [100, 95]
        })
        
        df_prepared = prepare_fuel_data_columns(df.copy())
        
        # Check that columns were renamed
        assert 'superheavy.fuel.lox.fullness' in df_prepared.columns
        assert 'starship.fuel.ch4.fullness' in df_prepared.columns
        
        # Check that values were copied correctly
        assert df_prepared['superheavy.fuel.lox.fullness'].iloc[0] == 100
        assert df_prepared['superheavy.fuel.lox.fullness'].iloc[1] == 90
        assert df_prepared['starship.fuel.ch4.fullness'].iloc[0] == 100
        assert df_prepared['starship.fuel.ch4.fullness'].iloc[1] == 95


class TestNormalizeFuelLevels:
    """Tests for normalize_fuel_levels function."""
    
    def test_normalize_fuel_levels(self, sample_fuel_dataframe):
        """Test basic fuel level normalization."""
        df_normalized = normalize_fuel_levels(sample_fuel_dataframe.copy())
        
        # First time point - levels equal, no normalization needed
        assert df_normalized['superheavy.fuel.lox.fullness'].iloc[0] == 100
        assert df_normalized['superheavy.fuel.ch4.fullness'].iloc[0] == 100
        
        # At 200s, levels differ by 40% - should be normalized to min value after 200s
        assert df_normalized['superheavy.fuel.lox.fullness'].iloc[4] == 20
        assert df_normalized['superheavy.fuel.ch4.fullness'].iloc[4] == 20
        
        # At 300s, levels differ by 35% - should be normalized to min value after 200s
        assert df_normalized['superheavy.fuel.lox.fullness'].iloc[6] == 5
        assert df_normalized['superheavy.fuel.ch4.fullness'].iloc[6] == 5
        
        # At 200s for starship, levels differ by 35% - should be normalized to min value after 200s
        assert df_normalized['starship.fuel.lox.fullness'].iloc[4] == 45
        assert df_normalized['starship.fuel.ch4.fullness'].iloc[4] == 45
    
    def test_normalize_fuel_levels_missing_columns(self):
        """Test with missing fuel columns."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1],
            "starship.speed": [0, 10]
        })
        
        # Should return DataFrame unchanged (no errors)
        df_normalized = normalize_fuel_levels(df.copy())
        assert df_normalized.equals(df)


class TestLoadAndCleanData:
    """Tests for load_and_clean_data function."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('plot.data_processing.clean_dataframe')
    @patch('plot.data_processing.process_engine_data')
    @patch('plot.data_processing.prepare_fuel_data_columns')
    @patch('plot.data_processing.normalize_fuel_levels')
    def test_load_and_clean_data_valid(self, mock_normalize, mock_prepare, 
                                       mock_process, mock_clean, mock_json_load, mock_file_open):
        """Test loading and cleaning valid data."""
        # Setup mocks
        mock_json_load.return_value = [
            {
                "frame_number": 0,
                "superheavy": {"speed": 0, "altitude": 0},
                "starship": {"speed": 0, "altitude": 0},
                "time": "00:00:00",
                "real_time_seconds": 0
            }
        ]
        
        # Mock the normalized DataFrame that should be returned
        mock_df = pd.DataFrame({
            "real_time_seconds": [0],
            "superheavy.speed": [0],
            "superheavy.altitude": [0],
            "starship.speed": [0],
            "starship.altitude": [0]
        })
        
        # Setup chain of mock returns
        mock_process.return_value = mock_df
        mock_clean.return_value = mock_df
        mock_prepare.return_value = mock_df
        mock_normalize.return_value = mock_df
        
        # Call function
        result = load_and_clean_data("test.json")
        
        # Verify results
        mock_file_open.assert_called_once_with("test.json", "r")
        mock_json_load.assert_called_once()
        mock_process.assert_called_once()
        mock_clean.assert_called_once()
        mock_prepare.assert_called_once()
        mock_normalize.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    
    @patch('builtins.open', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_load_and_clean_data_invalid_json(self, mock_file_open):
        """Test loading invalid JSON data."""
        result = load_and_clean_data("invalid.json")
        assert result.empty
    
    @patch('builtins.open', side_effect=Exception("Test error"))
    def test_load_and_clean_data_exception(self, mock_file_open):
        """Test handling of general exceptions."""
        result = load_and_clean_data("error.json")
        assert result.empty


class TestComputeAcceleration:
    """Tests for compute_acceleration function."""
    
    def test_compute_acceleration_basic(self):
        """Test basic acceleration computation."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1, 2, 3, 4],
            "speed": [0, 10, 20, 30, 40]  # Constant acceleration of 10 km/h per second
        })
        
        acceleration = compute_acceleration(df, "speed", frame_distance=1)
        
        # Convert km/h to m/s for expected values: 10 km/h = 2.78 m/s
        # Acceleration should be 2.78 m/s per second = 2.78 m/s²
        expected_acceleration = 10 * (1000 / 3600)  # 10 km/h = 2.78 m/s
        
        # Check first few values (last will be NaN due to frame_distance)
        assert acceleration.iloc[0] == pytest.approx(expected_acceleration)
        assert acceleration.iloc[1] == pytest.approx(expected_acceleration)
        assert acceleration.iloc[2] == pytest.approx(expected_acceleration)
        assert acceleration.iloc[3] == pytest.approx(expected_acceleration)
        assert pd.isna(acceleration.iloc[4])  # Last value should be NaN
    
    def test_compute_acceleration_with_larger_frame_distance(self):
        """Test acceleration with larger frame distance."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "speed": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 10 km/h per second
        })
        
        acceleration = compute_acceleration(df, "speed", frame_distance=3)
        
        # Convert km/h to m/s for expected values: 10 km/h = 2.78 m/s
        expected_acceleration = 10 * (1000 / 3600)  # Still 2.78 m/s²
        
        # Check a few values
        assert acceleration.iloc[0] == pytest.approx(expected_acceleration)
        assert acceleration.iloc[5] == pytest.approx(expected_acceleration)
        assert pd.isna(acceleration.iloc[7])  # Last "frame_distance" values should be NaN
        assert pd.isna(acceleration.iloc[8])
        assert pd.isna(acceleration.iloc[9])
    
    def test_compute_acceleration_with_missing_data(self):
        """Test acceleration with missing data."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1, 2, 3, 4],
            "speed": [0, None, 20, 30, 40]  # Missing value
        })
        
        acceleration = compute_acceleration(df, "speed", frame_distance=1)
        
        # Check values
        assert pd.isna(acceleration.iloc[0])  # NaN due to missing next value
        assert pd.isna(acceleration.iloc[1])  # NaN due to missing current value
        assert acceleration.iloc[2] == pytest.approx(10 * (1000 / 3600))
        assert acceleration.iloc[3] == pytest.approx(10 * (1000 / 3600))
    
    def test_compute_acceleration_over_limit(self):
        """Test acceleration values over the maximum limit."""
        df = pd.DataFrame({
            "real_time_seconds": [0, 1, 2],
            "speed": [0, 400, 800]  # Very high acceleration, over 100 m/s²
        })
        
        acceleration = compute_acceleration(df, "speed", frame_distance=1, max_accel=100.0)
        
        # Check values - should be filtered out as over limit
        assert pd.isna(acceleration.iloc[0])
        assert pd.isna(acceleration.iloc[1])


class TestComputeGForce:
    """Tests for compute_g_force function."""
    
    def test_compute_g_force(self):
        """Test basic G-force computation."""
        # Create a Series with acceleration values in m/s²
        acceleration = pd.Series([0, 9.81, 19.62, -9.81])
        
        g_force = compute_g_force(acceleration)
        
        # Check conversion - 9.81 m/s² = 1g
        assert g_force.iloc[0] == pytest.approx(0)
        assert g_force.iloc[1] == pytest.approx(1.0)
        assert g_force.iloc[2] == pytest.approx(2.0)
        assert g_force.iloc[3] == pytest.approx(-1.0)
    
    def test_compute_g_force_with_constant(self):
        """Test G-force conversion factor."""
        acceleration = pd.Series([G_FORCE_CONVERSION * 3])
        
        g_force = compute_g_force(acceleration)
        
        assert g_force.iloc[0] == pytest.approx(3.0)
    
    def test_compute_g_force_with_nan(self):
        """Test G-force with NaN values."""
        acceleration = pd.Series([10.0, None, 20.0])
        
        g_force = compute_g_force(acceleration)
        
        assert g_force.iloc[0] == pytest.approx(10.0 / G_FORCE_CONVERSION)
        assert pd.isna(g_force.iloc[1])
        assert g_force.iloc[2] == pytest.approx(20.0 / G_FORCE_CONVERSION)
