import json
import pandas as pd
from tqdm import tqdm
from constants import G_FORCE_CONVERSION


def validate_json(data: list) -> bool:
    """
    Validate the structure of the JSON data.

    Args:
        data (list): The JSON data to validate.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    required_keys = {"frame_number", "superheavy",
                     "starship", "time", "real_time_seconds"}
    for entry in data:
        if not required_keys.issubset(entry.keys()):
            return (False, entry)
    return (True, None)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Step 1: Ensure numeric values
    for column in ['starship.speed', 'superheavy.speed', 'starship.altitude', 'superheavy.altitude']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Step 2: Remove impossible values
    df['starship.speed'] = df['starship.speed'].clip(lower=0, upper=28000)
    df['superheavy.speed'] = df['superheavy.speed'].clip(lower=0, upper=6000)
    df['starship.altitude'] = df['starship.altitude'].clip(lower=0, upper=200)
    df['superheavy.altitude'] = df['superheavy.altitude'].clip(
        lower=0, upper=100)

    # Step 3: Detect abrupt changes
    df['starship.speed_diff'] = df['starship.speed'].diff().abs()
    df.loc[df['starship.speed_diff'] > 50, 'starship.speed'] = None
    df['superheavy.speed_diff'] = df['superheavy.speed'].diff().abs()
    df.loc[df['superheavy.speed_diff'] > 50, 'superheavy.speed'] = None
    df['starship.altitude_diff'] = df['starship.altitude'].diff().abs()
    df.loc[df['starship.altitude_diff'] > 1, 'starship.altitude'] = None
    df['superheavy.altitude_diff'] = df['superheavy.altitude'].diff().abs()
    df.loc[df['superheavy.altitude_diff'] > 1, 'superheavy.altitude'] = None

    return df


def process_engine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process engine data from the JSON and calculate number of active engines.
    
    Args:
        df (pd.DataFrame): DataFrame with raw engine data
        
    Returns:
        pd.DataFrame: DataFrame with processed engine data
    """
    # Create columns for engine counts
    df['superheavy_central_active'] = 0
    df['superheavy_central_total'] = 3
    df['superheavy_inner_active'] = 0  
    df['superheavy_inner_total'] = 10
    df['superheavy_outer_active'] = 0
    df['superheavy_outer_total'] = 20
    df['superheavy_all_active'] = 0
    df['superheavy_all_total'] = 33  # 3 + 10 + 20 = 33 engines total
    
    df['starship_rearth_active'] = 0
    df['starship_rearth_total'] = 3
    df['starship_rvac_active'] = 0
    df['starship_rvac_total'] = 3
    df['starship_all_active'] = 0
    df['starship_all_total'] = 6  # 3 + 3 = 6 engines total
    
    # Process engine data using the correct column structure
    try:
        # Check if the expected columns exist
        engine_columns = {
            'superheavy.engines.central_stack': 'superheavy_central_active',
            'superheavy.engines.inner_ring': 'superheavy_inner_active',
            'superheavy.engines.outer_ring': 'superheavy_outer_active',
            'starship.engines.rearth': 'starship_rearth_active',
            'starship.engines.rvac': 'starship_rvac_active'
        }
        
        for src_col, dest_col in tqdm(engine_columns.items(), desc="Processing engine columns"):
            if src_col in df.columns:
                # Sum the boolean values in each row to get active engine count
                # Each row contains a list of boolean values (True = engine active)
                df[dest_col] = df[src_col].apply(
                    lambda x: sum(1 for engine in x if engine) if isinstance(x, list) else 0
                )
                
        # Calculate total active engines
        df['superheavy_all_active'] = (
            df['superheavy_central_active'] + 
            df['superheavy_inner_active'] + 
            df['superheavy_outer_active']
        )
        
        df['starship_all_active'] = (
            df['starship_rearth_active'] + 
            df['starship_rvac_active']
        )
            
        # Drop the original engine columns as they're now processed
        for col in engine_columns.keys():
            if col in df.columns:
                df = df.drop(columns=[col])
                
    except Exception as e:
        print(f"Error processing engine data: {e}")
    
    return df


def load_and_clean_data(json_path: str) -> pd.DataFrame:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Use json_normalize with sep='.' to flatten nested dictionaries with dot notation
        df = pd.json_normalize(data)
        
        # Process engine data
        df = process_engine_data(df)
        
        # Drop time column as we're using real_time_seconds
        if 'time' in df.columns:
            df.drop(columns=["time"], inplace=True)
        
        # Clean velocity and altitude columns
        # Assuming they're now in the format 'superheavy.speed', 'starship.speed', etc.
        # Check if we need to rename columns
        if 'superheavy.speed' not in df.columns and 'superheavy.speed' not in df.columns:
            # Extract speed and altitude from nested dictionaries if needed
            for column in tqdm(["superheavy", "starship"], desc="Separating columns"):
                if column in df.columns:
                    df[[f"{column}.speed", f"{column}.altitude"]] = df[column].apply(pd.Series)
                    df.drop(columns=[column], inplace=True)
                    
        # Sort by time
        df.sort_values(by="real_time_seconds", inplace=True)
        
        # Clean data
        df = clean_dataframe(df)
        
        return df
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def compute_acceleration(df: pd.DataFrame, speed_column: str, frame_distance: int = 30, max_accel: float = 100.0) -> pd.Series:
    """
    Calculate acceleration from speed data using a fixed frame distance.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        speed_column (str): The column name for the speed data.
        frame_distance (int): Number of frames to look ahead for calculating acceleration.
        max_accel (float): Maximum allowed acceleration in m/s². Values above this will be set to None.

    Returns:
        pd.Series: The calculated acceleration.
    """
    # Create a series for storing accelerations
    acceleration = pd.Series(index=df.index, dtype=float)
    
    # Convert speed from km/h to m/s
    speed_m_per_s = df[speed_column] * (1000 / 3600)
    
    # Loop through the dataframe with a frame-distance offset
    for i in range(len(df) - frame_distance):
        if pd.isna(speed_m_per_s.iloc[i]) or pd.isna(speed_m_per_s.iloc[i + frame_distance]):
            acceleration.iloc[i] = None
            continue
            
        # Calculate speed difference over the frame distance
        speed_diff = speed_m_per_s.iloc[i + frame_distance] - speed_m_per_s.iloc[i]
        
        # Calculate time difference over the frame distance
        time_diff = df['real_time_seconds'].iloc[i + frame_distance] - df['real_time_seconds'].iloc[i]
        
        # Calculate acceleration if time difference is valid
        if time_diff > 0:
            accel_value = speed_diff / time_diff
            # Filter out unrealistic acceleration values
            if abs(accel_value) > max_accel:
                acceleration.iloc[i] = None
            else:
                acceleration.iloc[i] = accel_value
        else:
            acceleration.iloc[i] = None
    
    # The last frame_distance frames will have NaN acceleration
    return acceleration


def compute_g_force(acceleration_ms2: pd.Series) -> pd.Series:
    """
    Convert acceleration in m/s² to G-forces.
    
    Args:
        acceleration_ms2 (pd.Series): Acceleration values in m/s²
        
    Returns:
        pd.Series: G-force values (1G = 9.81 m/s²)
    """
    return acceleration_ms2 / G_FORCE_CONVERSION
