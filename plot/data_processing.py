import os
import json
import pandas as pd
from tqdm import tqdm


def validate_json(data: list) -> bool:
    """
    Validate the structure of the JSON data.

    Args:
        data (list): The JSON data to validate.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    required_keys = {"frame_number", "superheavy",
                     "starship", "time", "real_time"}
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
    for column in ['starship_speed', 'superheavy_speed', 'starship_altitude', 'superheavy_altitude']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Step 2: Remove impossible values
    df['starship_speed'] = df['starship_speed'].clip(lower=0, upper=28000)
    df['superheavy_speed'] = df['superheavy_speed'].clip(lower=0, upper=6000)
    df['starship_altitude'] = df['starship_altitude'].clip(lower=0, upper=200)
    df['superheavy_altitude'] = df['superheavy_altitude'].clip(
        lower=0, upper=100)

    # Step 3: Detect abrupt changes
    df['starship_speed_diff'] = df['starship_speed'].diff().abs()
    df.loc[df['starship_speed_diff'] > 50, 'starship_speed'] = None
    df['superheavy_speed_diff'] = df['superheavy_speed'].diff().abs()
    df.loc[df['superheavy_speed_diff'] > 50, 'superheavy_speed'] = None
    df['starship_altitude_diff'] = df['starship_altitude'].diff().abs()
    df.loc[df['starship_altitude_diff'] > 1, 'starship_altitude'] = None
    df['superheavy_altitude_diff'] = df['superheavy_altitude'].diff().abs()
    df.loc[df['superheavy_altitude_diff'] > 1, 'superheavy_altitude'] = None

    return df


def load_and_clean_data(json_path: str) -> pd.DataFrame:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.drop(columns=["time"], inplace=True)
        for column in tqdm(["superheavy", "starship"], desc="Separating columns"):
            df[[f"{column}_speed", f"{column}_altitude"]
               ] = df[column].apply(pd.Series)
        df.drop(columns=["superheavy", "starship"], inplace=True)
        df.sort_values(by="real_time", inplace=True)
        df = clean_dataframe(df)
        return df
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


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
        time_diff = df['real_time'].iloc[i + frame_distance] - df['real_time'].iloc[i]
        
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
