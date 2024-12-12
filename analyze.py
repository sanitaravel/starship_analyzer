from datetime import date
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # Add this import
import os  # Add this import

def validate_json_structure(data: list) -> bool:
    """
    Validate the structure of the JSON data.

    Args:
        data (list): The JSON data to validate.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    required_keys = {"frame_number", "superheavy", "starship", "time", "real_time"}
    for entry in data:
        if not required_keys.issubset(entry.keys()):
            return (False, entry)
    return (True, None)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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
    df['superheavy_altitude'] = df['superheavy_altitude'].clip(lower=0, upper=100)

    # Step 3: Detect abrupt changes
    df['starship_speed_diff'] = df['starship_speed'].diff().abs()
    df.loc[df['starship_speed_diff'] > 10, 'starship_speed'] = None
    df['superheavy_speed_diff'] = df['superheavy_speed'].diff().abs()
    df.loc[df['superheavy_speed_diff'] > 10, 'superheavy_speed'] = None
    df['starship_altitude_diff'] = df['starship_altitude'].diff().abs()
    df.loc[df['starship_altitude_diff'] > 1, 'starship_altitude'] = None
    df['superheavy_altitude_diff'] = df['superheavy_altitude'].diff().abs()
    df.loc[df['superheavy_altitude_diff'] > 1, 'superheavy_altitude'] = None

    # # Step 4: Interpolate missing data
    # df['starship_speed'] = df['starship_speed'].interpolate(method='linear', limit_direction='both')
    # df['superheavy_speed'] = df['superheavy_speed'].interpolate(method='linear', limit_direction='both')
    # df['starship_altitude'] = df['starship_altitude'].interpolate(method='linear', limit_direction='both')
    # df['superheavy_altitude'] = df['superheavy_altitude'].interpolate(method='linear', limit_direction='both')

    # # Step 5: Smooth the date
    # df['starship_speed_smoothed'] = df['starship_speed'].rolling(window=2500, min_periods=1).mean()
    # df['superheavy_speed_smoothed'] = df['superheavy_speed'].rolling(window=2500, min_periods=1).mean()
    # df['starship_altitude_smoothed'] = df['starship_altitude'].rolling(window=2500, min_periods=1).mean()
    # df['superheavy_altitude_smoothed'] = df['superheavy_altitude'].rolling(window=2500, min_periods=1).mean()

    return df

def create_scatter_plot(df: pd.DataFrame, x: str, y: str, y_smoothed: str, title: str, filename: str, label: str) -> None:
    """
    Create and save a scatter plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        label (str): The label for the scatter plot.
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=df, label=f"Original {label}")
    sns.scatterplot(x=x, y=y_smoothed, data=df, label=f"Smoothed {label}")
    plt.xlabel('Real Time (s)')
    plt.ylabel(y.capitalize())
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.show()

def create_height_vs_speed_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, label: str) -> None:
    """
    Create and save a scatter plot for height relative to speed.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        label (str): The label for the scatter plot.
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=df, label=label)
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.show()

def create_comparison_plot(df: pd.DataFrame, x: str, y: str, y_smoothed: str, title: str, filename: str, label: str) -> None:
    """
    Create and save a comparison scatter plot for the original and smoothed data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the original y-axis data.
        y_smoothed (str): The column name for the smoothed y-axis data.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        label (str): The label for the scatter plot.
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=df, label=f"Original {label}")
    sns.scatterplot(x=x, y=y_smoothed, data=df, label=f"Smoothed {label}")
    plt.xlabel('Real Time (s)')
    plt.ylabel(y.capitalize())
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.show()

def analyze_results(json_path: str) -> None:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        pd.set_option('display.max_columns', None)  # Add this line to display all columns
        result, entry = validate_json_structure(data)
        if not result:
            print("Invalid JSON structure.", entry)
            return
        df = pd.DataFrame(data)
        df.drop(columns=["time"], inplace=True)  # Drop the "time" column
        
        # Split "superheavy" and "starship" columns into separate "speed" and "altitude" columns with progress bar
        for column in tqdm(["superheavy", "starship"], desc="Separating columns"):
            df[[f"{column}_speed", f"{column}_altitude"]] = df[column].apply(pd.Series) 
        
        # Drop the original "superheavy" and "starship" columns
        df.drop(columns=["superheavy", "starship"], inplace=True)
        
        # Sort the DataFrame by real time
        df.sort_values(by="real_time", inplace=True)
        
        # Clean the data
        df = clean_data(df)
        
        # Set all Superheavy's data to None after 7 minutes
        seven_minutes = 7 * 60  # 7 minutes in seconds
        df.loc[df['real_time'] > seven_minutes, ['superheavy_speed', 'superheavy_altitude']] = None
        
        print(df)
        
        # Create comparison scatter plots
        create_comparison_plot(df, 'real_time', 'superheavy_speed', 'superheavy_speed_smoothed', 'Speed of Superheavy Relative to Real Time', 'sh_speed_vs_time_comparison.png', 'SH Speed')
        create_comparison_plot(df, 'real_time', 'starship_speed', 'starship_speed_smoothed', 'Speed of Starship Relative to Real Time', 'ss_speed_vs_time_comparison.png', 'SS Speed')
        create_comparison_plot(df, 'real_time', 'superheavy_altitude', 'superheavy_altitude_smoothed', 'Altitude of Superheavy Relative to Real Time', 'sh_altitude_vs_time_comparison.png', 'SH Altitude')
        create_comparison_plot(df, 'real_time', 'starship_altitude', 'starship_altitude_smoothed', 'Altitude of Starship Relative to Real Time', 'ss_altitude_vs_time_comparison.png', 'SS Altitude')
        
        # Create height vs speed comparison plots
        create_comparison_plot(df, 'superheavy_speed', 'superheavy_altitude', 'superheavy_altitude_smoothed', 'Altitude of Superheavy Relative to Speed', 'sh_altitude_vs_speed_comparison.png', 'SH Altitude vs Speed')
        create_comparison_plot(df, 'starship_speed', 'starship_altitude', 'starship_altitude_smoothed', 'Altitude of Starship Relative to Speed', 'ss_altitude_vs_speed_comparison.png', 'SS Altitude vs Speed')
        
    except json.JSONDecodeError:
        print("Invalid JSON format.")