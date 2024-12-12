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

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change obvious outliers to None in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Define thresholds for outliers
    sh_speed_threshold = 6000  # Example threshold for Superheavy speed 
    sh_altitude_threshold = 100  # Example threshold for Superheavy altitude
    ss_speed_threshold = 30000  # Example threshold for Starship speed
    ss_altitude_threshold = 200  # Example threshold for Starship altitude

    # Change outliers to None
    df.loc[df['superheavy_speed'] > sh_speed_threshold, 'superheavy_speed'] = None
    df.loc[df['superheavy_altitude'] > sh_altitude_threshold, 'superheavy_altitude'] = None
    df.loc[df['starship_speed'] > ss_speed_threshold, 'starship_speed'] = None
    df.loc[df['starship_altitude'] > ss_altitude_threshold, 'starship_altitude'] = None

    return df

def delete_outliers_within_window(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Delete outliers within a rolling window from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        window_size (int): The window size for the rolling operations.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    for column in ['superheavy_speed', 'superheavy_altitude', 'starship_speed', 'starship_altitude']:
        rolling_median = df[column].rolling(window=window_size, center=True).median()
        rolling_std = df[column].rolling(window=window_size, center=True).std()
        df = df[(df[column] >= (rolling_median - 2 * rolling_std)) & (df[column] <= (rolling_median + 2 * rolling_std))]
    return df

def create_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, label: str) -> None:
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
    sns.scatterplot(x=x, y=y, data=df, label=label)
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
        
        # # # Remove obvious outliers
        df = remove_outliers(df)
        
        delete_outliers_within_window(df, window_size=10000)
        
        # Set all Superheavy's data to None after 7 minutes
        seven_minutes = 7 * 60  # 7 minutes in seconds
        df.loc[df['real_time'] > seven_minutes, ['superheavy_speed', 'superheavy_altitude']] = None
        
        print(df)
        
        # Create scatter plots
        create_scatter_plot(df, 'real_time', 'superheavy_speed', 'Speed of Superheavy Relative to Real Time', 'sh_speed_vs_time.png', 'SH Speed')
        create_scatter_plot(df, 'real_time', 'starship_speed', 'Speed of Starship Relative to Real Time', 'ss_speed_vs_time.png', 'SS Speed')
        create_scatter_plot(df, 'real_time', 'superheavy_altitude', 'Altitude of Superheavy Relative to Real Time', 'sh_altitude_vs_time.png', 'SH Altitude')
        create_scatter_plot(df, 'real_time', 'starship_altitude', 'Altitude of Starship Relative to Real Time', 'ss_altitude_vs_time.png', 'SS Altitude')
        
        # Create height vs speed plots
        create_height_vs_speed_plot(df, 'superheavy_speed', 'superheavy_altitude', 'Altitude of Superheavy Relative to Speed', 'sh_altitude_vs_speed.png', 'SH Altitude vs Speed')
        create_height_vs_speed_plot(df, 'starship_speed', 'starship_altitude', 'Altitude of Starship Relative to Speed', 'ss_altitude_vs_speed.png', 'SS Altitude vs Speed')
        
    except json.JSONDecodeError:
        print("Invalid JSON format.")