import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .data_processing import load_and_clean_data, compute_acceleration
from utils import extract_launch_number

# Global plot parameters for analyze_results
ANALYZE_RESULTS_PLOT_PARAMS = [
    # Speed vs Time
    ('real_time', 'superheavy_speed', 'Speed of Superheavy Relative to Time',
     'sh_speed_vs_time_comparison.png', 'SH Speed'),
    ('real_time', 'starship_speed', 'Speed of Starship Relative to Time',
     'ss_speed_vs_time_comparison.png', 'SS Speed'),
    # Altitude vs Time
    ('real_time', 'superheavy_altitude', 'Altitude of Superheavy Relative to Time',
     'sh_altitude_vs_time_comparison.png', 'SH Altitude'),
    ('real_time', 'starship_altitude', 'Altitude of Starship Relative to Time',
     'ss_altitude_vs_time_comparison.png', 'SS Altitude'),
    # Altitude vs Speed
    ('superheavy_speed', 'superheavy_altitude', 'Altitude of Superheavy Relative to Speed',
     'sh_altitude_vs_speed_comparison.png', 'SH Altitude vs Speed'),
    ('starship_speed', 'starship_altitude', 'Altitude of Starship Relative to Speed',
     'ss_altitude_vs_speed_comparison.png', 'SS Altitude vs Speed'),
    # Acceleration vs Time
    ('real_time', 'superheavy_acceleration', 'Acceleration of Superheavy Relative to Time',
     'sh_acceleration_vs_time.png', 'SH Acceleration'),
    ('real_time', 'starship_acceleration', 'Acceleration of Starship Relative to Time',
     'ss_acceleration_vs_time.png', 'SS Acceleration')
]

# Global plot parameters for plot_two_launches
PLOT_TWO_LAUNCHES_PARAMS = [
    # Superheavy speeds
    ('real_time', 'superheavy_speed', 'Comparison of Superheavy Speeds',
     'comparison_superheavy_speeds.png'),
    # Starship speeds
    ('real_time', 'starship_speed', 'Comparison of Starship Speeds',
     'comparison_starship_speeds.png'),
    # Superheavy altitudes
    ('real_time', 'superheavy_altitude', 'Comparison of Superheavy Altitudes',
     'comparison_superheavy_altitudes.png'),
    # Starship altitudes
    ('real_time', 'starship_altitude', 'Comparison of Starship Altitudes',
     'comparison_starship_altitudes.png')
]


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, label: str, folder: str) -> None:
    """
    Create and save a scatter plot for the original and smoothed data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the original y-axis data.
        y_smoothed (str): The column name for the smoothed y-axis data.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        label (str): The label for the scatter plot.
        folder (str): The folder to save the graph in.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=df, label=f"{label}", s=5)
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/{filename}")
    plt.show()


def plot_flight_data(json_path: str) -> None:
    df = load_and_clean_data(json_path)
    if df.empty:
        return  # Exit if the DataFrame is empty due to JSON error

    # Set all Superheavy's data to None after 7 minutes and 30 seconds
    seven_minutes = 7 * 60 + 30  # 7 minutes and 30 seconds in seconds
    df.loc[df['real_time'] > seven_minutes, [
        'superheavy_speed', 'superheavy_altitude']] = None

    # Calculate acceleration
    df['superheavy_acceleration'] = compute_acceleration(
        df, 'superheavy_speed')
    df['starship_acceleration'] = compute_acceleration(df, 'starship_speed')

    # Determine the folder name based on the launch number
    launch_number = extract_launch_number(json_path)
    folder = f"results/launch_{launch_number}"

    for params in ANALYZE_RESULTS_PLOT_PARAMS:
        create_scatter_plot(df, *params, folder)


def plot_multiple_launches(df_list: list, x: str, y: str, title: str, filename: str, folder: str, labels: list[str]) -> None:
    """
    Plot a comparison of multiple dataframes.

    Args:
        df_list (list): List of dataframes to compare.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        labels (list): List of labels for the dataframes.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        folder (str): The folder to save the graph in.
    """
    plt.figure(figsize=(10, 6))
    for df, label in zip(df_list, labels):
        sns.scatterplot(x=x, y=y, data=df, label=f'{label}', s=10)
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}\\{filename}")
    plt.show()


def compare_multiple_launches(timeframe: int, *json_paths: str) -> None:
    """
    Plot multiple launches on the same plot with a specified timeframe.

    Args:
        timeframe (int): The timeframe in seconds to plot.
        *json_paths (str): Variable number of JSON file paths containing the results.
    """
    df_list = []
    labels = []
    
    timeframe = None if timeframe == -1 else timeframe

    for json_path in json_paths:
        df = load_and_clean_data(json_path)
        if df.empty:
            continue  # Skip if the DataFrame is empty due to JSON error
        df = df[df['real_time'] >= 0]
        if timeframe is not None:
            df = df[df['real_time'] <= timeframe]
        df_list.append(df)
        labels.append(f'launch {extract_launch_number(json_path)}')

    # Sort labels and create folder name
    labels.sort()
    folder_name = f"results\\compare_launches\\launches_{'_'.join(labels)}"
    os.makedirs(folder_name, exist_ok=True)

    for params in PLOT_TWO_LAUNCHES_PARAMS:
        plot_multiple_launches(df_list, *params, folder_name, labels)
