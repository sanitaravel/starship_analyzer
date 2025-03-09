import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from .data_processing import load_and_clean_data, compute_acceleration
from utils import extract_launch_number

# Global constant for converting m/s² to G-forces
G_FORCE_CONVERSION = 9.81  # 1G = 9.81 m/s²

# Update plot parameter titles to reflect the new calculation method
ANALYZE_RESULTS_PLOT_PARAMS = [
    # Speed vs Time (update with axis labels)
    ('real_time', 'superheavy_speed', 'Speed of Superheavy Relative to Time',
     'sh_speed_vs_time_comparison.png', 'SH Speed', 'Real Time (s)', 'Speed (km/h)'),
    ('real_time', 'starship_speed', 'Speed of Starship Relative to Time',
     'ss_speed_vs_time_comparison.png', 'SS Speed', 'Real Time (s)', 'Speed (km/h)'),
    # Altitude vs Time
    ('real_time', 'superheavy_altitude', 'Altitude of Superheavy Relative to Time',
     'sh_altitude_vs_time_comparison.png', 'SH Altitude', 'Real Time (s)', 'Altitude (km)'),
    ('real_time', 'starship_altitude', 'Altitude of Starship Relative to Time',
     'ss_altitude_vs_time_comparison.png', 'SS Altitude', 'Real Time (s)', 'Altitude (km)'),
    # # 60-Point MA of Acceleration vs Time
    # ('real_time', 'superheavy_acceleration_ma', '60-Point MA of Superheavy Acceleration Relative to Time',
    #  'sh_acceleration_ma_vs_time.png', 'SH Acceleration (MA)', 'Real Time (s)', 'Acceleration (m/s²)'),
    # ('real_time', 'starship_acceleration_ma', '60-Point MA of Starship Acceleration Relative to Time',
    #  'ss_acceleration_ma_vs_time.png', 'SS Acceleration (MA)', 'Real Time (s)', 'Acceleration (m/s²)'),
    # # 60-Point MA of G-Force vs Time (new)
    # ('real_time', 'superheavy_g_force_ma', '60-Point MA of Superheavy G-Force Relative to Time',
    #  'sh_g_force_ma_vs_time.png', 'SH G-Force (MA)', 'Real Time (s)', 'G-Force (g)'),
    # ('real_time', 'starship_g_force_ma', '60-Point MA of Starship G-Force Relative to Time',
    #  'ss_g_force_ma_vs_time.png', 'SS G-Force (MA)', 'Real Time (s)', 'G-Force (g)'),
    # 10-Frame Distance Acceleration vs Time (updated titles)
    ('real_time', 'superheavy_acceleration', 'Superheavy Acceleration (10-Frame Distance)',
     'sh_acceleration_vs_time.png', 'SH Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    ('real_time', 'starship_acceleration', 'Starship Acceleration (10-Frame Distance)',
     'ss_acceleration_vs_time.png', 'SS Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    # G-Force vs Time (new)
    ('real_time', 'superheavy_g_force', 'Superheavy G-Force (10-Frame Distance)',
     'sh_g_force_vs_time.png', 'SH G-Force', 'Real Time (s)', 'G-Force (g)'),
    ('real_time', 'starship_g_force', 'Starship G-Force (10-Frame Distance)',
     'ss_g_force_vs_time.png', 'SS G-Force', 'Real Time (s)', 'G-Force (g)'),
]

# Updated global plot parameters for plot_multiple_launches with axis names.
PLOT_MULTIPLE_LAUNCHES_PARAMS = [
    ('real_time', 'superheavy_speed', 'Comparison of Superheavy Speeds',
     'comparison_superheavy_speeds.png', 'Real Time (s)', 'Superheavy Speed (km/h)'),
    ('real_time', 'starship_speed', 'Comparison of Starship Speeds',
     'comparison_starship_speeds.png', 'Real Time (s)', 'Starship Speed (km/h)'),
    ('real_time', 'superheavy_altitude', 'Comparison of Superheavy Altitudes',
     'comparison_superheavy_altitudes.png', 'Real Time (s)', 'Superheavy Altitude (km)'),
    ('real_time', 'starship_altitude', 'Comparison of Starship Altitudes',
     'comparison_starship_altitudes.png', 'Real Time (s)', 'Starship Altitude (km)'),
    # ('real_time', 'superheavy_acceleration_ma', 'Comparison of Superheavy Accelerations',
    #  'comparison_superheavy_accelerations.png', 'Real Time (s)', 'Superheavy Acceleration (m/s²)'),
    # ('real_time', 'starship_acceleration_ma', 'Comparison of Starship Accelerations',
    #  'comparison_starship_accelerations.png', 'Real Time (s)', 'Starship Acceleration (m/s²)'),
    # ('real_time', 'superheavy_g_force_ma', 'Comparison of Superheavy G-Forces',
    #  'comparison_superheavy_g_forces.png', 'Real Time (s)', 'Superheavy G-Force (g)'),
    # ('real_time', 'starship_g_force_ma', 'Comparison of Starship G-Forces',
    #  'comparison_starship_g_forces.png', 'Real Time (s)', 'Starship G-Force (g)'),
    ('real_time', 'superheavy_acceleration', 'Comparison of Superheavy Accelerations',
     'comparison_superheavy_accelerations.png', 'Real Time (s)', 'Superheavy Acceleration (m/s²)'),
    ('real_time', 'starship_acceleration', 'Comparison of Starship Accelerations',
     'comparison_starship_accelerations.png', 'Real Time (s)', 'Starship Acceleration (m/s²)'),
    ('real_time', 'superheavy_g_force', 'Comparison of Superheavy G-Forces',
     'comparison_superheavy_g_forces.png', 'Real Time (s)', 'Superheavy G-Force (g)'),
    ('real_time', 'starship_g_force', 'Comparison of Starship G-Forces',
     'comparison_starship_g_forces.png', 'Real Time (s)', 'Starship G-Force (g)'),
]


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, label: str, x_axis: str, y_axis: str, folder: str, show_figures: bool) -> None:
    """
    Create and save a scatter plot for the original and smoothed data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the original y-axis data.
        title (str): The title of the graph.
        filename (str): The filename to save the graph as.
        label (str): The label for the scatter plot.
        x_axis (str): The label for the x-axis.
        y_axis (str): The label for the y-axis.
        folder (str): The folder to save the graph in.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(16, 9))
    
    # Create scatter plot with increased transparency
    sns.scatterplot(x=x, y=y, data=df, label=f"{label} (Raw Data)", s=4, alpha=0.2)
    
    # Add trendline for acceleration and g-force plots
    if 'acceleration' in y or 'g_force' in y:
        # Only use non-null values for the trendline
        valid_data = df[[x, y]].dropna()
        
        if len(valid_data) > 10:  # Only add trendline if we have enough data points
            # Use LOWESS to create a smooth trendline (adjust frac for different smoothing levels)
            z = lowess(valid_data[y], valid_data[x], frac=0.01)
            plt.plot(z[:, 0], z[:, 1], 'r-', linewidth=2, label=f"{label} (Trend)")
    
    plt.xlabel(x_axis if x_axis else x.capitalize())
    plt.ylabel(y_axis if y_axis else y.capitalize())
    plt.title(title)
    
    # Create a custom legend with more visible scatter points
    handles, labels = plt.gca().get_legend_handles_labels()
    for handle in handles:
        if not isinstance(handle, plt.Line2D):  # This is a scatter point
            handle.set_alpha(1.0)     # Make it fully opaque
    
    plt.legend(handles=handles, labels=labels)
    plt.grid(True)
    plt.savefig(f"{folder}/{filename}")
    if show_figures:
        plt.show()
    else:
        plt.close()


def plot_flight_data(json_path: str, start_time: int = 0, end_time: int = -1, show_figures:bool=True) -> None:
    """
    Plot flight data from a JSON file with optional time window limits.

    Args:
        json_path (str): Path to the JSON file containing the flight data.
        start_time (int): Minimum time in seconds to include in plots. Default is 0.
        end_time (int): Maximum time in seconds to include in plots. Use -1 for all data.
    """
    df = load_and_clean_data(json_path)
    if df.empty:
        return  # Exit if the DataFrame is empty due to JSON error

    # Filter data by time window
    df = df[df['real_time'] >= start_time]
    if end_time != -1:
        df = df[df['real_time'] <= end_time]

    # Set all Superheavy's data to None after 7 minutes and 30 seconds
    seven_minutes = 7 * 60 + 30  # 7 minutes and 30 seconds in seconds
    df.loc[df['real_time'] > seven_minutes, [
        'superheavy_speed', 'superheavy_altitude']] = None

    # Calculate acceleration using 10-frame distance instead of rolling window
    df['superheavy_acceleration'] = compute_acceleration(df, 'superheavy_speed', frame_distance=10)
    df['starship_acceleration'] = compute_acceleration(df, 'starship_speed', frame_distance=10)
    
    # Calculate G-forces
    df['superheavy_g_force'] = df['superheavy_acceleration'] / G_FORCE_CONVERSION
    df['starship_g_force'] = df['starship_acceleration'] / G_FORCE_CONVERSION

    # Determine the folder name based on the launch number
    launch_number = extract_launch_number(json_path)
    folder = f"results/launch_{launch_number}"

    # Updated plotting: if tuple has 7 items, pass x_axis and y_axis labels.
    for params in ANALYZE_RESULTS_PLOT_PARAMS:
        if len(params) == 5:
            create_scatter_plot(df, *params, folder, show_figures)
        else:
            # Unpack: x, y, title, filename, label, x_axis, y_axis.
            create_scatter_plot(df, *params, folder, show_figures)


def plot_multiple_launches(df_list: list, x: str, y: str, title: str, filename: str, folder: str,
                           labels: list[str], x_axis: str = None, y_axis: str = None) -> None:
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
        x_axis (str): The label for the x-axis.
        y_axis (str): The label for the y-axis.
    """
    plt.figure(figsize=(16, 9))
    
    # Plot each dataset with both scatter points and trendline
    for i, (df, label) in enumerate(zip(df_list, labels)):
        # Get a unique color from the color cycle
        color = plt.cm.tab10(i)
        
        
        
        # Add trendline for acceleration and g-force plots
        if 'acceleration' in y or 'g_force' in y:
            # Create scatter plot with increased transparency
            sns.scatterplot(x=x, y=y, data=df, label=f"{label} (Raw)", s=4, alpha=0.2, color=color)
            
            # Only use non-null values for the trendline
            valid_data = df[[x, y]].dropna()
            
            if len(valid_data) > 10:  # Only add trendline if we have enough data points
                # Use LOWESS to create a smooth trendline
                z = lowess(valid_data[y], valid_data[x], frac=0.01)
                plt.plot(z[:, 0], z[:, 1], '-', linewidth=2, label=f"{label} (Trend)", color=color)
        else:
            # Create scatter plot with increased transparency
            sns.scatterplot(x=x, y=y, data=df, label=f"{label} (Raw)", s=4, alpha=0.5, color=color)
    
    plt.xlabel(x_axis if x_axis else x.capitalize())
    plt.ylabel(y_axis if y_axis else y.capitalize())
    plt.title(title)
    
    # Create a custom legend with more visible scatter points
    handles, labels = plt.gca().get_legend_handles_labels()
    for handle in handles:
        if not isinstance(handle, plt.Line2D):  # This is a scatter point
            handle.set_rasterized(8)  # Make the marker bigger in the legend
            handle.set_alpha(1.0)     # Make it fully opaque
    
    plt.legend(handles=handles, labels=labels)
    plt.grid(True)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}\\{filename}")
    plt.show()


def compare_multiple_launches(start_time: int, end_time: int, *json_paths: str) -> None:
    """
    Plot multiple launches on the same plot with a specified time window.

    Args:
        start_time (int): Minimum time in seconds to include in plots.
        end_time (int): Maximum time in seconds to include in plots. Use -1 for all data.
        *json_paths (str): Variable number of JSON file paths containing the results.
    """
    df_list = []
    labels = []
    
    for json_path in json_paths:
        df = load_and_clean_data(json_path)
        if df.empty:
            continue  # Skip if the DataFrame is empty due to JSON error
        
        # Filter by time window
        df = df[df['real_time'] >= start_time]
        if end_time != -1:
            df = df[df['real_time'] <= end_time]
            
        # Calculate acceleration using 10-frame distance
        df['superheavy_acceleration'] = compute_acceleration(df, 'superheavy_speed', frame_distance=10)
        df['starship_acceleration'] = compute_acceleration(df, 'starship_speed', frame_distance=10)
        
        # Calculate G-forces
        df['superheavy_g_force'] = df['superheavy_acceleration'] / G_FORCE_CONVERSION
        df['starship_g_force'] = df['starship_acceleration'] / G_FORCE_CONVERSION
        
        df_list.append(df)
        labels.append(f'launch {extract_launch_number(json_path)}')

    # Sort labels and create folder name
    labels.sort()
    folder_name = f"results\\compare_launches\\launches_{'_'.join(labels)}"
    os.makedirs(folder_name, exist_ok=True)

    for params in PLOT_MULTIPLE_LAUNCHES_PARAMS:
        if len(params) == 4:
            plot_multiple_launches(df_list, *params, folder_name, labels)
        else:
            # Unpack: x, y, title, filename, x_axis, y_axis.
            x, y, title, filename, x_axis, y_axis = params
            plot_multiple_launches(df_list, x, y, title, filename, folder_name, labels, x_axis, y_axis)
