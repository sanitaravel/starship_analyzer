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

# Add new plot parameters for engine visualization - using active counts instead of percentages
ENGINE_PLOT_PARAMS = [
    # Superheavy engines
    ('real_time', 'superheavy_central_active', 'Superheavy Central Stack Engines',
     'sh_central_engines.png', 'Central Stack', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'superheavy_inner_active', 'Superheavy Inner Ring Engines',
     'sh_inner_engines.png', 'Inner Ring', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'superheavy_outer_active', 'Superheavy Outer Ring Engines',
     'sh_outer_engines.png', 'Outer Ring', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'superheavy_all_active', 'All Superheavy Engines',
     'sh_all_engines.png', 'All Engines', 'Real Time (s)', 'Active Engines (count)'),
     
    # Starship engines
    ('real_time', 'starship_rearth_active', 'Starship Raptor Earth Engines',
     'ss_rearth_engines.png', 'Raptor Earth', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'starship_rvac_active', 'Starship Raptor Vacuum Engines', 
     'ss_rvac_engines.png', 'Raptor Vacuum', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'starship_all_active', 'All Starship Engines',
     'ss_all_engines.png', 'All Engines', 'Real Time (s)', 'Active Engines (count)')
]

# Update plot parameter titles to reflect the new calculation method
ANALYZE_RESULTS_PLOT_PARAMS = [
    # Speed vs Time (update with axis labels)
    ('real_time', 'superheavy.speed', 'Speed of Superheavy Relative to Time',
     'sh.speed_vs_time_comparison.png', 'SH Speed', 'Real Time (s)', 'Speed (km/h)'),
    ('real_time', 'starship.speed', 'Speed of Starship Relative to Time',
     'ss.speed_vs_time_comparison.png', 'SS Speed', 'Real Time (s)', 'Speed (km/h)'),
    # Altitude vs Time
    ('real_time', 'superheavy.altitude', 'Altitude of Superheavy Relative to Time',
     'sh.altitude_vs_time_comparison.png', 'SH Altitude', 'Real Time (s)', 'Altitude (km)'),
    ('real_time', 'starship.altitude', 'Altitude of Starship Relative to Time',
     'ss.altitude_vs_time_comparison.png', 'SS Altitude', 'Real Time (s)', 'Altitude (km)'),
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
    ('real_time', 'superheavy_acceleration', 'Superheavy Acceleration (30-Frame Distance)',
     'sh_acceleration_vs_time.png', 'SH Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    ('real_time', 'starship_acceleration', 'Starship Acceleration (10-Frame Distance)',
     'ss_acceleration_vs_time.png', 'SS Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    # G-Force vs Time (new)
    ('real_time', 'superheavy_g_force', 'Superheavy G-Force (30-Frame Distance)',
     'sh_g_force_vs_time.png', 'SH G-Force', 'Real Time (s)', 'G-Force (g)'),
    ('real_time', 'starship_g_force', 'Starship G-Force (10-Frame Distance)',
     'ss_g_force_vs_time.png', 'SS G-Force', 'Real Time (s)', 'G-Force (g)'),
] + ENGINE_PLOT_PARAMS

# Updated global plot parameters for plot_multiple_launches with axis names.
PLOT_MULTIPLE_LAUNCHES_PARAMS = [
    ('real_time', 'superheavy.speed', 'Comparison of Superheavy Speeds',
     'comparison_superheavy.speeds.png', 'Real Time (s)', 'Superheavy Speed (km/h)'),
    ('real_time', 'starship.speed', 'Comparison of Starship Speeds',
     'comparison_starship.speeds.png', 'Real Time (s)', 'Starship Speed (km/h)'),
    ('real_time', 'superheavy.altitude', 'Comparison of Superheavy Altitudes',
     'comparison_superheavy.altitudes.png', 'Real Time (s)', 'Superheavy Altitude (km)'),
    ('real_time', 'starship.altitude', 'Comparison of Starship Altitudes',
     'comparison_starship.altitudes.png', 'Real Time (s)', 'Starship Altitude (km)'),
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
] + ENGINE_PLOT_PARAMS + [
    ('real_time', 'superheavy_all_active', 'Comparison of Superheavy Engine Activity',
     'comparison_superheavy_engines.png', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time', 'starship_all_active', 'Comparison of Starship Engine Activity',
     'comparison_starship_engines.png', 'Real Time (s)', 'Active Engines (count)'),
]


def create_engine_timeline_plot(df: pd.DataFrame, folder: str, title: str = "Engine Activity Timeline", show_figures: bool = True):
    """
    Create a specialized plot showing engine activity over time.
    
    Args:
        df (pd.DataFrame): DataFrame with processed engine data
        folder (str): Folder to save the plot
        title (str): Title for the plot
        show_figures (bool): Whether to display the figures
    """
    # Create figure
    plt.figure(figsize=(16, 9))
    
    # Set up a grid of subplots - 2 rows (Superheavy & Starship)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
    # Plot Superheavy engine data using raw counts
    ax1.plot(df['real_time'], df['superheavy_central_active'], 'ro-', label='Central Stack (max 3)', alpha=0.4)
    ax1.plot(df['real_time'], df['superheavy_inner_active'], 'go-', label='Inner Ring (max 10)', alpha=0.4)
    ax1.plot(df['real_time'], df['superheavy_outer_active'], 'bo-', label='Outer Ring (max 20)', alpha=0.4)
    ax1.plot(df['real_time'], df['superheavy_all_active'], 'ko-', label='All Engines (max 33)', linewidth=2)
    
    ax1.set_title('Superheavy Engine Activity')
    ax1.set_ylabel('Active Engines (count)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0, 35)  # Set y-axis limit to slightly above max engine count (33)
    
    # Plot Starship engine data using raw counts
    ax2.plot(df['real_time'], df['starship_rearth_active'], 'ro-', label='Raptor Earth (max 3)', alpha=0.4)
    ax2.plot(df['real_time'], df['starship_rvac_active'], 'go-', label='Raptor Vacuum (max 3)', alpha=0.4)
    ax2.plot(df['real_time'], df['starship_all_active'], 'ko-', label='All Engines (max 6)', linewidth=2)
    
    ax2.set_title('Starship Engine Activity')
    ax2.set_xlabel('Real Time (s)')
    ax2.set_ylabel('Active Engines (count)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 7)  # Set y-axis limit to slightly above max engine count (6)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    # Save figure
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/engine_timeline.png")
    
    if show_figures:
        plt.show()
    else:
        plt.close()


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


def create_engine_performance_correlation(df: pd.DataFrame, 
                                         x_col: str = 'real_time',
                                         y_col: str = 'superheavy.speed', 
                                         color_col: str = 'superheavy_all_active',
                                         title: str = 'Speed vs. Engine Activity',
                                         x_label: str = 'Time (s)',
                                         y_label: str = 'Speed (km/h)',
                                         color_label: str = 'Active Engines (count)',
                                         filename: str = 'engine_speed_correlation.png',
                                         folder: str = 'results',
                                         cmap: str = 'viridis',
                                         alpha: float = 0.5,
                                         point_size: int = 10,
                                         show_figures: bool = True) -> None:
    """
    Create a plot showing correlation between engine activity and vehicle performance.
    
    Args:
        df (pd.DataFrame): DataFrame with processed engine data
        x_col (str): Column name for x-axis data (default: 'real_time')
        y_col (str): Column name for y-axis data (default: 'superheavy.speed')
        color_col (str): Column name for color data (default: 'superheavy_all_active')
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        color_label (str): Colorbar label
        filename (str): Filename for saving the plot
        folder (str): Folder to save the plot
        cmap (str): Colormap name for the scatter plot
        alpha (float): Alpha transparency for scatter points
        point_size (int): Size of scatter points
        show_figures (bool): Whether to display the figure
    """
    # Create figure
    plt.figure(figsize=(16, 9))
    
    # Set up scatter plot with engine activity as color
    scatter = plt.scatter(df[x_col], df[y_col], 
                          c=df[color_col], 
                          cmap=cmap, 
                          alpha=alpha,
                          s=point_size)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label)
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    
    # Save figure
    os.makedirs(folder, exist_ok=True)
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
        show_figures (bool): Whether to show figures or just save them.
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
    
    # Check which column naming scheme is used and set values accordingly
    speed_col = 'superheavy.speed' if 'superheavy.speed' in df.columns else 'superheavy_speed'
    alt_col = 'superheavy.altitude' if 'superheavy.altitude' in df.columns else 'superheavy_altitude'
    df.loc[df['real_time'] > seven_minutes, [speed_col, alt_col]] = None

    # Calculate acceleration using 30-frame distance
    # Make sure to use the correct column names
    sh_speed_col = 'superheavy.speed' if 'superheavy.speed' in df.columns else 'superheavy_speed'
    ss_speed_col = 'starship.speed' if 'starship.speed' in df.columns else 'starship_speed'
    
    df['superheavy_acceleration'] = compute_acceleration(df, sh_speed_col)
    df['starship_acceleration'] = compute_acceleration(df, ss_speed_col)
    
    # Calculate G-forces
    df['superheavy_g_force'] = df['superheavy_acceleration'] / G_FORCE_CONVERSION
    df['starship_g_force'] = df['starship_acceleration'] / G_FORCE_CONVERSION

    # Determine the folder name based on the launch number
    launch_number = extract_launch_number(json_path)
    folder = os.path.join("results", f"launch_{launch_number}")

    # Create specialized engine timeline plot
    create_engine_timeline_plot(df, folder, f"Launch {launch_number} - Engine Activity Timeline", show_figures)

    # Create correlation plots between engine activity and performance using the reusable function
    # Superheavy correlation
    create_engine_performance_correlation(
        df=df,
        x_col='real_time',
        y_col='superheavy.speed',
        color_col='superheavy_all_active',
        title='Superheavy Speed vs. Engine Activity',
        x_label='Real Time (s)',
        y_label='Speed (km/h)',
        color_label='Active Engines (count)',
        filename='superheavy_engine_speed_correlation.png',
        folder=folder,
        show_figures=show_figures
    )
    
    # Starship correlation
    create_engine_performance_correlation(
        df=df,
        x_col='real_time',
        y_col='starship.speed',
        color_col='starship_all_active',
        title='Starship Speed vs. Engine Activity',
        x_label='Real Time (s)',
        y_label='Speed (km/h)',
        color_label='Active Engines (count)',
        filename='starship_engine_speed_correlation.png',
        folder=folder,
        show_figures=show_figures
    )
    
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
    plt.savefig(os.path.join(folder, filename))
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
            
        # Calculate acceleration using 30-frame distance
        df['superheavy_acceleration'] = compute_acceleration(df, 'superheavy.speed')
        df['starship_acceleration'] = compute_acceleration(df, 'starship.speed')
        
        # Calculate G-forces
        df['superheavy_g_force'] = df['superheavy_acceleration'] / G_FORCE_CONVERSION
        df['starship_g_force'] = df['starship_acceleration'] / G_FORCE_CONVERSION
        
        df_list.append(df)
        labels.append(f'launch {extract_launch_number(json_path)}')

    # Sort labels and create folder name
    labels.sort()
    folder_name = os.path.join("results", "compare_launches", f"launches_{'_'.join(labels)}")
    os.makedirs(folder_name, exist_ok=True)

    for params in PLOT_MULTIPLE_LAUNCHES_PARAMS:
        if len(params) == 4:
            plot_multiple_launches(df_list, *params, folder_name, labels)
        else:
            # Unpack: x, y, title, filename, x_axis, y_axis.
            x, y, title, filename, x_axis, y_axis = params
            plot_multiple_launches(df_list, x, y, title, filename, folder_name, labels, x_axis, y_axis)
