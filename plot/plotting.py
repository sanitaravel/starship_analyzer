import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from .data_processing import load_and_clean_data, compute_acceleration, compute_g_force
from constants import (ANALYZE_RESULTS_PLOT_PARAMS,
                       PLOT_MULTIPLE_LAUNCHES_PARAMS)
from utils import extract_launch_number

# Set seaborn style globally for all plots
sns.set_theme(style="whitegrid", context="talk",
              palette="colorblind", font_scale=1.1)


def create_engine_timeline_plot(df: pd.DataFrame, folder: str, title: str = "Engine Activity Timeline", show_figures: bool = True):
    """
    Create a specialized plot showing engine activity over time using seaborn.

    Args:
        df (pd.DataFrame): DataFrame with processed engine data
        folder (str): Folder to save the plot
        title (str): Title for the plot
        show_figures (bool): Whether to display the figures
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Plot Superheavy engine data using raw counts with seaborn
    sns.lineplot(x='real_time_seconds', y='superheavy_central_active', data=df,
                 label='Central Stack (max 3)', ax=ax1, marker='o', alpha=0.6, color='red')
    sns.lineplot(x='real_time_seconds', y='superheavy_inner_active', data=df,
                 label='Inner Ring (max 10)', ax=ax1, marker='o', alpha=0.6, color='green')
    sns.lineplot(x='real_time_seconds', y='superheavy_outer_active', data=df,
                 label='Outer Ring (max 20)', ax=ax1, marker='o', alpha=0.6, color='blue')
    sns.lineplot(x='real_time_seconds', y='superheavy_all_active', data=df,
                 label='All Engines (max 33)', ax=ax1, marker='o', linewidth=2.5, color='black')

    ax1.set_title('Superheavy Engine Activity')
    ax1.set_ylabel('Active Engines (count)')
    # Set y-axis limit to slightly above max engine count (33)
    ax1.set_ylim(0, 35)

    # Plot Starship engine data using raw counts with seaborn
    sns.lineplot(x='real_time_seconds', y='starship_rearth_active', data=df,
                 label='Raptor Earth (max 3)', ax=ax2, marker='o', alpha=0.6, color='red')
    sns.lineplot(x='real_time_seconds', y='starship_rvac_active', data=df,
                 label='Raptor Vacuum (max 3)', ax=ax2, marker='o', alpha=0.6, color='green')
    sns.lineplot(x='real_time_seconds', y='starship_all_active', data=df,
                 label='All Engines (max 6)', ax=ax2, marker='o', linewidth=2.5, color='black')

    ax2.set_title('Starship Engine Activity')
    ax2.set_xlabel('Real Time (s)')
    ax2.set_ylabel('Active Engines (count)')
    # Set y-axis limit to slightly above max engine count (6)
    ax2.set_ylim(0, 7)

    # Add overall title
    fig.suptitle(title, fontsize=16)
    # Adjust layout to make room for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/engine_timeline.png", dpi=300, bbox_inches='tight')

    if show_figures:
        plt.show()
    else:
        plt.close()


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, label: str, x_axis: str, y_axis: str, folder: str, show_figures: bool) -> None:
    """
    Create and save a scatter plot for the original and smoothed data using seaborn.

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

    # Create figure with seaborn styling
    plt.figure(figsize=(16, 9))

    # Create scatter plot with seaborn
    scatter_plot = sns.scatterplot(x=x, y=y, data=df, label=f"{label} (Raw Data)",
                                   s=30, alpha=0.3, edgecolor=None)

    # Add trendline for acceleration and g-force plots
    if 'acceleration' in y or 'g_force' in y:
        # Only use non-null values for the trendline
        valid_data = df[[x, y]].dropna()

        if len(valid_data) > 10:  # Only add trendline if we have enough data points
            # Use LOWESS to create a smooth trendline
            z = lowess(valid_data[y], valid_data[x], frac=0.01)

            # Add trendline with seaborn styling
            plt.plot(z[:, 0], z[:, 1], color='crimson',
                     linewidth=2.5, label=f"{label} (Trend)")

    # Add NASA's G limit lines for G-force plots
    if 'g_force' in y:
        plt.axhline(y=4.5, color='red', linestyle='--', linewidth=2,
                    label="NASA's 4.5G Maximum Sustained Acceleration Limit")
        plt.axhline(y=-4.5, color='red', linestyle='--', linewidth=2)

    # Set labels with seaborn styling
    plt.xlabel(x_axis if x_axis else x.capitalize(), fontsize=12)
    plt.ylabel(y_axis if y_axis else y.capitalize(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add legend with improved visibility
    plt.legend(frameon=True, fontsize=10)

    # Save with high quality
    plt.savefig(f"{folder}/{filename}", dpi=300, bbox_inches='tight')

    if show_figures:
        plt.show()
    else:
        plt.close()


def create_engine_performance_correlation(df: pd.DataFrame,
                                          x_col: str = 'real_time_seconds',
                                          y_col: str = 'superheavy.speed',
                                          color_col: str = 'superheavy_all_active',
                                          title: str = 'Speed vs. Engine Activity',
                                          x_label: str = 'Time (s)',
                                          y_label: str = 'Speed (km/h)',
                                          color_label: str = 'Active Engines (count)',
                                          filename: str = 'engine_speed_correlation.png',
                                          folder: str = 'results',
                                          cmap: str = 'viridis',
                                          alpha: float = 0.7,
                                          point_size: int = 50,
                                          show_figures: bool = True) -> None:
    """
    Create a plot showing correlation between engine activity and vehicle performance using seaborn.
    """
    # Create figure with seaborn styling
    plt.figure(figsize=(16, 9))

    # Create advanced scatter plot with seaborn
    scatter = sns.scatterplot(
        x=x_col,
        y=y_col,
        hue=color_col,
        size=color_col,
        sizes=(20, 200),  # Range of point sizes
        palette=cmap,     # Use colormap
        alpha=alpha,      # Transparency
        data=df           # Data source
    )

    # Add a legend with custom title
    legend = scatter.legend(title=color_label, fontsize=10)
    plt.setp(legend.get_title(), fontsize=12)

    # Add labels and title with seaborn styling
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add smoothed trend line
    try:
        # Try to add a trend line if we have valid data
        valid_data = df[[x_col, y_col]].dropna()
        if len(valid_data) > 10:
            sns.regplot(x=x_col, y=y_col, data=df, scatter=False,
                        line_kws={"color": "red", "alpha": 0.7, "lw": 2, "linestyle": "--"})
    except:
        # Skip trend line if there's an error
        pass

    # Save figure with high quality
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{filename}", dpi=300, bbox_inches='tight')

    if show_figures:
        plt.show()
    else:
        plt.close()


def plot_flight_data(json_path: str, start_time: int = 0, end_time: int = -1, show_figures: bool = True) -> None:
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
    df = df[df['real_time_seconds'] >= start_time]
    if end_time != -1:
        df = df[df['real_time_seconds'] <= end_time]

    # Set all Superheavy's data to None after 7 minutes and 30 seconds
    seven_minutes = 7 * 60 + 30  # 7 minutes and 30 seconds in seconds

    # Check which column naming scheme is used and set values accordingly
    speed_col = 'superheavy.speed' if 'superheavy.speed' in df.columns else 'superheavy_speed'
    alt_col = 'superheavy.altitude' if 'superheavy.altitude' in df.columns else 'superheavy_altitude'
    df.loc[df['real_time_seconds'] > seven_minutes, [speed_col, alt_col]] = None

    # Calculate acceleration using 30-frame distance
    # Make sure to use the correct column names
    sh_speed_col = 'superheavy.speed' if 'superheavy.speed' in df.columns else 'superheavy_speed'
    ss_speed_col = 'starship.speed' if 'starship.speed' in df.columns else 'starship_speed'

    df['superheavy_acceleration'] = compute_acceleration(df, sh_speed_col)
    df['starship_acceleration'] = compute_acceleration(df, ss_speed_col)

    # Calculate G-forces
    df['superheavy_g_force'] = compute_g_force(df['superheavy_acceleration'])
    df['starship_g_force'] = compute_g_force(df['starship_acceleration'])

    # Determine the folder name based on the launch number
    launch_number = extract_launch_number(json_path)
    folder = os.path.join("results", f"launch_{launch_number}")

    # Create specialized engine timeline plot
    create_engine_timeline_plot(
        df, folder, f"Launch {launch_number} - Engine Activity Timeline", show_figures)

    # Create correlation plots between engine activity and performance using the reusable function
    # Superheavy correlation
    create_engine_performance_correlation(
        df=df,
        x_col='real_time_seconds',
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
        x_col='real_time_seconds',
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
                           labels: list[str], x_axis: str = None, y_axis: str = None, show_figures: bool = True) -> None:
    """
    Plot a comparison of multiple dataframes using seaborn.

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
        show_figures (bool): Whether to show figures or just save them.
    """
    # Create figure with seaborn styling
    plt.figure(figsize=(16, 9))

    # Custom color palette with distinct colors for each launch
    palette = sns.color_palette("husl", len(df_list))

    # Plot each dataset with both scatter points and trendline
    for i, (df, label) in enumerate(zip(df_list, labels)):
        color = palette[i]

        # Add scatter plot with seaborn
        scatter = sns.scatterplot(
            x=x,
            y=y,
            data=df,
            label=f"{label} (Raw)",
            color=color,
            alpha=0.3,
            s=30
        )

        # Add trendline for acceleration and g-force plots
        if 'acceleration' in y or 'g_force' in y:
            # Only use non-null values for the trendline
            valid_data = df[[x, y]].dropna()

            if len(valid_data) > 10:  # Only add trendline if we have enough data points
                # Use LOWESS to create a smooth trendline
                z = lowess(valid_data[y], valid_data[x], frac=0.01)
                plt.plot(z[:, 0], z[:, 1], '-', linewidth=2.5,
                         label=f"{label} (Trend)", color=color)
        else:
            # For non-acceleration plots, add a smoothed line
            try:
                sns.lineplot(x=x, y=y, data=df, color=color,
                             label=f"{label} (Trend)")
            except:
                # Skip if there's an error
                pass

    # Add NASA's G limit line for G-force plots
    if 'g_force' in y:
        plt.axhline(y=3, color='red', linestyle='--', linewidth=2,
                    label="NASA's 3G Maximum Sustained Acceleration Limit")

    # Set labels with seaborn styling
    plt.xlabel(x_axis if x_axis else x.capitalize(), fontsize=12)
    plt.ylabel(y_axis if y_axis else y.capitalize(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add legend with improved visibility
    plt.legend(frameon=True, fontsize=10)

    # Save figure with high quality
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches='tight')

    if show_figures:
        plt.show()
    else:
        plt.close()


def compare_multiple_launches(start_time: int, end_time: int, *json_paths: str, show_figures: bool = True) -> None:
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
        df = df[df['real_time_seconds'] >= start_time]
        if end_time != -1:
            df = df[df['real_time_seconds'] <= end_time]

        # Calculate acceleration using 30-frame distance
        df['superheavy_acceleration'] = compute_acceleration(
            df, 'superheavy.speed')
        df['starship_acceleration'] = compute_acceleration(
            df, 'starship.speed')

        # Calculate G-forces
        df['superheavy_g_force'] = compute_g_force(
            df['superheavy_acceleration'])
        df['starship_g_force'] = compute_g_force(df['starship_acceleration'])

        df_list.append(df)
        labels.append(f'launch {extract_launch_number(json_path)}')

    # Sort labels and create folder name
    labels.sort()
    folder_name = os.path.join(
        "results", "compare_launches", f"launches_{'_'.join(labels)}")
    os.makedirs(folder_name, exist_ok=True)

    for params in PLOT_MULTIPLE_LAUNCHES_PARAMS:
        if len(params) == 4:
            plot_multiple_launches(
                df_list, *params, folder_name, labels, show_figures=show_figures)
        else:
            # Unpack: x, y, title, filename, x_axis, y_axis.
            x, y, title, filename, x_axis, y_axis = params
            plot_multiple_launches(df_list, x, y, title, filename, folder_name,
                                   labels, x_axis, y_axis, show_figures=show_figures)
