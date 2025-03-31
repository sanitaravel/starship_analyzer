import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from .data_processing import load_and_clean_data, compute_acceleration, compute_g_force
from constants import PLOT_MULTIPLE_LAUNCHES_PARAMS
from utils import extract_launch_number
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set seaborn style globally for all plots
sns.set_theme(style="whitegrid", context="talk",
              palette="colorblind", font_scale=1.1)


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
    logger.info(f"Creating multi-launch comparison plot: {title}")
    logger.debug(f"Comparing {len(df_list)} launches: {', '.join(labels)}")
    
    # Create figure with seaborn styling
    plt.figure(figsize=(16, 9))

    # Custom color palette with distinct colors for each launch
    palette = sns.color_palette("husl", len(df_list))

    # Plot each dataset with both scatter points and trendline
    for i, (df, label) in enumerate(zip(df_list, labels)):
        color = palette[i]
        
        # Log data points per launch
        data_count = df[y].notna().sum()
        logger.debug(f"Launch {label}: {data_count} data points for {y}")

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
                logger.debug(f"Launch {label}: Adding LOWESS trendline with {len(valid_data)} points")
                # Use LOWESS to create a smooth trendline
                z = lowess(valid_data[y], valid_data[x], frac=0.01)
                plt.plot(z[:, 0], z[:, 1], '-', linewidth=2.5,
                         label=f"{label} (Trend)", color=color)
        else:
            # For non-acceleration plots, add a smoothed line
            try:
                valid_data = df[[x, y]].dropna()
                if len(valid_data) > 10:
                    logger.debug(f"Launch {label}: Adding trend line with {len(valid_data)} points")
                    sns.lineplot(x=x, y=y, data=df, color=color,
                                label=f"{label} (Trend)")
            except Exception as e:
                logger.warning(f"Could not add trend line for {label}: {str(e)}")

    # Add NASA's G limit line for G-force plots
    if 'g_force' in y:
        logger.debug("Adding NASA G-force limit line at 3G")
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
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {save_path}")

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
        show_figures (bool): Whether to show figures or just save them.
    """
    logger.info(f"Comparing multiple launches (time window: {start_time}s to {end_time if end_time != -1 else 'end'}s)")
    logger.debug(f"Loading data from {len(json_paths)} JSON files")
    
    df_list = []
    labels = []

    for json_path in json_paths:
        logger.info(f"Processing {json_path}")
        df = load_and_clean_data(json_path)
        if df.empty:
            logger.warning(f"Empty DataFrame for {json_path}, skipping")
            continue  # Skip if the DataFrame is empty due to JSON error

        # Filter by time window
        original_count = len(df)
        df = df[df['real_time_seconds'] >= start_time]
        if end_time != -1:
            df = df[df['real_time_seconds'] <= end_time]
        logger.debug(f"Using {len(df)} of {original_count} data points after time filtering")

        # Calculate acceleration using 30-frame distance
        sh_speed_col = 'superheavy.speed' if 'superheavy.speed' in df.columns else 'superheavy_speed'
        ss_speed_col = 'starship.speed' if 'starship.speed' in df.columns else 'starship_speed'
        
        df['superheavy_acceleration'] = compute_acceleration(df, sh_speed_col)
        df['starship_acceleration'] = compute_acceleration(df, ss_speed_col)

        # Calculate G-forces
        df['superheavy_g_force'] = compute_g_force(df['superheavy_acceleration'])
        df['starship_g_force'] = compute_g_force(df['starship_acceleration'])

        df_list.append(df)
        launch_id = extract_launch_number(json_path)
        labels.append(f'launch {launch_id}')
        logger.info(f"Successfully processed launch {launch_id}")

    if not df_list:
        logger.error("No valid data available for comparison. Exiting.")
        return
    
    # Sort labels and create folder name
    labels.sort()
    folder_name = os.path.join(
        "results", "compare_launches", f"launches_{'_'.join(labels)}")
    logger.info(f"Creating comparison plots in folder {folder_name}")

    # Create all comparison plots defined in constants
    logger.info(f"Creating {len(PLOT_MULTIPLE_LAUNCHES_PARAMS)} comparison plots")
    for params in PLOT_MULTIPLE_LAUNCHES_PARAMS:
        if len(params) == 4:
            plot_multiple_launches(
                df_list, *params, folder_name, labels, show_figures=show_figures)
        else:
            # Unpack: x, y, title, filename, x_axis, y_axis.
            x, y, title, filename, x_axis, y_axis = params
            plot_multiple_launches(df_list, x, y, title, filename, folder_name,
                                   labels, x_axis, y_axis, show_figures=show_figures)
    
    logger.info("Completed all comparison plots")
