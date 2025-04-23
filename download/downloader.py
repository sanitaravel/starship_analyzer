"""
Core download functionality for different video platforms.
"""
import subprocess
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def download_twitter_broadcast(url, flight_number, output_path="flight_recordings"):
    """
    Downloads a Twitter/X broadcast video using yt-dlp.

    Args:
        url (str): The URL of the Twitter/X broadcast.
        flight_number (int): The flight number to use in the filename.
        output_path (str): The directory to save the downloaded video.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Define output template with flight number
        output_template = f"{output_path}/flight_{flight_number}.%(ext)s"
        
        logger.info(f"Downloading Twitter broadcast from {url}")
        logger.info(f"Output file will be saved as: {output_template}")
        
        # Run yt-dlp to download the video only
        subprocess.run([
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]/bestvideo/best",
            "--no-audio",  # Explicitly disable audio download
            "-o", output_template,
            url
        ], check=True)
        
        logger.info("Download completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download error: {e}")
        print(f"An error occurred during download: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")
        return False

def download_youtube_video(url, flight_number, output_path="flight_recordings"):
    """
    Downloads a YouTube video using yt-dlp.

    Args:
        url (str): The URL of the YouTube video.
        flight_number (int): The flight number to use in the filename.
        output_path (str): The directory to save the downloaded video.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Define output template with flight number
        output_template = f"{output_path}/flight_{flight_number}.%(ext)s"
        
        logger.info(f"Downloading YouTube video from {url}")
        logger.info(f"Output file will be saved as: {output_template}")
        
        # Run yt-dlp to download video only, without audio
        subprocess.run([
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]/bestvideo/best",
            "--no-audio",  # Explicitly disable audio download
            "-o", output_template,
            url
        ], check=True)
        
        logger.info("Download completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"YouTube download error: {e}")
        print(f"An error occurred during YouTube download: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")
        return False
