import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from utils import display_image
import time
from tqdm import tqdm

# Mode selection
process_whole_video = True  # Set to True to process the entire video
use_static_image = False    # Only used if process_whole_video is False
create_animation = True     # Set to True to create an animation of strip brightness

# Timeframe selection (in seconds)
start_time = 0      # Start time in seconds (0 for beginning of video)
end_time = None    # End time in seconds (None for end of video)

# Image/video path
static_image_path = ".\\.tmp\\random_frame.jpg"
video_path = ".\\flight_recordings\\flight_2.mp4"

# Strip coordinates and reference pixel setup
strip_coords = [
    (275, 1007),
    (275, 1042),
    (1455, 1007),
    (1455, 1037)
]
strip_length = 240
strip_height = 1  # 1-pixel tall strips

ref_pixel_coords = [
    (255, 1006),
    (227, 1042),
    (1435, 1006),
    (1407, 1037)
]

threshold = 0.9  # Brightness threshold for determining "bright" pixels
ref_threshold = 0.3  # Separate threshold for reference pixels

# Function to process a single frame
def process_frame(frame, frame_number=0, fps=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate current time in seconds
    current_time = frame_number / fps if fps > 0 else 0
    
    # Process each strip - collect all data
    initial_results = []
    initial_lengths = []
    ref_brightness_values = []
    all_ref_is_bright = []
    
    # First, analyze all strips
    for i, ((x, y), (ref_x, ref_y)) in enumerate(zip(strip_coords, ref_pixel_coords), 1):
        # Calculate second reference pixel position
        if i == 1 or i == 3:  # For strips 1 and 3, shift 5px to the right
            ref_x2 = ref_x + 5
            ref_y2 = ref_y
        else:  # For strips 2 and 4, shift 5px to the left
            ref_x2 = ref_x - 5
            ref_y2 = ref_y
        
        # Check both reference pixels are within image bounds
        ref_pixel1_normalized = 0
        ref_pixel2_normalized = 0
        
        if (0 <= ref_y < gray.shape[0] and 0 <= ref_x < gray.shape[1] and 
            0 <= ref_y2 < gray.shape[0] and 0 <= ref_x2 < gray.shape[1]):
            # Get reference pixel values and normalize them
            ref_pixel1_value = gray[ref_y, ref_x]
            ref_pixel2_value = gray[ref_y2, ref_x2]
            
            # Normalize
            min_val = gray.min()
            ptp_val = np.ptp(gray) or 1
            ref_pixel1_normalized = (ref_pixel1_value - min_val) / ptp_val
            ref_pixel2_normalized = (ref_pixel2_value - min_val) / ptp_val
            
            # Check if the difference between the pixels is noticeable
            pixel_diff = abs(ref_pixel2_normalized - ref_pixel1_normalized)
            ref_is_bright = pixel_diff > 0.2  # Adjust this threshold as needed
        else:
            ref_is_bright = False
        
        all_ref_is_bright.append(ref_is_bright)
        ref_brightness_values.append((ref_pixel1_normalized, ref_pixel2_normalized))
        
        # Extract and process strip
        y_start = max(0, y - strip_height//2)
        y_end = min(gray.shape[0], y + strip_height//2 + 1)
        x_end = min(gray.shape[1], x + strip_length)
        
        strip = gray[y_start:y_end, x:x_end]
        brightness_profile = strip.mean(axis=0)
        norm_brightness = (brightness_profile - brightness_profile.min()) / (np.ptp(brightness_profile) or 1)
        bright_pixels = norm_brightness > threshold
        
        if ref_is_bright:
            bright_indices = np.where(bright_pixels)[0]
            rightmost_pos = bright_indices.max() if len(bright_indices) > 0 else 0
            effective_length = rightmost_pos + 1  # +1 because indices are 0-based
            fullness_percentage = (rightmost_pos / strip_length) * 100 if strip_length > 0 else 0
        else:
            rightmost_pos = 0
            effective_length = 0
            fullness_percentage = 0
        
        initial_lengths.append(effective_length)
        initial_results.append(fullness_percentage)
    
    # Apply grouping rules
    results = initial_results.copy()
    
    # Group 1: bars 1-2
    if abs(initial_results[0] - initial_results[1]) > 30:
        # Use max value in the first 200 seconds, min value after that
        if current_time < 200:
            chosen_value = max(initial_results[0], initial_results[1])
        else:
            chosen_value = min(initial_results[0], initial_results[1])
        results[0] = chosen_value
        results[1] = chosen_value
    
    # Group 2: bars 3-4
    if abs(initial_results[2] - initial_results[3]) > 30:
        # Use max value in the first 200 seconds, min value after that
        if current_time < 200:
            chosen_value = max(initial_results[2], initial_results[3])
        else:
            chosen_value = min(initial_results[2], initial_results[3])
        results[2] = chosen_value
        results[3] = chosen_value
    
    return results, ref_brightness_values

# Main execution
if process_whole_video:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Convert time values to frame numbers
    start_frame = int(start_time * fps) if start_time is not None else 0
    end_frame = int(end_time * fps) if end_time is not None else total_frames
    
    # Make sure frame numbers are within valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    frames_to_process = end_frame - start_frame
    
    print(f"Processing frames from {start_time}s to {end_time if end_time is not None else 'end'} " +
          f"({start_frame}-{end_frame}, {frames_to_process} frames) at {fps} fps")
    
    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    start_time_processing = time.time()
    
    # Arrays to store results
    frame_results = []  # Will be a list of lists, each inner list containing 4 bar values
    frame_numbers = []  # Store processed frame numbers
    
    # Special handling for bars 3-4
    hundred_second_frame = int(100 * fps)
    hundred_second_index = -1  # Will store the index where we hit 100s
    bars_34_values = None  # Will store the values for bars 3-4 at 100s
    
    # For animation: store brightness profiles and other data
    if create_animation:
        max_frames_for_animation = 300  # Limit frames to keep memory usage reasonable
        animation_interval = max(1, total_frames // max_frames_for_animation)
        all_brightness_profiles = []  # Store brightness profiles for animation
        all_rightmost_positions = []  # Store rightmost bright positions for animation
        all_ref_statuses = []  # Store reference pixel statuses for animation
        animation_frame_numbers = []  # Store actual frame numbers used in animation
        
    # Process frames with tqdm progress bar
    frame_count = start_frame
    
    # Use tqdm for progress bar - process frames in selected range
    with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame - now passing frame number and fps
            bar_values, ref_values = process_frame(frame, frame_count, fps)
            frame_results.append(bar_values)
            frame_numbers.append(frame_count)
            
            # Check if we reached the 100 second mark
            if frame_count >= hundred_second_frame and hundred_second_index == -1:
                hundred_second_index = len(frame_results) - 1
                bars_34_values = [bar_values[2], bar_values[3]]
                print(f"Captured bars 3-4 values at 100s: {bars_34_values}")
            
            # For animation: collect additional data
            if create_animation:
                # Adjust interval based on timeframe length
                animation_interval = max(1, frames_to_process // 300)
                
                if (frame_count - start_frame) % animation_interval == 0:
                    # Need to re-process frame to get brightness profiles
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_brightness_profiles = []
                    frame_rightmost_positions = []
                    frame_ref_statuses = []
                    animation_frame_numbers.append(frame_count)
                    
                    for i, ((x, y), (ref_x, ref_y)) in enumerate(zip(strip_coords, ref_pixel_coords)):
                        # Calculate second reference pixel position
                        if i == 0 or i == 2:  # For strips 1 and 3, shift 5px to the right
                            ref_x2 = ref_x + 5
                            ref_y2 = ref_y
                        else:  # For strips 2 and 4, shift 5px to the left
                            ref_x2 = ref_x - 5
                            ref_y2 = ref_y
                        
                        # Check both reference pixels
                        ref_pixel1_normalized = 0
                        ref_pixel2_normalized = 0
                        
                        if (0 <= ref_y < gray.shape[0] and 0 <= ref_x < gray.shape[1] and 
                            0 <= ref_y2 < gray.shape[0] and 0 <= ref_x2 < gray.shape[1]):
                            # Get reference pixel values and normalize them
                            ref_pixel1_value = gray[ref_y, ref_x]
                            ref_pixel2_value = gray[ref_y2, ref_x2]
                            
                            min_val = gray.min()
                            ptp_val = np.ptp(gray) or 1
                            ref_pixel1_normalized = (ref_pixel1_value - min_val) / ptp_val
                            ref_pixel2_normalized = (ref_pixel2_value - min_val) / ptp_val
                            
                            pixel_diff = abs(ref_pixel2_normalized - ref_pixel1_normalized)
                            ref_is_bright = pixel_diff > 0.2
                        else:
                            ref_is_bright = False
                            pixel_diff = 0
                        
                        frame_ref_statuses.append((ref_is_bright, pixel_diff))
                        
                        # Extract strip
                        y_start = max(0, y - strip_height//2)
                        y_end = min(gray.shape[0], y + strip_height//2 + 1)
                        x_end = min(gray.shape[1], x + strip_length)
                        strip = gray[y_start:y_end, x:x_end]
                        
                        # Get brightness profile
                        brightness_profile = strip.mean(axis=0)
                        norm_brightness = (brightness_profile - brightness_profile.min()) / (np.ptp(brightness_profile) or 1)
                        bright_pixels = norm_brightness > threshold
                        
                        # Get rightmost bright position
                        bright_indices = np.where(bright_pixels)[0]
                        rightmost_pos = bright_indices.max() if len(bright_indices) > 0 else 0
                        
                        frame_brightness_profiles.append(norm_brightness)
                        frame_rightmost_positions.append(rightmost_pos)
                    
                    all_brightness_profiles.append(frame_brightness_profiles)
                    all_rightmost_positions.append(frame_rightmost_positions)
                    all_ref_statuses.append(frame_ref_statuses)
            
            frame_count += 1
            pbar.update(1)
    
    # Apply special handling for bars 3-4: overwrite all values before 100s
    if hundred_second_index != -1 and bars_34_values is not None:
        for i in range(hundred_second_index):
            # Overwrite bars 3-4 with the values from 100s
            frame_results[i][2] = bars_34_values[0]
            frame_results[i][3] = bars_34_values[1]
    
    cap.release()
    
    elapsed_time = time.time() - start_time_processing
    print(f"Processed {len(frame_numbers)} frames in {elapsed_time:.1f} seconds")
    
    # Create animation if requested
    if create_animation and all_brightness_profiles:
        print("Creating animation...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Initialize plots
        lines = []
        threshold_lines = []
        rightmost_lines = []
        
        for i, ax in enumerate(axes):
            line, = ax.plot([], [], label='Brightness')
            threshold_line = ax.axhline(threshold, color='red', linestyle='--', label='Threshold')
            rightmost_line = ax.axvline(0, color='green', linestyle='--', label='Rightmost Bright')
            
            ax.set_ylim(0, 1.1)
            ax.set_xlim(0, strip_length)
            ax.grid(True)
            ax.legend()
            ax.set_title(f'Strip {i+1}')
            
            lines.append(line)
            threshold_lines.append(threshold_line)
            rightmost_lines.append(rightmost_line)
        
        # Animation update function
        def update(frame):
            frame_idx = frame
            brightness_profiles = all_brightness_profiles[frame_idx]
            rightmost_positions = all_rightmost_positions[frame_idx]
            ref_statuses = all_ref_statuses[frame_idx]
            actual_frame = animation_frame_numbers[frame_idx]
            
            # Calculate time in seconds
            time_seconds = actual_frame / fps
            
            for i, ax in enumerate(axes):
                lines[i].set_data(range(len(brightness_profiles[i])), brightness_profiles[i])
                rightmost_lines[i].set_xdata([rightmost_positions[i]])
                
                # Get reference status information
                ref_active, ref_diff = ref_statuses[i]
                ref_status = "Active" if ref_active else "Inactive"
                
                # Update title with time and reference status
                fullness = (rightmost_positions[i] / strip_length) * 100 if strip_length > 0 else 0
                ax.set_title(f'Strip {i+1}: T={time_seconds:.1f}s, Pos: {rightmost_positions[i]}, Full: {fullness:.1f}%\n' + 
                             f'Ref: {ref_status} (Diff: {ref_diff:.3f})')
            
            return lines + rightmost_lines
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(all_brightness_profiles),
            interval=100, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        # Optional: save animation
        # ani.save('strip_brightness_animation.mp4', writer='ffmpeg', fps=10)
    
    # Plot time series results (standard plot)
    # Convert results to numpy array for easier processing
    results_array = np.array(frame_results)
    
    # Create time axis (in seconds)
    time_axis = np.array(frame_numbers) / fps
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Group 1 (bars 1-2)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, results_array[:, 0], 'b-', label='Bar 1')
    plt.plot(time_axis, results_array[:, 1], 'g-', label='Bar 2')
    plt.title('Group 1: Bars 1-2 Fullness Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Fullness %')
    plt.grid(True)
    plt.legend()
    
    # Group 2 (bars 3-4)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, results_array[:, 2], 'r-', label='Bar 3')
    plt.plot(time_axis, results_array[:, 3], 'c-', label='Bar 4')
    plt.title('Group 2: Bars 3-4 Fullness Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Fullness %')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

else:
    # Original static image or random frame processing
    if use_static_image:
        print(f"Using static image: {static_image_path}")
        img = cv2.imread(static_image_path)
        if img is None:
            print(f"Error: Could not read image file {static_image_path}")
            exit()
    else:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_idx = random.randint(0, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        success, img = cap.read()
        cap.release()

        if not success:
            print(f"Error: Could not read frame {random_frame_idx}")
            exit()

        print(f"Analyzing random frame #{random_frame_idx} out of {total_frames}")
    
    display_image(img, "Original Image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    initial_results = []
    initial_lengths = []
    ref_brightness_values = []
    all_norm_brightness = []
    all_bright_pixels = []
    all_ref_is_bright = []

    for i, ((x, y), (ref_x, ref_y)) in enumerate(zip(strip_coords, ref_pixel_coords), 1):
        # Calculate second reference pixel position
        if i == 1 or i == 3:  # For strips 1 and 3, shift 5px to the right
            ref_x2 = ref_x + 5
            ref_y2 = ref_y
        else:  # For strips 2 and 4, shift 5px to the left
            ref_x2 = ref_x - 5
            ref_y2 = ref_y
        
        # Check both reference pixels
        ref_pixel1_normalized = 0
        ref_pixel2_normalized = 0
        
        if (0 <= ref_y < gray.shape[0] and 0 <= ref_x < gray.shape[1] and 
            0 <= ref_y2 < gray.shape[0] and 0 <= ref_x2 < gray.shape[1]):
            # Get reference pixel values and normalize them
            ref_pixel1_value = gray[ref_y, ref_x]
            ref_pixel2_value = gray[ref_y2, ref_x2]
            
            # Normalize
            min_val = gray.min()
            ptp_val = np.ptp(gray) or 1
            ref_pixel1_normalized = (ref_pixel1_value - min_val) / ptp_val
            ref_pixel2_normalized = (ref_pixel2_value - min_val) / ptp_val
            
            # Check if the difference between the pixels is noticeable
            pixel_diff = abs(ref_pixel2_normalized - ref_pixel1_normalized)
            ref_is_bright = pixel_diff > 0.2  # Adjust this threshold as needed
        else:
            ref_is_bright = False
        
        all_ref_is_bright.append(ref_is_bright)
        ref_brightness_values.append((ref_pixel1_normalized, ref_pixel2_normalized))
        
        y_start = max(0, y - strip_height//2)
        y_end = min(gray.shape[0], y + strip_height//2 + 1)
        x_end = min(gray.shape[1], x + strip_length)
        
        strip = gray[y_start:y_end, x:x_end]
        
        brightness_profile = strip.mean(axis=0)
        
        norm_brightness = (brightness_profile - brightness_profile.min()) / (np.ptp(brightness_profile) or 1)
        all_norm_brightness.append(norm_brightness)
        
        bright_pixels = norm_brightness > threshold
        all_bright_pixels.append(bright_pixels)
        
        if ref_is_bright:
            bright_indices = np.where(bright_pixels)[0]
            rightmost_pos = bright_indices.max() if len(bright_indices) > 0 else 0
            effective_length = rightmost_pos + 1
            fullness_percentage = (rightmost_pos / strip_length) * 100 if strip_length > 0 else 0
        else:
            rightmost_pos = 0
            effective_length = 0
            fullness_percentage = 0
        
        initial_lengths.append(effective_length)
        initial_results.append(fullness_percentage)

    results = initial_results.copy()
    lengths = initial_lengths.copy()

    if abs(initial_results[0] - initial_results[1]) > 10:
        min_value = min(initial_results[0], initial_results[1])
        min_length = min(initial_lengths[0], initial_lengths[1]) if min_value > 0 else 0
        results[0] = min_value
        results[1] = min_value
        lengths[0] = min_length
        lengths[1] = min_length
        print(f"Group 1 (bars 1-2) difference > 10%, using smaller value: {min_value:.1f}%")

    if abs(initial_results[2] - initial_results[3]) > 10:
        min_value = min(initial_results[2], initial_results[3])
        min_length = min(initial_lengths[2], initial_lengths[3]) if min_value > 0 else 0
        results[2] = min_value
        results[3] = min_value
        lengths[2] = min_length
        lengths[3] = min_length
        print(f"Group 2 (bars 3-4) difference > 10%, using smaller value: {min_value:.1f}%")

    plt.figure(figsize=(15, 10))

    for i, ((x, y), (ref_x, ref_y)) in enumerate(zip(strip_coords, ref_pixel_coords), 1):
        norm_brightness = all_norm_brightness[i-1]
        ref_is_bright = all_ref_is_bright[i-1]
        
        fullness_percentage = results[i-1]
        effective_length = lengths[i-1]
        
        rightmost_pos = effective_length - 1 if effective_length > 0 else 0
        
        plt.subplot(2, 2, i)
        plt.plot(norm_brightness, label='Brightness')
        plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
        plt.axvline(rightmost_pos, color='green', linestyle='--', label='Rightmost Bright')
        
        # Calculate second reference pixel position
        if i == 1 or i == 3:
            ref_x2 = ref_x + 5
        else:
            ref_x2 = ref_x - 5
        ref_y2 = ref_y
        
        ref_status = "Active" if all_ref_is_bright[i-1] else "Inactive"
        ref1, ref2 = ref_brightness_values[i-1]
        pixel_diff = abs(ref2 - ref1)
        
        # Include both reference pixels info in the title
        group_num = 1 if i <= 2 else 2
        plt.title(f'Strip {i} (Group {group_num}): Length: {effective_length}, Fullness: {fullness_percentage:.1f}%\n' + 
                  f'Ref Pixels ({ref_x},{ref_y})/({ref_x2},{ref_y2}): {ref_status}, Diff: {pixel_diff:.3f}')
        plt.legend()

    plt.tight_layout()
    plt.show()

    for i, (percentage, length, ref_pair) in enumerate(zip(results, lengths, ref_brightness_values[:4]), 1):
        ref1, ref2 = ref_pair
        pixel_diff = abs(ref2 - ref1)
        ref_status = "Active" if pixel_diff > 0.2 else "Inactive"
        print(f'Strip {i}: Length: {length}, Fullness: {percentage:.1f}%, ' + 
              f'Ref Pixel Diff: {pixel_diff:.3f} ({ref_status})')
