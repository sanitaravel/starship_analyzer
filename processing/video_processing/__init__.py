"""
Video processing module for analyzing rocket launch videos.
"""
from .validation import validate_video, get_video_properties
from .frame_processing import process_frame, process_single_frame
from .batch_processing import process_batch, process_video_frames, create_batches, summarize_batch
from .results import calculate_real_times, save_results
from .main_processing import iterate_through_frames, process_frames

__all__ = [
    'validate_video',
    'get_video_properties',
    'process_frame',
    'process_single_frame',
    'process_batch',
    'process_video_frames',
    'create_batches',
    'summarize_batch',
    'calculate_real_times',
    'save_results',
    'iterate_through_frames',
    'process_frames'
]
