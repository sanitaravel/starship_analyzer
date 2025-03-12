import numpy as np
from typing import Dict

# Engine coordinates
SUPERHEAVY_ENGINES = {
    "central_stack": [(109, 970), (120, 989), (98, 989)],
    "inner_ring": [
        (102, 1018), (82, 1006), (74, 986), (78, 964), (94, 950),
        (116, 948), (136, 958), (144, 978), (140, 1000), (124, 1016)
    ],
    "outer_ring": [
        (106, 1044), (86, 1040), (70, 1030), (57, 1016), (49, 998),
        (47, 980), (51, 960), (61, 944), (75, 930), (93, 922),
        (112, 920), (131, 924), (148, 934), (161, 948), (169, 966),
        (171, 986), (167, 1005), (157, 1022), (143, 1034), (125, 1042)
    ]
}

STARSHIP_ENGINES = {
    "rearth": [(1801, 986), (1830, 986), (1815, 1012)],
    "rvac": [(1764, 1024), (1815, 937), (1866, 1024)]
}


def detect_engine_status(image: np.ndarray, debug: bool = False) -> Dict:
    """
    Detect whether engines are turned on by checking pixel values at specific coordinates.
    An engine is considered on if the pixel is fully white (#FFFFFF).

    Args:
        image (numpy.ndarray): The image to process.
        debug (bool): Whether to enable debug prints.

    Returns:
        dict: A dictionary containing engine statuses for Superheavy and Starship.
    """
    # Define threshold for white (can be adjusted if needed)
    WHITE_THRESHOLD = 240
    
    # Initialize engine status dictionaries
    superheavy_engines = {
        "central_stack": [],
        "inner_ring": [],
        "outer_ring": []
    }
    
    starship_engines = {
        "rearth": [],
        "rvac": []
    }
    
    # Check Superheavy engines
    for section, coordinates in SUPERHEAVY_ENGINES.items():
        for i, (x, y) in enumerate(coordinates):
            # Check if coordinates are within image boundaries
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                # Get pixel value (BGR format)
                pixel = image[y, x]
                # Check if all channels are above threshold (close to white)
                is_on = all(channel >= WHITE_THRESHOLD for channel in pixel)
                superheavy_engines[section].append(is_on)
                if debug:
                    print(f"Superheavy {section} engine {i+1}: {is_on} (pixel value: {pixel})")
            else:
                # If coordinates are out of bounds, consider engine off
                superheavy_engines[section].append(False)
                if debug:
                    print(f"Superheavy {section} engine {i+1}: Coordinates out of bounds")
    
    # Check Starship engines
    for section, coordinates in STARSHIP_ENGINES.items():
        for i, (x, y) in enumerate(coordinates):
            # Check if coordinates are within image boundaries
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                # Get pixel value
                pixel = image[y, x]
                # Check if all channels are above threshold (close to white)
                is_on = all(channel >= WHITE_THRESHOLD for channel in pixel)
                starship_engines[section].append(is_on)
                if debug:
                    print(f"Starship {section} engine {i+1}: {is_on} (pixel value: {pixel})")
            else:
                # If coordinates are out of bounds, consider engine off
                starship_engines[section].append(False)
                if debug:
                    print(f"Starship {section} engine {i+1}: Coordinates out of bounds")
    
    return {
        "superheavy": superheavy_engines,
        "starship": starship_engines
    }
