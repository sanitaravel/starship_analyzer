"""
Constants used throughout the starship_analyzer application.
This file centralizes all constant values to make them easier to maintain.
"""

# ------------------------------
# Engine Detection Constants
# ------------------------------

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

# ------------------------------
# OCR Constants
# ------------------------------

# OCR thresholds
WHITE_THRESHOLD = 230

# ------------------------------
# Data Processing Constants
# ------------------------------

# Physics constants
G_FORCE_CONVERSION = 9.81  # 1G = 9.81 m/s²

# ------------------------------
# Visualization Constants
# ------------------------------

# Figure size and style
FIGURE_SIZE = (16, 9)  # 16:9 aspect ratio for fullscreen

# Font sizes
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
TICK_FONT_SIZE = 10

# Marker styling
MARKER_SIZE = 25
MARKER_ALPHA = 0.5

# Line styling
LINE_WIDTH = 2.5
LINE_ALPHA = 0.8

# Engine plot parameters (using consistent naming and styling)
ENGINE_TIMELINE_PARAMS = {
    "superheavy": {
        "title": "Superheavy Booster Engine Activity",
        "ylabel": "Active Engines",
        "ylim": (0, 35),
        "groups": [
            {"column": "superheavy_central_active", "label": "Central Stack (3)", "color": "red"},
            {"column": "superheavy_inner_active", "label": "Inner Ring (10)", "color": "green"},
            {"column": "superheavy_outer_active", "label": "Outer Ring (20)", "color": "blue"},
            {"column": "superheavy_all_active", "label": "All Engines (33)", "color": "black"}
        ]
    },
    "starship": {
        "title": "Starship Engine Activity",
        "ylabel": "Active Engines",
        "ylim": (0, 7),
        "groups": [
            {"column": "starship_rearth_active", "label": "Raptor Earth (3)", "color": "red"},
            {"column": "starship_rvac_active", "label": "Raptor Vacuum (3)", "color": "green"},
            {"column": "starship_all_active", "label": "All Engines (6)", "color": "black"}
        ]
    },
    "xlabel": "Mission Time (seconds)",
    "overall_title": "Engine Activity Timeline"
}

# Analysis plot parameters (using consistent naming and styling)
ANALYZE_RESULTS_PLOT_PARAMS = [
    # Speed vs Time
    ('real_time_seconds', 'superheavy.speed', 'Superheavy Booster Velocity',
     'superheavy_velocity.png', 'Booster', 'Mission Time (seconds)', 'Velocity (km/h)'),
    ('real_time_seconds', 'starship.speed', 'Starship Velocity',
     'starship_velocity.png', 'Starship', 'Mission Time (seconds)', 'Velocity (km/h)'),
    # Altitude vs Time
    ('real_time_seconds', 'superheavy.altitude', 'Superheavy Booster Altitude',
     'superheavy_altitude.png', 'Booster', 'Mission Time (seconds)', 'Altitude (km)'),
    ('real_time_seconds', 'starship.altitude', 'Starship Altitude',
     'starship_altitude.png', 'Starship', 'Mission Time (seconds)', 'Altitude (km)'),
    # Acceleration vs Time
    ('real_time_seconds', 'superheavy_acceleration', 'Superheavy Booster Acceleration',
     'superheavy_acceleration.png', 'Booster', 'Mission Time (seconds)', 'Acceleration (m/s²)'),
    ('real_time_seconds', 'starship_acceleration', 'Starship Acceleration',
     'starship_acceleration.png', 'Starship', 'Mission Time (seconds)', 'Acceleration (m/s²)'),
    # G-Force vs Time
    ('real_time_seconds', 'superheavy_g_force', 'Superheavy Booster G-Force',
     'superheavy_g_force.png', 'Booster', 'Mission Time (seconds)', 'G-Force (g)'),
    ('real_time_seconds', 'starship_g_force', 'Starship G-Force',
     'starship_g_force.png', 'Starship', 'Mission Time (seconds)', 'G-Force (g)'),
]

# Engine performance correlation parameters
ENGINE_PERFORMANCE_PARAMS = {
    "superheavy": {
        "x_col": "real_time_seconds",
        "y_col": "superheavy.speed",
        "color_col": "superheavy_all_active",
        "title": "Superheavy Booster Velocity vs Engine Activity",
        "x_label": "Mission Time (seconds)",
        "y_label": "Velocity (km/h)",
        "color_label": "Active Engines",
        "filename": "superheavy_velocity_vs_engines.png",
        "cmap": "viridis"
    },
    "starship": {
        "x_col": "real_time_seconds",
        "y_col": "starship.speed",
        "color_col": "starship_all_active",
        "title": "Starship Velocity vs Engine Activity",
        "x_label": "Mission Time (seconds)",
        "y_label": "Velocity (km/h)",
        "color_label": "Active Engines",
        "filename": "starship_velocity_vs_engines.png",
        "cmap": "viridis"
    }
}

# Multi-launch comparison plot parameters (using consistent naming and styling)
PLOT_MULTIPLE_LAUNCHES_PARAMS = [
    ('real_time_seconds', 'superheavy.speed', 'Superheavy Booster Velocity Comparison',
     'comparison_superheavy_velocity.png', 'Mission Time (seconds)', 'Velocity (km/h)'),
    ('real_time_seconds', 'starship.speed', 'Starship Velocity Comparison',
     'comparison_starship_velocity.png', 'Mission Time (seconds)', 'Velocity (km/h)'),
    ('real_time_seconds', 'superheavy.altitude', 'Superheavy Booster Altitude Comparison',
     'comparison_superheavy_altitude.png', 'Mission Time (seconds)', 'Altitude (km)'),
    ('real_time_seconds', 'starship.altitude', 'Starship Altitude Comparison',
     'comparison_starship_altitude.png', 'Mission Time (seconds)', 'Altitude (km)'),
    ('real_time_seconds', 'superheavy_acceleration', 'Superheavy Booster Acceleration Comparison',
     'comparison_superheavy_acceleration.png', 'Mission Time (seconds)', 'Acceleration (m/s²)'),
    ('real_time_seconds', 'starship_acceleration', 'Starship Acceleration Comparison',
     'comparison_starship_acceleration.png', 'Mission Time (seconds)', 'Acceleration (m/s²)'),
    ('real_time_seconds', 'superheavy_g_force', 'Superheavy Booster G-Force Comparison',
     'comparison_superheavy_g_force.png', 'Mission Time (seconds)', 'G-Force (g)'),
    ('real_time_seconds', 'starship_g_force', 'Starship G-Force Comparison',
     'comparison_starship_g_force.png', 'Mission Time (seconds)', 'G-Force (g)'),
    ('real_time_seconds', 'superheavy_all_active', 'Superheavy Booster Engine Activity Comparison',
     'comparison_superheavy_engines.png', 'Mission Time (seconds)', 'Active Engines'),
    ('real_time_seconds', 'starship_all_active', 'Starship Engine Activity Comparison',
     'comparison_starship_engines.png', 'Mission Time (seconds)', 'Active Engines'),
]
