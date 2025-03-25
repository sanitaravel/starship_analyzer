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
# Plotting Constants
# ------------------------------

# Engine plot parameters
ENGINE_PLOT_PARAMS = [
    # Superheavy engines
    ('real_time_seconds', 'superheavy_central_active', 'Superheavy Central Stack Engines',
     'sh_central_engines.png', 'Central Stack', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'superheavy_inner_active', 'Superheavy Inner Ring Engines',
     'sh_inner_engines.png', 'Inner Ring', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'superheavy_outer_active', 'Superheavy Outer Ring Engines',
     'sh_outer_engines.png', 'Outer Ring', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'superheavy_all_active', 'All Superheavy Engines',
     'sh_all_engines.png', 'All Engines', 'Real Time (s)', 'Active Engines (count)'),
     
    # Starship engines
    ('real_time_seconds', 'starship_rearth_active', 'Starship Raptor Earth Engines',
     'ss_rearth_engines.png', 'Raptor Earth', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'starship_rvac_active', 'Starship Raptor Vacuum Engines', 
     'ss_rvac_engines.png', 'Raptor Vacuum', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'starship_all_active', 'All Starship Engines',
     'ss_all_engines.png', 'All Engines', 'Real Time (s)', 'Active Engines (count)')
]

# Analysis plot parameters
ANALYZE_RESULTS_PLOT_PARAMS = [
    # Speed vs Time
    ('real_time_seconds', 'superheavy.speed', 'Speed of Superheavy Relative to Time',
     'sh.speed_vs_time_comparison.png', 'SH Speed', 'Real Time (s)', 'Speed (km/h)'),
    ('real_time_seconds', 'starship.speed', 'Speed of Starship Relative to Time',
     'ss.speed_vs_time_comparison.png', 'SS Speed', 'Real Time (s)', 'Speed (km/h)'),
    # Altitude vs Time
    ('real_time_seconds', 'superheavy.altitude', 'Altitude of Superheavy Relative to Time',
     'sh.altitude_vs_time_comparison.png', 'SH Altitude', 'Real Time (s)', 'Altitude (km)'),
    ('real_time_seconds', 'starship.altitude', 'Altitude of Starship Relative to Time',
     'ss.altitude_vs_time_comparison.png', 'SS Altitude', 'Real Time (s)', 'Altitude (km)'),
    # 10-Frame Distance Acceleration vs Time
    ('real_time_seconds', 'superheavy_acceleration', 'Superheavy Acceleration (30-Frame Distance)',
     'sh_acceleration_vs_time.png', 'SH Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    ('real_time_seconds', 'starship_acceleration', 'Starship Acceleration (10-Frame Distance)',
     'ss_acceleration_vs_time.png', 'SS Acceleration', 'Real Time (s)', 'Acceleration (m/s²)'),
    # G-Force vs Time
    ('real_time_seconds', 'superheavy_g_force', 'Superheavy G-Force (30-Frame Distance)',
     'sh_g_force_vs_time.png', 'SH G-Force', 'Real Time (s)', 'G-Force (g)'),
    ('real_time_seconds', 'starship_g_force', 'Starship G-Force (10-Frame Distance)',
     'ss_g_force_vs_time.png', 'SS G-Force', 'Real Time (s)', 'G-Force (g)'),
]

# Multi-launch comparison plot parameters
PLOT_MULTIPLE_LAUNCHES_PARAMS = [
    ('real_time_seconds', 'superheavy.speed', 'Comparison of Superheavy Speeds',
     'comparison_superheavy.speeds.png', 'Real Time (s)', 'Superheavy Speed (km/h)'),
    ('real_time_seconds', 'starship.speed', 'Comparison of Starship Speeds',
     'comparison_starship.speeds.png', 'Real Time (s)', 'Starship Speed (km/h)'),
    ('real_time_seconds', 'superheavy.altitude', 'Comparison of Superheavy Altitudes',
     'comparison_superheavy.altitudes.png', 'Real Time (s)', 'Superheavy Altitude (km)'),
    ('real_time_seconds', 'starship.altitude', 'Comparison of Starship Altitudes',
     'comparison_starship.altitudes.png', 'Real Time (s)', 'Starship Altitude (km)'),
    ('real_time_seconds', 'superheavy_acceleration', 'Comparison of Superheavy Accelerations',
     'comparison_superheavy_accelerations.png', 'Real Time (s)', 'Superheavy Acceleration (m/s²)'),
    ('real_time_seconds', 'starship_acceleration', 'Comparison of Starship Accelerations',
     'comparison_starship_accelerations.png', 'Real Time (s)', 'Starship Acceleration (m/s²)'),
    ('real_time_seconds', 'superheavy_g_force', 'Comparison of Superheavy G-Forces',
     'comparison_superheavy_g_forces.png', 'Real Time (s)', 'Superheavy G-Force (g)'),
    ('real_time_seconds', 'starship_g_force', 'Comparison of Starship G-Forces',
     'comparison_starship_g_forces.png', 'Real Time (s)', 'Starship G-Force (g)'),
    ('real_time_seconds', 'superheavy_all_active', 'Comparison of Superheavy Engine Activity',
     'comparison_superheavy_engines.png', 'Real Time (s)', 'Active Engines (count)'),
    ('real_time_seconds', 'starship_all_active', 'Comparison of Starship Engine Activity',
     'comparison_starship_engines.png', 'Real Time (s)', 'Active Engines (count)'),
]
