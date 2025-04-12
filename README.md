# 🚀 Starship Analyzer

<img src="https://img.shields.io/badge/SpaceX-Starship%20Analysis-blue?style=for-the-badge" alt="Starship Launch">
<img src="https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/License-MIT%20with%20Attribution-green?style=for-the-badge" alt="License">
<img src="https://img.shields.io/badge/OCR-EasyOCR-orange?style=for-the-badge" alt="OCR">
<img src="https://img.shields.io/badge/CV-OpenCV-red?style=for-the-badge&logo=opencv" alt="Computer Vision">

A powerful Python toolkit for extracting, analyzing, and visualizing telemetry data from SpaceX Starship launch videos using computer vision and optical character recognition.

## Table of Contents

- [🚀 Starship Analyzer](#-starship-analyzer)
  - [Table of Contents](#table-of-contents)
  - [📊 What is Starship Analyzer?](#-what-is-starship-analyzer)
  - [✨ Key Features](#-key-features)
  - [🛠️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Quick Start](#quick-start)
  - [📋 Usage Guide](#-usage-guide)
    - [Getting Started](#getting-started)
    - [Workflow](#workflow)
    - [Logging System](#logging-system)
    - [Debug Mode](#debug-mode)
    - [Available Commands](#available-commands)
  - [🔍 How It Works](#-how-it-works)
  - [📂 Project Structure](#-project-structure)
  - [📊 Example Outputs](#-example-outputs)
  - [🚀 Performance Tips](#-performance-tips)
  - [👥 Contributing](#-contributing)
  - [📄 License](#-license)
  - [📚 Citation](#-citation)
  - [📧 Contact](#-contact)
  - [🛡️ Data Collection Notice](#️-data-collection-notice)

## 📊 What is Starship Analyzer?

Starship Analyzer automatically extracts critical flight data from SpaceX's Starship launch webcasts, including:

- **Speed and altitude measurements** extracted from video telemetry overlay
- **Engine ignition status and patterns** across all Raptor engines
- **Fuel level monitoring** for LOX (liquid oxygen) and CH4 (methane) in both stages
- **Timestamps and synchronization** to T-0 events
- **Acceleration and G-force calculations** for engineering analysis

The tool processes video frames in parallel, cleans the extracted data, and generates comprehensive visualizations to help you understand the performance characteristics of each launch.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Telemetry Extraction** | OCR system optimized for SpaceX's Starship telemetry overlay |
| **Engine Status Detection** | Real-time tracking of individual engine ignition states |
| **Fuel Level Analysis** | Monitoring of LOX and CH4 tank levels in Superheavy booster and Starship |
| **Performance Analysis** | Calculates derived metrics like acceleration and G-forces |
| **Multi-launch Comparison** | Compare performance metrics across different Starship test flights |
| **Interactive Visualizations** | Generate graphs and plots |
| **Parallel Processing** | Efficiently processes video frames using multi-core architecture |
| **User-friendly CLI** | Simple menu-driven interface with no programming knowledge required |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended but optional)
- 8GB+ RAM recommended for processing high-resolution videos

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/sanitaravel/starship_analyzer.git
   cd starship_analyzer
   ```

2. **Run the setup script**

   ```bash
   python setup.py
   ```

   This will:
   - Create a Python virtual environment
   - Detect CUDA availability for GPU acceleration
   - Install the right PyTorch version for your system
   - Set up all dependencies automatically

3. **Activate the virtual environment**
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

## 📋 Usage Guide

### Getting Started

1. Place your Starship launch videos in the `flight_recordings` folder
2. Run the application:

   ```bash
   python main.py
   ```

3. Follow the interactive menu to process videos and generate analyses

### Workflow

```text
Flight Recording → Frame Processing → Data Extraction → Analysis → Visualization
```

1. **Input**: Add SpaceX webcast recordings to the `flight_recordings` directory
2. **Processing**: Extract telemetry data through parallel frame processing
3. **Analysis**: Clean data, calculate derived metrics, and detect patterns
4. **Output**: Generate visualizations and comparison plots in the `results` directory

### Logging System

The application maintains detailed logs to help with troubleshooting:

- Each session creates a timestamped log file in the `logs` directory
- Log files follow the format `starship_analyzer_YYYYMMDD_HHMMSS.log`
- Console output shows essential information while full details are saved to log files
- System hardware and software details are logged at startup for troubleshooting
- For debugging issues, check the latest log file in the `logs` directory

### Debug Mode

Debug mode provides enhanced logging and diagnostic information to help troubleshoot issues:

- Enable/disable debug mode directly from the main menu using the "Toggle Debug Mode" option
- When enabled, detailed diagnostic information is logged about:
  - OCR processing and text extraction
  - Engine detection with pixel values
  - Memory usage and CUDA device information
  - Detailed data processing steps and statistics
- Use debug mode when:
  - Troubleshooting extraction issues with specific frames
  - Diagnosing performance problems or accuracy issues
  - Developing new features or fixing bugs
  - Analyzing the internal behavior of the application

Debug mode logs are more verbose but provide valuable insights when resolving complex issues.

### Available Commands

The interactive menu offers several options:

| Command | Description |
|---------|-------------|
| **Process random video frame** | Test extraction on a single frame to validate setup |
| **Process complete video** | Extract data from all frames in a recording |
| **Visualize flight data** | Generate plots from processed launch data |
| **Visualize multiple launches** | Compare metrics across different flights |

## 🔍 How It Works

Starship Analyzer uses a multi-stage pipeline:

1. **Frame Extraction**: Video frames are extracted and queued for processing
2. **OCR Processing**: Specialized regions of interest (ROIs) are analyzed to extract telemetry
3. **Engine Detection**: Computer vision techniques identify active engines
4. **Fuel Level Detection**: Analysis of propellant gauge indicators for LOX and CH4 tank levels
5. **Data Cleaning**: Statistical methods remove outliers and noise
6. **Analysis**: Calculates acceleration, G-forces, and performance metrics
7. **Visualization**: Generates plots showing vehicle performance, engine status, and fuel consumption

## 📂 Project Structure

```text
starship_analyzer/
├── ocr/                # Optical Character Recognition subsystem
│   ├── engine_detection.py  # Engine state detection
│   ├── extract_data.py      # Main data extraction logic
│   └── ocr.py               # Text recognition from telemetry
├── plot/               # Data processing and visualization tools
│   ├── data_processing.py   # Data cleaning and calculation
│   └── plotting.py          # Graph generation
├── processing/         # Video and frame processing engine
│   ├── frame_processing.py  # Single frame analysis
│   └── video_processing.py  # Batch processing of videos
├── flight_recordings/  # Input directory for launch videos
├── results/            # Output directory for processed data
├── main.py             # Application entry point
└── setup.py            # Installation and configuration script
```

## 📊 Example Outputs

The tool generates several types of visualizations:

- **Telemetry Plots**: Speed and altitude over time with smooth trend lines
- **Performance Analysis**: Acceleration and G-force profiles with NASA threshold lines
- **Engine Activity**: Timelines showing which engines are firing with color-coded indicators
- **Fuel Consumption**: Tracking of LOX and CH4 levels in both vehicle stages over time
- **Correlation Analysis**: Relationship between engine patterns and vehicle performance
- **Launch Comparisons**: Side-by-side analysis of different Starship flights for trend analysis

## 🚀 Performance Tips

- Processing high-resolution videos requires significant computing resources
- GPU acceleration dramatically improves OCR processing speed (5-10x faster)
- Adjust batch sizes in the menu for optimal performance on your system:
  - For systems with <16GB RAM: Use batch sizes of 5-10
  - For systems with >16GB RAM: Batch sizes up to 20-30 are effective

## 👥 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License with Attribution Requirement.

You may freely use and modify this software, provided you:

- Include the original copyright notice
- Provide attribution to the original author
- Indicate if changes were made

See the [LICENSE](LICENSE) file for complete details.

## 📚 Citation

If you use this software in academic or research contexts, please cite it as:

```text
Koshcheev, A. (2025). Starship Analyzer: Telemetry extraction and analysis tool 
for SpaceX Starship launches. GitHub. https://github.com/sanitaravel/starship_analyzer
```

## 📧 Contact

Alexander Koshcheev - [GitHub Profile](https://github.com/sanitaravel)

Project Link: [https://github.com/sanitaravel/starship_analyzer](https://github.com/sanitaravel/starship_analyzer)

## 🛡️ Data Collection Notice

**System Information**: At the start of each session, Starship Analyzer collects basic system information including:

- Hardware details (CPU, RAM size, GPU specifications)
- Platform information (OS version, architecture)
- Python and critical library versions
- CUDA availability and version

This information is stored **only in your local log files** and is used exclusively for:

- Troubleshooting technical issues
- Optimizing performance for your hardware
- Debugging version-specific problems

The application does not transmit any data to external servers or share this information with third parties. All logs remain on your local system unless you explicitly share them when seeking technical support.

You can inspect the collected information in the log files located in the `logs` directory.
