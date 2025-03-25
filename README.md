# 🚀 Starship Analyzer

![Starship Launch](https://img.shields.io/badge/SpaceX-Starship%20Analysis-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT%20with%20Attribution-green?style=for-the-badge)

A powerful Python tool for extracting, analyzing, and visualizing telemetry data from SpaceX Starship launch videos using advanced computer vision and OCR techniques.

- [🚀 Starship Analyzer](#-starship-analyzer)
  - [📊 What is Starship Analyzer?](#-what-is-starship-analyzer)
  - [✨ Key Features](#-key-features)
  - [🛠️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Quick Start](#quick-start)
  - [📋 Usage Guide](#-usage-guide)
    - [Getting Started](#getting-started)
    - [Workflow](#workflow)
    - [Available Commands](#available-commands)
  - [🔍 How It Works](#-how-it-works)
  - [📂 Project Structure](#-project-structure)
  - [📊 Example Outputs](#-example-outputs)
  - [🚀 Performance Tips](#-performance-tips)
  - [👥 Contributing](#-contributing)
  - [📄 License](#-license)
  - [📚 Citation](#-citation)
  - [📧 Contact](#-contact)

## 📊 What is Starship Analyzer?

Starship Analyzer automatically extracts critical flight data from SpaceX's Starship launch webcasts, including:

- Speed and altitude measurements
- Engine ignition status and patterns
- Precise timestamps and synchronization
- Acceleration and G-force calculations

The tool processes video frames in parallel, cleans the extracted data, and generates comprehensive visualizations to help you understand the performance characteristics of each launch.

## ✨ Key Features

- **Advanced Telemetry Extraction**: Robust OCR system optimized for SpaceX's telemetry overlay
- **Engine Status Detection**: Real-time tracking of individual engine ignition states
- **Performance Analysis**: Calculates derived metrics like acceleration and G-forces
- **Multi-launch Comparison**: Compare performance metrics across different Starship test flights
- **Interactive Visualizations**: Generate publication-quality graphs and plots
- **Parallel Processing**: Efficiently processes video frames using multi-core architecture
- **User-friendly CLI**: Simple menu-driven interface with no programming knowledge required

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
   - Create a virtual environment
   - Detect CUDA availability
   - Install the right PyTorch version for your system
   - Set up all dependencies

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

### Available Commands

The interactive menu offers several options:

- **Process random video frame**: Test extraction on a single frame
- **Process complete video**: Extract data from all frames in a recording
- **Visualize flight data**: Generate plots from processed launch data
- **Visualize multiple launches**: Compare metrics across different flights

## 🔍 How It Works

Starship Analyzer uses a multi-stage pipeline:

1. **Frame Extraction**: Video frames are extracted and queued for processing
2. **OCR Processing**: Specialized regions of interest (ROIs) are analyzed to extract telemetry
3. **Engine Detection**: Computer vision techniques identify active engines
4. **Data Cleaning**: Statistical methods remove outliers and noise
5. **Analysis**: Calculates acceleration, G-forces, and performance metrics
6. **Visualization**: Generates plots showing vehicle performance and engine status

## 📂 Project Structure

```text
starship_analyzer/
├── ocr/                # Optical Character Recognition subsystem
├── plot/               # Data processing and visualization tools
├── processing/         # Video and frame processing engine
├── flight_recordings/  # Input directory for launch videos
├── results/            # Output directory for processed data
├── main.py             # Application entry point
└── setup.py            # Installation and configuration script
```

## 📊 Example Outputs

The tool generates several types of visualizations:

- **Telemetry Plots**: Speed and altitude over time
- **Performance Analysis**: Acceleration and G-force profiles
- **Engine Activity**: Timelines showing which engines are firing
- **Correlation Analysis**: Relationship between engine patterns and vehicle performance
- **Launch Comparisons**: Side-by-side analysis of different Starship flights

## 🚀 Performance Tips

- Processing high-resolution videos requires significant computing resources
- GPU acceleration dramatically improves OCR processing speed
- Adjust batch sizes in the menu for optimal performance on your system
- For large videos, consider processing in multiple sessions

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
Koshcheev, A. (2025). Starship Analyzer: Telemetry extraction and analysis 
tool for SpaceX Starship launches. GitHub. 
https://github.com/sanitaravel/starship_analyzer
```

## 📧 Contact

Alexander Koshcheev - [GitHub Profile](https://github.com/sanitaravel)

Project Link: [https://github.com/sanitaravel/starship_analyzer](https://github.com/sanitaravel/starship_analyzer)
