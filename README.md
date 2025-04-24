# 🚀 Starship Analyzer

![Starship Launch](https://img.shields.io/badge/SpaceX-Starship%20Analysis-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT%20with%20Attribution-green?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-EasyOCR-orange?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/CV-OpenCV-red?style=for-the-badge&logo=opencv)

A powerful Python toolkit for extracting, analyzing, and visualizing telemetry data from SpaceX Starship launch webcasts using computer vision and optical character recognition. This tool helps engineers, space enthusiasts, and analysts track performance metrics and compare data across different Starship test flights.

## Table of Contents

- [🚀 Starship Analyzer](#-starship-analyzer)
  - [Table of Contents](#table-of-contents)
  - [📊 What is Starship Analyzer?](#-what-is-starship-analyzer)
  - [✨ Key Features](#-key-features)
  - [📚 Documentation Wiki](#-documentation-wiki)
  - [🛠️ Quick Installation](#️-quick-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [📋 Basic Usage](#-basic-usage)
  - [🔍 How It Works](#-how-it-works)
  - [👥 Contributing](#-contributing)
  - [📄 License](#-license)
  - [📧 Contact](#-contact)

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
| **Interactive Visualizations** | Generate detailed graphs and plots with zoom capabilities, tooltips, and exportable formats |
| **Parallel Processing** | Efficiently processes video frames using multi-core architecture |
| **User-friendly CLI** | Simple menu-driven interface with no programming knowledge required |

## 📚 Documentation Wiki

For detailed documentation, please visit our [GitHub Wiki](https://github.com/sanitaravel/starship_analyzer/wiki) which covers:

- [Installation Guide](https://github.com/sanitaravel/starship_analyzer/wiki/Installation) - Detailed setup instructions
- [Usage Guide](https://github.com/sanitaravel/starship_analyzer/wiki/Usage-Guide) - Complete usage instructions
- [How It Works](https://github.com/sanitaravel/starship_analyzer/wiki/How-It-Works) - Technical explanation
- [API Documentation](https://github.com/sanitaravel/starship_analyzer/wiki/API-Documentation) - For developers
- [Contributing Guidelines](https://github.com/sanitaravel/starship_analyzer/wiki/Contributing) - How to contribute
- [FAQ](https://github.com/sanitaravel/starship_analyzer/wiki/FAQ) - Common questions
- [Troubleshooting](https://github.com/sanitaravel/starship_analyzer/wiki/Troubleshooting) - Solutions to common problems

## 🛠️ Quick Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended but optional)
- 8GB+ RAM recommended for processing high-resolution videos

### Setup

```bash
# Clone the repository
git clone https://github.com/sanitaravel/starship_analyzer.git
cd starship_analyzer

# Run the setup script
python setup.py

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

For detailed installation instructions, including manual setup and troubleshooting, see the [Installation Wiki](https://github.com/sanitaravel/starship_analyzer/wiki/Installation).

## 📋 Basic Usage

1. Place your Starship launch videos in the `flight_recordings` folder
2. Run the application:

   ```bash
   python main.py
   ```

3. Follow the interactive menu to process videos and generate analyses

The workflow follows this pattern:
```text
Flight Recording → Frame Processing → Data Extraction → Analysis → Visualization
```

1. **Input**: Add SpaceX webcast recordings to the `flight_recordings` directory
2. **Processing**: Extract telemetry data through parallel frame processing
3. **Analysis**: Clean data, calculate derived metrics, and detect patterns
4. **Output**: Generate visualizations and comparison plots in the `results` directory

## 🔍 How It Works

Starship Analyzer uses a multi-stage pipeline:

1. **Frame Extraction**: Video frames are extracted and queued for processing
2. **OCR Processing**: Specialized regions of interest (ROIs) are analyzed to extract telemetry
3. **Engine Detection**: Computer vision techniques identify active engines
4. **Fuel Level Detection**: Analysis of propellant gauge indicators for LOX and CH4 tank levels
5. **Data Cleaning**: Statistical methods remove outliers and noise
6. **Analysis**: Calculates acceleration, G-forces, and performance metrics
7. **Visualization**: Generates plots showing vehicle performance, engine status, and fuel consumption

The processed data is available through an interactive visualization interface that lets you explore:

- Time-synchronized telemetry readings
- Engine activation patterns
- Fuel consumption rates
- Performance metrics across different flight phases
- Comparative analysis between multiple launches

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

## 📧 Contact

Alexander Koshcheev - [GitHub Profile](https://github.com/sanitaravel)

Project Link: [https://github.com/sanitaravel/starship_analyzer](https://github.com/sanitaravel/starship_analyzer)
