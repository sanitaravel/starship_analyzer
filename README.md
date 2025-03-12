# Starship Launch Data Extraction and Analysis

This project extracts and analyzes telemetry data from SpaceX Starship launch videos. Using advanced Optical Character Recognition (OCR), the tool extracts speed, altitude, time data, and engine status from video frames and provides comprehensive tools for plotting and comparing the extracted data across different launches.

## Features

- **Automated Data Extraction**: Extract telemetry data from individual images or video frames
- **Engine Status Detection**: Track which engines are active during flight
- **Data Visualization**: Generate detailed plots of speed, altitude, acceleration, and G-forces
- **Multi-launch Comparison**: Compare telemetry data between different Starship launches
- **Interactive UI**: User-friendly menu-driven interface
- **Data Cleaning**: Intelligent filtering and validation of extracted data

## Project Structure

```text
starship_analyzer/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── setup.py
├── main.py
├── utils.py
├── ocr/
│   ├── __init__.py
│   ├── extract_data.py
│   ├── engine_detection.py
│   └── ocr.py
├── plot/
│   ├── __init__.py
│   ├── data_processing.py
│   └── plotting.py
├── processing/
│   ├── __init__.py
│   ├── frame_processing.py
│   └── video_processing.py
└── results/
    └── launch_X/
        └── results.json
```

## Installation

1. **Prerequisites**:
   - Python 3.8 or higher
   - Tesseract OCR installed ([Installation guide](https://github.com/tesseract-ocr/tesseract))

2. **Clone the repository**:

   ```sh
   git clone https://github.com/sanitaravel/starship_analyzer.git
   cd starship_analyzer
   ```

3. **Set up the virtual environment and install dependencies**:

   ```sh
   python setup.py
   ```

## Usage

### Running the Application

Launch the application with:

```sh
python main.py
```

### Workflow

1. Place flight videos in the `flight_recordings` folder
2. Use the menu to process frames or entire videos
3. View results in real-time or analyze saved data
4. Compare multiple launches for comprehensive analysis

### Menu Options

1. **Process a single image**: Extract telemetry data from a specific image.
2. **Extract data from a random frame**: Get telemetry from a random frame within a specified timeframe.
3. **Extract data from a specified frame**: Process data from a specific frame number.
4. **Extract data from a user-specified frame**: Interactively select a frame to analyze.
5. **Process entire video**: Extract telemetry from all frames in a video.
6. **Analyze flight data**: Generate plots and visualizations from saved results.
7. **Compare multiple launches**: Create comparative visualizations between different flights.
8. **Exit**: Quit the application.

## Modules

### ocr

Handles image processing and data extraction via OCR:

- **extract_data.py**: Coordinates the extraction of telemetry data from image regions
- **ocr.py**: Core OCR functionality for recognizing text in images
- **engine_detection.py**: Detects active engines in Superheavy and Starship vehicles

### plot

Manages data cleaning, processing and visualization:

- **data_processing.py**: Validates, cleans and processes telemetry data
- **plotting.py**: Creates visualizations of speed, altitude, acceleration, and engine status

### processing

Handles video processing workflows:

- **frame_processing.py**: Functions for processing individual video frames
- **video_processing.py**: Parallel processing of video frames for efficient data extraction

### utils.py

Utility functions used across the project:

- **display_image()**: Shows image data in a window for debugging
- **extract_launch_number()**: Parses launch identifiers from file paths

## Output

The tool generates JSON data and visualizations including:

- Speed vs. time plots
- Altitude vs. time plots
- Acceleration and G-force analysis
- Engine activity timelines
- Correlation plots between engine activity and vehicle performance

## Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License with Attribution Requirement. 

This means you are free to use, modify, and distribute this software, but you must give appropriate credit to the original author, provide a link to the license, and indicate if changes were made.

See the [LICENSE](LICENSE) file for the full license text.

## Citation

If you use this software in your project or research, please cite it as:

```text
Starship Analyzer by Alexander Koshcheev
GitHub: https://github.com/sanitaravel/starship_analyzer
```
