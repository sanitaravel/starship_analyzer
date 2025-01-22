# Starship Launch Data Extraction and Analysis

This project is designed to extract and analyze data from Starship launch videos. It uses Optical Character Recognition (OCR) to extract speed, altitude, and time data from video frames and provides tools for plotting and comparing the extracted data.

## Project Structure

```text
starship_analyzer/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── setup.py
├── main.py
├── ocr/
│   ├── __init__.py
│   ├── extract_data.py
│   └── ocr.py
├── plot/
│   ├── __init__.py
│   ├── data_processing.py
│   └── plotting.py
├── processing/
│   ├── __init__.py
│   ├── frame_processing.py
│   └── video_processing.py
└── utils.py
```

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/sanitaravel/starship_analyzer.git
   cd starship-launch-data
   ```

2. **Set up the virtual environment and install dependencies:**

   ```sh
   python setup.py
   ```

3. **Set up the environment variables:**
   - Copy the `.env.example` file to `.env` and set the path to the Tesseract executable.

   ```sh
   cp .env.example .env
   ```

## Usage

### Running the Main Script

The main script provides a step-by-step menu for processing images and videos, analyzing flight data, and comparing multiple launches.

```sh
python main.py
```

### Menu Options

1. **Process a single image:** Extract data from a specified image file.
2. **Extract data from a random frame in a video:** Extract data from a random frame within a specified timeframe in a video.
3. **Extract data from a specified frame in a video:** Extract data from a specific frame in a video.
4. **Extract data from a user-specified frame in a video:** Extract data from a user-specified frame in a video.
5. **Run through whole video:** Iterate through all frames in a video and extract data.
6. **Analyze flight data:** Analyze and plot data from a JSON file containing the results.
7. **Compare multiple launches:** Compare data from multiple launches and plot the results.
8. **Exit:** Exit the program.

## Modules

### ocr

- **Purpose:** Handle OCR and data extraction from images.
- **Files:**
  - `extract_data.py`: Functions for extracting data from images.
  - `ocr.py`: Functions for OCR processing.

### plot

- **Purpose:** Handle plotting and analysis of extracted data.
- **Files:**
  - `data_processing.py`: Functions for validating and cleaning data.
  - `plotting.py`: Functions for plotting and comparing flight data.

### processing

- **Purpose:** Handle video processing tasks.
- **Files:**
  - `frame_processing.py`: Functions for processing individual frames.
  - `video_processing.py`: Functions for processing video frames and extracting data.

### `utils.py`

- **Purpose:** Utility functions used across the project.
- **Functions:**
  - `display_image(image: np.ndarray, text: str) -> None`: Display an image in a window.
  - `extract_launch_number(json_path: str) -> str`: Extract the launch number from a JSON file path.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
