# Starship Analyzer

A small toolkit to analyze Starship flight recordings and extract telemetry and events from video.

This repository contains video-processing utilities, OCR-based telemetry extraction, plotting, and helper scripts used to analyze flight recordings (stored under `flight_recordings/`). It is intended for local analysis and lightweight automation (for example, publishing notable events).

## Key capabilities

- Extract text overlays and telemetry from videos using OCR (see `ocr/`).
- Process frames and regions-of-interest (ROIs) defined in `configs/`.
- Produce plots and analysis artifacts into `plot/` and `results/`.
- Helper scripts for downloading, preprocessing and broadcasting results (see `download/` and `twitter_broadcast.test_script.py`).

## Repository layout (important files/folders)

- `main.py` — primary entry point for running analyses.
- `download/` — video download and preparation helpers.
- `ocr/` — OCR helpers and models integration.
- `configs/` — ROI and configuration JSON files (e.g. `default_rois.json`).
- `flight_recordings/` — sample and raw MP4 recordings used for analysis.
- `plot/`, `results/` — output directories for plots and results.
- `logs/` — runtime log files created when running the tools.
- `tests/` — unit and integration tests.
- `requirements.txt` — Python dependencies.

## Quick start

### Prerequisites

- Python 3.8+ is sufficient; Python 3.11/3.12 are recommended for best compatibility.
- FFmpeg installed and available on your `PATH` (used for frame extraction).
- (Optional) NVIDIA GPU with CUDA support — recommended if you plan to use GPU-accelerated OCR or ML models for significantly better performance.

### Install

Run the included installer (recommended). The repository provides a small installer script `setup.py` that wraps the `setup` package and automates virtual environment creation, dependency installation (including CUDA-aware PyTorch), and verification.

```pwsh
# Interactive installer (recommended)
python setup.py

# Example: unattended installation and force CPU-only PyTorch
python setup.py --unattended --force-cpu

# To update an existing installation
python setup.py --update
```

You can also run the installer directly as a module (equivalent):

```pwsh
python -c "import setup; setup.run_setup()"
```

If you prefer to manage the virtual environment and installation manually, follow these steps instead:

Create a virtual environment and install Python dependencies:

```pwsh
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Available installer flags (supported by `setup.py` / `setup.run_setup`):

- `--update` : Update dependencies in an existing virtual environment.
- `--force-cpu` : Force CPU-only installations (skip CUDA/PyTorch GPU variant).
- `--unattended` : Run without interactive prompts where possible.
- `--recreate` : Recreate the virtual environment if it already exists.
- `--keep` : Keep the existing virtual environment and skip recreation.
- `--debug` : Show detailed installation output for troubleshooting.

Note: On systems where the installer tries to install NVIDIA drivers or CUDA, you may need elevated privileges (sudo on Linux or administrator on Windows). The installer will guide you and print links for manual driver/CUDA downloads when automatic installation isn't appropriate.

### Run an analysis

The repository exposes `main.py`, which coordinates processing: parsing recordings (frame extraction, OCR-based telemetry parsing and event extraction) and generating plots/analysis artifacts saved to `plot/` and `results/`.

```pwsh
# Interactive menu (no args)
python main.py
```

`main.py` also supports lightweight profiling to help diagnose performance hotspots:

- `--profile` / `-p` : enable cProfile and write stats to `profile.stats` (or provide a filename).
- `--profile-print` : when profiling, print top functions by cumulative time after the run.
- `--profile-top N` : number of top functions to print (default 50).

See `python main.py --help` for the full set of options and menu-driven features.

## Configuration and ROIs

Region-of-interest (ROI) and flight-specific settings live in `configs/`. The file `configs/default_rois.json` provides a baseline. To analyze a recording with a specific ROI configuration, point the analysis to that JSON or copy/modify it for a flight.

Example ROI configuration (JSON) matching the project's schema:

```json
{
  "version": 2,
  "time_unit": "frames",
  "rois": [
    {
      "id": "SS_SPEED",
      "label": "Starship Speed",
      "x": 1544,
      "y": 970,
      "w": 114,
      "h": 37,
      "start_time": 83301,
      "end_time": 88329,
      "match_to_role": "ss_speed"
    },
    {
      "id": "TIME",
      "label": "Time Display",
      "x": 827,
      "y": 968,
      "w": 265,
      "h": 44,
      "start_time": 83301,
      "end_time": 168659,
      "match_to_role": "time"
    }
  ]
}
```

### ROI JSON field reference

Below are all fields observed in the `configs/` ROI files and their meanings. Use this as a quick reference when creating or editing ROI JSON files.

- `version` (integer)
  - Schema version for the ROI file. Increment when the format changes. Examples: `1`, `2`.

- `time_unit` (string)
  - Unit used for `start_time` and `end_time`. Common values: `frames`, `seconds`.

- `rois` (array of objects)
  - List of ROI objects; each object describes a single region to process.

Per-ROI object fields

- `id` (string)
  - Short unique identifier for the ROI (e.g., `SS_SPEED`, `TIME`). Used to reference the ROI in logs and outputs.

- `label` (string)
  - Human-friendly description of the ROI (e.g., `Starship Speed`). Used in UIs and reports.

- `x` (integer)
  - X coordinate (pixels) of the top-left corner of the ROI, relative to the frame's left edge.

- `y` (integer)
  - Y coordinate (pixels) of the top-left corner of the ROI, relative to the frame's top edge.

- `w` (integer)
  - Width (pixels) of the ROI rectangle.

- `h` (integer)
  - Height (pixels) of the ROI rectangle.

- `start_time` (integer or null)
  - When the ROI becomes active. Interpreted in the unit specified by `time_unit`. Use `null` if the ROI is active from the beginning.

- `end_time` (integer or null)
  - When the ROI stops being active. Interpreted in the unit specified by `time_unit`. Use `null` if the ROI remains active until the end of the recording.

- `match_to_role` (string)
  - Logical role name used by the analyzer to map OCR or detections to specific telemetry fields (for example `ss_speed`, `sh_altitude`, `time`).

Notes and best practices

- Coordinates and sizes are integer pixel values — double-check these values at the video resolution you are analyzing (e.g., 1920x1080 vs 1280x720).
- Use `start_time`/`end_time` to avoid running OCR on regions that are not present for the whole recording (this speeds up processing).
- Keep `id` values unique within a single file. You may reuse `match_to_role` values across ROIs if they map to the same telemetry field across different time ranges.
- When converting `seconds` to `frames`, multiply seconds by the video's frames-per-second (FPS). The project does not assume a default FPS — supply `time_unit` and values consistent with your workflow.
- `time_unit` indicates the unit used for `start_time`/`end_time` (e.g., `frames` or `seconds`).
- Coordinates are in pixels (`x`, `y`, `w`, `h`) relative to the top-left of the frame.
- `start_time` and `end_time` define when the ROI is active in the recording.
- Save custom configs into the `configs/` directory and name them clearly (e.g., `flight_9_rois.json`).
- The analyzer will read the selected JSON and apply OCR or detection routines to each ROI; check logs in `logs/` for ROI processing messages.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests. For code changes, fork the repo, create a feature branch, and open a pull request against `master`.

## License

See the `LICENSE` file in the repository root for license details.

## Contact

If you need help running the project locally, open an issue or contact the repository owner.
