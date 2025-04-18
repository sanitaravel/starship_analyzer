"""
Starship Analyzer Setup Script

This script serves as the entry point for setting up the Starship Analyzer application.
It uses the 'setup' module to perform the actual setup tasks.

Usage:
    python setup.py              - Run the standard setup process
    python setup.py --update     - Update the application's dependencies
    python setup.py --force-cpu  - Force CPU-only installation
    python setup.py --unattended - Run in unattended mode (no user interaction)
    python setup.py --recreate   - Recreate the virtual environment
    python setup.py --keep       - Keep the existing virtual environment
    python setup.py --debug      - Show detailed installation output
    python setup.py --help       - Show all available options
"""

import argparse
from setup import run_setup

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Starship Analyzer Setup")
    parser.add_argument("--update", action="store_true", help="Update the application")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only installation")
    parser.add_argument("--unattended", action="store_true", help="Run in unattended mode")
    parser.add_argument("--recreate", action="store_true", help="Recreate virtual environment")
    parser.add_argument("--keep", action="store_true", help="Keep existing virtual environment")
    parser.add_argument("--debug", action="store_true", help="Show detailed installation output")
    
    args = parser.parse_args()
    run_setup(args)