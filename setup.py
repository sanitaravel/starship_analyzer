"""
Starship Analyzer Setup Script

This script serves as the entry point for setting up the Starship Analyzer application.
It uses the 'setup' module to perform the actual setup tasks.
"""

from setup import run_setup
import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Starship Analyzer Setup')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for verbose installation logs')
    args = parser.parse_args()
    
    # Pass debug flag to the setup function
    run_setup(debug=args.debug)