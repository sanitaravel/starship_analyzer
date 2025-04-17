import os
import sys
import platform
import shutil
import time
import subprocess
from pathlib import Path

from .utilities import print_step, print_success, print_info, print_warning, print_error

def try_force_remove_venv(venv_dir, debug=False):
    """
    Try alternative methods to forcibly remove the virtual environment.
    
    Args:
        venv_dir (str): Path to the virtual environment directory
        debug (bool): Whether to show detailed output
    """
    try:
        if platform.system() == "Windows":
            # Use subprocess to run system commands to force directory removal
            if debug:
                subprocess.run(["cmd", "/c", f"rmdir /s /q {venv_dir}"], check=False)
            else:
                subprocess.run(["cmd", "/c", f"rmdir /s /q {venv_dir}"], 
                           check=False, capture_output=True)
        else:
            # For Unix systems, use rm -rf
            if debug:
                subprocess.run(["rm", "-rf", venv_dir], check=False)
            else:
                subprocess.run(["rm", "-rf", venv_dir], 
                           check=False, capture_output=True)
    except Exception as e:
        print_warning(f"Force removal method also failed")
        if debug:
            print_warning(f"Error details: {e}")

def create_virtual_environment(step_num=1, unattended=False, recreate=False, keep=False, debug=False):
    """
    Create a Python virtual environment.
    
    Args:
        step_num (int or float): Step number to display in the console output
        unattended (bool): Whether to run in unattended mode
        recreate (bool): Whether to recreate the virtual environment if it exists
        keep (bool): Whether to keep the virtual environment if it exists
        debug (bool): Whether to show detailed output
    
    Returns:
        bool: True if virtual environment was created successfully, False otherwise.
    """
    print_step(step_num, "Creating Python virtual environment")
    venv_dir = "venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print_warning(f"Virtual environment already exists at '{venv_dir}'")
        
        # Determine what to do with existing venv
        if unattended:
            if recreate:
                print_warning("Unattended mode: Recreating virtual environment")
                should_recreate = True
            elif keep:
                print_warning("Unattended mode: Using existing virtual environment")
                return True
            else:
                # Default behavior for unattended mode is to keep
                print_warning("Unattended mode: Using existing virtual environment (default)")
                return True
        else:
            should_recreate = input("Do you want to recreate it? (y/n): ").lower().strip() == 'y'
        
        if should_recreate or recreate:
            try:
                # Try to remove the virtual environment
                print_warning("Removing existing virtual environment...")
                
                # On Windows, try to handle permission errors
                if platform.system() == "Windows":
                    # First, try to gracefully remove
                    try:
                        shutil.rmtree(venv_dir)
                    except PermissionError as e:
                        if "Access is denied" in str(e):
                            print_warning("Access denied error detected. This usually happens when:")
                            print_warning("1. A terminal/console is still using the virtual environment")
                            print_warning("2. An IDE or editor is accessing files in the virtual environment")
                            print_warning("3. A Python process from this environment is still running")
                            
                            if not unattended:
                                print_warning("Please close all terminals and applications that might be using the virtual environment")
                                retry = input("Try again after closing applications? (y/n): ").lower().strip() == 'y'
                                if retry:
                                    # Wait a moment and try again
                                    time.sleep(2)
                                    try:
                                        shutil.rmtree(venv_dir)
                                        print_success("Successfully removed virtual environment on second attempt")
                                    except Exception as e2:
                                        print_error(f"Failed again")
                                        if debug:
                                            print_error(f"Error details: {e2}")
                                        print_warning("Let's try using a more aggressive approach...")
                                        try_force_remove_venv(venv_dir, debug=debug)
                                        if not os.path.exists(venv_dir):
                                            print_success("Successfully removed virtual environment using force method")
                                        else:
                                            print_error("Could not remove the virtual environment")
                                            print_warning("Please manually delete the folder and run this script again")
                                            return False
                                else:
                                    print_warning("Keeping existing virtual environment")
                                    return True
                            else:
                                print_warning("Unattended mode: Trying alternative removal method...")
                                try_force_remove_venv(venv_dir, debug=debug)
                                if not os.path.exists(venv_dir):
                                    print_success("Successfully removed virtual environment using force method")
                                else:
                                    print_error("Could not remove the virtual environment in unattended mode")
                                    if recreate:  # Only return False if we really wanted to recreate
                                        return False
                                    print_warning("Proceeding with existing virtual environment")
                                    return True
                        else:
                            raise
                else:
                    # Non-Windows: just use rmtree
                    shutil.rmtree(venv_dir)
                    
                print_success("Removed existing virtual environment")
            except Exception as e:
                print_error(f"Failed to remove existing virtual environment")
                if debug:
                    print_error(f"Error details: {e}")
                if unattended and not recreate:
                    print_warning("Unattended mode: Continuing with existing virtual environment despite error")
                    return True
                return False
        else:
            print_warning("Using existing virtual environment")
            return True
    
    try:
        print_info("Creating new virtual environment...")
        if debug:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], 
                          check=True, capture_output=True, text=True)
        print_success(f"Created virtual environment in '{venv_dir}' directory")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment")
        if debug:
            print_error(f"Error details: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error creating virtual environment")
        if debug:
            print_error(f"Error details: {e}")
        return False

def create_required_directories(step_num=2):
    """
    Create directories required for the application.
    
    Args:
        step_num (int or float): Step number to display in the console output
    """
    print_step(step_num, "Creating required directories")
    directories = ['flight_recordings', 'results', '.tmp', 'logs']
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory '{directory}': {e}")
