import os
import sys
import platform
import shutil
import time
import subprocess
from pathlib import Path

from .utilities import print_step, print_success, print_warning, print_error, print_debug

def try_force_remove_venv(venv_dir, debug=False):
    """
    Try alternative methods to forcibly remove the virtual environment.
    
    Args:
        venv_dir (str): Path to the virtual environment directory
        debug (bool): Whether to enable debug output
    """
    try:
        if platform.system() == "Windows":
            # Use subprocess to run system commands to force directory removal
            cmd = ["cmd", "/c", f"rmdir /s /q {venv_dir}"]
            print_debug(f"Running force removal command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False, capture_output=True)
            if debug:
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        else:
            # For Unix systems, use rm -rf
            cmd = ["rm", "-rf", venv_dir]
            print_debug(f"Running force removal command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False, capture_output=True)
            if debug:
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
    except Exception as e:
        print_warning(f"Force removal method also failed: {e}")

def create_virtual_environment(step_num=1, unattended=False, recreate=False, keep=False, debug=False):
    """
    Create a Python virtual environment.
    
    Args:
        step_num (int or float): Step number to display in the console output
        unattended (bool): Whether to run in unattended mode
        recreate (bool): Whether to recreate the virtual environment if it exists
        keep (bool): Whether to keep the virtual environment if it exists
        debug (bool): Whether to enable debug output
    
    Returns:
        bool: True if virtual environment was created successfully, False otherwise.
    """
    print_step(step_num, "Creating Python virtual environment")
    venv_dir = "venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print_warning(f"Virtual environment already exists at '{venv_dir}'")
        print_debug(f"venv directory exists: {os.path.abspath(venv_dir)}")
        
        # Determine what to do with existing venv
        if unattended:
            if recreate:
                print_warning("Unattended mode: Recreating virtual environment")
                should_recreate = True
                print_debug("unattended=True, recreate=True -> should_recreate=True")
            elif keep:
                print_warning("Unattended mode: Using existing virtual environment")
                print_debug("unattended=True, keep=True -> using existing venv")
                return True
            else:
                # Default behavior for unattended mode is to keep
                print_warning("Unattended mode: Using existing virtual environment (default)")
                print_debug("unattended=True, keep=False, recreate=False -> using existing venv (default)")
                return True
        else:
            print_debug("Interactive mode - asking user for input")
            should_recreate = input("Do you want to recreate it? (y/n): ").lower().strip() == 'y'
            print_debug(f"User input for recreation: should_recreate={should_recreate}")
        
        if should_recreate or recreate:
            try:
                # Try to remove the virtual environment
                print_warning("Removing existing virtual environment...")
                print_debug(f"Attempting to remove directory: {os.path.abspath(venv_dir)}")
                
                # On Windows, try to handle permission errors
                if platform.system() == "Windows":
                    # First, try to gracefully remove
                    try:
                        shutil.rmtree(venv_dir)
                        print_debug("Successfully removed venv directory with shutil.rmtree")
                    except PermissionError as e:
                        print_debug(f"Permission error encountered: {e}")
                        if "Access is denied" in str(e):
                            print_warning("Access denied error detected. This usually happens when:")
                            print_warning("1. A terminal/console is still using the virtual environment")
                            print_warning("2. An IDE or editor is accessing files in the virtual environment")
                            print_warning("3. A Python process from this environment is still running")
                            
                            if not unattended:
                                print_warning("Please close all terminals and applications that might be using the virtual environment")
                                retry = input("Try again after closing applications? (y/n): ").lower().strip() == 'y'
                                print_debug(f"User input for retry: {retry}")
                                if retry:
                                    # Wait a moment and try again
                                    print_debug("Waiting 2 seconds before retry...")
                                    time.sleep(2)
                                    try:
                                        shutil.rmtree(venv_dir)
                                        print_success("Successfully removed virtual environment on second attempt")
                                        print_debug("Second attempt with shutil.rmtree succeeded")
                                    except Exception as e2:
                                        print_error(f"Failed again: {e2}")
                                        print_debug(f"Second attempt failed with: {e2}")
                                        print_warning("Let's try using a more aggressive approach...")
                                        try_force_remove_venv(venv_dir, debug)
                                        if not os.path.exists(venv_dir):
                                            print_success("Successfully removed virtual environment using force method")
                                            print_debug("Force removal succeeded")
                                        else:
                                            print_error("Could not remove the virtual environment")
                                            print_warning("Please manually delete the folder and run this script again")
                                            print_debug(f"Force removal failed, directory still exists: {os.path.exists(venv_dir)}")
                                            return False
                                else:
                                    print_warning("Keeping existing virtual environment")
                                    print_debug("User chose not to retry, keeping existing environment")
                                    return True
                            else:
                                print_warning("Unattended mode: Trying alternative removal method...")
                                print_debug("In unattended mode, trying force removal immediately")
                                try_force_remove_venv(venv_dir, debug)
                                if not os.path.exists(venv_dir):
                                    print_success("Successfully removed virtual environment using force method")
                                    print_debug("Force removal in unattended mode succeeded")
                                else:
                                    print_error("Could not remove the virtual environment in unattended mode")
                                    print_debug("Force removal in unattended mode failed")
                                    if recreate:  # Only return False if we really wanted to recreate
                                        return False
                                    print_warning("Proceeding with existing virtual environment")
                                    return True
                        else:
                            raise
                else:
                    # Non-Windows: just use rmtree
                    print_debug("Non-Windows OS, using standard rmtree")
                    shutil.rmtree(venv_dir)
                    
                print_success("Removed existing virtual environment")
            except Exception as e:
                print_error(f"Failed to remove existing virtual environment: {e}")
                print_debug(f"Exception during venv removal: {e}")
                if unattended and not recreate:
                    print_warning("Unattended mode: Continuing with existing virtual environment despite error")
                    print_debug("In unattended mode and not forcing recreate, continuing with existing venv")
                    return True
                return False
        else:
            print_warning("Using existing virtual environment")
            print_debug("User chose not to recreate, using existing environment")
            return True
    
    try:
        cmd = [sys.executable, "-m", "venv", venv_dir]
        print_debug(f"Creating virtual environment with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print_success(f"Created virtual environment in '{venv_dir}' directory")
        print_debug(f"Virtual environment successfully created at: {os.path.abspath(venv_dir)}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        print_debug(f"CalledProcessError during venv creation: {e}")
        print_debug(f"Return code: {e.returncode}, Output: {e.output if hasattr(e, 'output') else 'None'}")
        return False
    except Exception as e:
        print_error(f"Unexpected error creating virtual environment: {e}")
        print_debug(f"Unexpected exception during venv creation: {e}")
        return False

def create_required_directories(step_num=2, debug=False):
    """
    Create directories required for the application.
    
    Args:
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to enable debug output
    """
    print_step(step_num, "Creating required directories")
    directories = ['flight_recordings', 'results', '.tmp', 'logs']
    
    for directory in directories:
        try:
            dir_path = Path(directory)
            print_debug(f"Creating directory: {dir_path.absolute()}")
            dir_path.mkdir(exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory '{directory}': {e}")
            print_debug(f"Exception during directory creation: {e}")
