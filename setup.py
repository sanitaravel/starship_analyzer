import os
import subprocess
import sys
import platform
import shutil
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_step(step_num, message):
    """Print a step in the setup process."""
    print(f"{BOLD}Step {step_num}: {message}{RESET}")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def check_gpu_availability():
    """
    Check if CUDA is available for GPU acceleration with EasyOCR.
    
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    print_step(1, "Checking GPU availability for EasyOCR")
    
    try:
        # Try to import torch and check CUDA availability
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU is available: {gpu_name}")
            print_success(f"CUDA Version: {torch.version.cuda}")
            return True
        else:
            print_warning("CUDA is not available. EasyOCR will run on CPU mode (slower).")
            print_warning("To use GPU acceleration, ensure you have a compatible NVIDIA GPU and CUDA installed.")
            return False
    except ImportError:
        print_warning("PyTorch is not installed yet. GPU availability will be checked after dependencies installation.")
        return False
    except Exception as e:
        print_warning(f"Error checking GPU availability: {e}")
        print_warning("Will continue with CPU mode for EasyOCR.")
        return False

def create_virtual_environment():
    """
    Create a Python virtual environment.
    
    Returns:
        bool: True if virtual environment was created successfully, False otherwise.
    """
    print_step(2, "Creating Python virtual environment")
    venv_dir = "venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print_warning(f"Virtual environment already exists at '{venv_dir}'")
        recreate = input("Do you want to recreate it? (y/n): ").lower().strip() == 'y'
        if recreate:
            try:
                shutil.rmtree(venv_dir)
                print_success("Removed existing virtual environment")
            except Exception as e:
                print_error(f"Failed to remove existing virtual environment: {e}")
                return False
        else:
            print_warning("Using existing virtual environment")
            return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print_success(f"Created virtual environment in '{venv_dir}' directory")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error creating virtual environment: {e}")
        return False

def create_required_directories():
    """Create directories required for the application."""
    print_step(3, "Creating required directories")
    directories = ['flight_recordings', 'results', '.tmp']
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory '{directory}': {e}")

def install_dependencies():
    """
    Install Python dependencies from requirements.txt into the virtual environment.
    
    Returns:
        bool: True if dependencies were installed successfully, False otherwise.
    """
    print_step(4, "Installing dependencies into virtual environment")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print_error("requirements.txt not found")
        return False
    
    # Get the python/pip paths based on the OS
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
        pip_path = os.path.join("venv", "Scripts", "pip.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
        pip_path = os.path.join("venv", "bin", "pip")
    
    # Verify the virtual environment exists and has the necessary executables
    if not os.path.exists(python_path) or not os.path.exists(pip_path):
        print_error(f"Virtual environment executables not found at expected locations")
        print_warning("Make sure the virtual environment was created correctly")
        return False
    
    try:
        # Upgrade pip first using the virtual environment's python
        print_warning("Upgrading pip in virtual environment...")
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install dependencies using the virtual environment's pip
        print_warning("Installing requirements into virtual environment... (this may take a while)")
        subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        # Verify each important dependency individually
        print_warning("Verifying installations...")
        
        # List of core dependencies to verify
        dependencies = [
            ("numpy", "NumPy (array processing)"),
            ("cv2", "OpenCV (image processing)"),
            ("torch", "PyTorch (deep learning)"),
            ("easyocr", "EasyOCR (optical character recognition)")
        ]
        
        all_successful = True
        for module_name, description in dependencies:
            try:
                # Try to import the module
                cmd = f"import {module_name}; print('Success')"
                result = subprocess.run([python_path, "-c", cmd], 
                                      capture_output=True, text=True, check=False)
                
                if "Success" in result.stdout:
                    # Get version if successful
                    version_cmd = (
                        f"import {module_name}; " 
                        f"print(getattr({module_name}, '__version__', 'unknown version'))"
                    )
                    version_result = subprocess.run([python_path, "-c", version_cmd], 
                                                  capture_output=True, text=True, check=False)
                    version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown version"
                    
                    print_success(f"{description} - Installed ({version})")
                else:
                    print_error(f"{description} - Failed to import")
                    all_successful = False
                    print_warning(f"Error: {result.stderr.strip()}")
            except Exception as e:
                print_error(f"{description} - Error during verification: {e}")
                all_successful = False
        
        # Special check for GPU support
        try:
            gpu_cmd = "import torch; print(torch.cuda.is_available())"
            gpu_check = subprocess.run([python_path, "-c", gpu_cmd], 
                                      capture_output=True, text=True, check=True)
            
            if "True" in gpu_check.stdout:
                device_cmd = "import torch; print(torch.cuda.get_device_name(0))"
                device_check = subprocess.run([python_path, "-c", device_cmd], 
                                            capture_output=True, text=True, check=False)
                gpu_name = device_check.stdout.strip() if device_check.returncode == 0 else "unknown device"
                
                print_success(f"GPU Acceleration - Available ({gpu_name})")
            else:
                print_warning("GPU Acceleration - Not available (EasyOCR will run in CPU mode)")
        except Exception as e:
            print_warning(f"GPU Acceleration - Could not verify: {e}")
        
        # Return overall success status
        if all_successful:
            print_success("All core dependencies were installed successfully")
            return True
        else:
            print_error("Some dependencies failed to install correctly")
            print_warning("Try manually installing the missing packages or check for errors")
            return False
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        if hasattr(e, 'output') and e.output:
            print_warning(f"Error details: {e.output}")
        return False
    except Exception as e:
        print_error(f"Unexpected error installing dependencies: {e}")
        import traceback
        print_warning(traceback.format_exc())
        return False

def print_next_steps():
    """Print instructions for the next steps."""
    print("\n" + "="*60)
    print(f"{BOLD}Setup Complete!{RESET}")
    print("="*60)
    
    # Activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"\n{BOLD}Next Steps:{RESET}")
    print(f"1. Activate the virtual environment:")
    print(f"   {YELLOW}{activate_cmd}{RESET}")
    print(f"2. Place your flight recordings in the 'flight_recordings' directory")
    print(f"3. Run the application:")
    print(f"   {YELLOW}python main.py{RESET}")
    print("\n" + "="*60 + "\n")

def main():
    """Main setup function."""
    print("\n" + "="*60)
    print(f"{BOLD}Starship Analyzer Setup{RESET}")
    print("="*60 + "\n")
    
    # Execute setup steps
    gpu_available = check_gpu_availability()
    venv_created = create_virtual_environment()
    create_required_directories()
    
    # Only continue with dependency installation if venv was created successfully
    deps_installed = False
    if venv_created:
        deps_installed = install_dependencies()
    
    # Print summary
    print("\n" + "="*60)
    print(f"{BOLD}Setup Summary:{RESET}")
    print("="*60)
    print(f"GPU Acceleration: {'✓' if gpu_available else '⚠ (CPU mode)'}")
    print(f"Virtual Environment: {'✓' if venv_created else '✗'}")
    print(f"Dependencies: {'✓' if deps_installed else '✗'}")
    
    # Print next steps
    if venv_created and deps_installed:
        print_next_steps()
    else:
        print_error("Setup completed with errors. Please fix the issues and try again.")

if __name__ == "__main__":
    main()