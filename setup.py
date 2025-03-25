import os
import subprocess
import sys
import platform
import shutil
import re
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

def create_virtual_environment():
    """
    Create a Python virtual environment.
    
    Returns:
        bool: True if virtual environment was created successfully, False otherwise.
    """
    print_step(1, "Creating Python virtual environment")
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
    print_step(2, "Creating required directories")
    directories = ['flight_recordings', 'results', '.tmp', 'logs']
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory '{directory}': {e}")

def check_cuda_version():
    """
    Check the installed CUDA version on the system.
    
    Returns:
        str or None: CUDA version (e.g. '12.6', '12.4', '11.8') or None if not found
    """
    print_step(3, "Checking CUDA version for PyTorch installation")
    
    cuda_version = None
    
    try:
        # Try nvidia-smi first (works on both Windows and Linux)
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                print_success(f"CUDA version {cuda_version} detected using nvidia-smi")
                return cuda_version
    except Exception as e:
        print_warning(f"nvidia-smi check failed: {e}")
    
    # Try Windows-specific checks
    if platform.system() == "Windows":
        try:
            # Try checking Windows registry
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA") as key:
                    cuda_version = winreg.QueryValueEx(key, "Version")[0]
                    print_success(f"CUDA version {cuda_version} detected from registry")
                    return cuda_version
            except Exception:
                pass
                
            # Check common install locations
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                os.path.expanduser("~") + r"\NVIDIA GPU Computing Toolkit\CUDA"
            ]
            
            for base_path in cuda_paths:
                if os.path.exists(base_path):
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v")]
                    if versions:
                        # Sort versions and take the latest
                        versions.sort(reverse=True)
                        cuda_version = versions[0][1:]  # Remove 'v' prefix
                        print_success(f"CUDA version {cuda_version} detected in {base_path}")
                        return cuda_version
        except Exception as e:
            print_warning(f"Windows registry check failed: {e}")
    
    # Try Linux-specific checks
    elif platform.system() == "Linux":
        try:
            # Check CUDA_PATH environment variable
            if "CUDA_PATH" in os.environ:
                path = os.environ["CUDA_PATH"]
                match = re.search(r"/cuda-(\d+\.\d+)", path)
                if match:
                    cuda_version = match.group(1)
                    print_success(f"CUDA version {cuda_version} detected from CUDA_PATH")
                    return cuda_version
            
            # Check common install locations on Linux
            for cuda_path in ["/usr/local/cuda", "/usr/cuda"]:
                if os.path.islink(cuda_path):
                    target = os.readlink(cuda_path)
                    match = re.search(r"cuda-(\d+\.\d+)", target)
                    if match:
                        cuda_version = match.group(1)
                        print_success(f"CUDA version {cuda_version} detected from {cuda_path}")
                        return cuda_version
                elif os.path.isdir(cuda_path):
                    # Try to find version from cuda binary
                    nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                    if os.path.exists(nvcc_path):
                        result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, check=False)
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print_success(f"CUDA version {cuda_version} detected from nvcc")
                            return cuda_version
        except Exception as e:
            print_warning(f"Linux CUDA check failed: {e}")
    
    # If we get here, no CUDA version was found
    print_warning("No CUDA installation detected. Will install CPU-only version of PyTorch.")
    return None

def install_torch_with_cuda(pip_path, cuda_version):
    """
    Install PyTorch with the appropriate CUDA support.
    
    Args:
        pip_path (str): Path to the pip executable
        cuda_version (str or None): CUDA version or None for CPU-only
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    # Map CUDA versions to PyTorch installation commands
    cuda_map = {
        "12.6": "cu126",
        "12.4": "cu124",
        "11.8": "cu118",
        "11.7": "cu118",  # Fall back to 11.8 for 11.7
        "11.6": "cu118",  # Fall back to 11.8 for 11.6
        "11.5": "cu118",  # Fall back to 11.8 for 11.5
    }
    
    # Normalize CUDA version - take only major.minor
    if cuda_version:
        cuda_version = ".".join(cuda_version.split(".")[:2])
    
    # Determine installation command
    if cuda_version and cuda_version in cuda_map:
        cuda_tag = cuda_map[cuda_version]
        print_warning(f"Installing PyTorch with CUDA {cuda_version} support ({cuda_tag})...")
        url = f"https://download.pytorch.org/whl/{cuda_tag}"
        
        try:
            # Install PyTorch with specific CUDA support
            result = subprocess.run(
                [pip_path, "install", "torch", "torchvision", "--index-url", url],
                check=True, capture_output=True, text=True
            )
            print_success(f"PyTorch installed with CUDA {cuda_version} support")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA {cuda_version} failed:")
            print_warning(e.stdout)
            print_error(e.stderr)
            print_warning("Falling back to CPU-only PyTorch installation")
    else:
        print_warning("No compatible CUDA version found or CUDA not detected")
    
    # Fall back to CPU-only installation
    try:
        print_warning("Installing CPU-only PyTorch...")
        result = subprocess.run(
            [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True, capture_output=True, text=True
        )
        print_success("CPU-only PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("CPU-only PyTorch installation failed:")
        print_warning(e.stdout)
        print_error(e.stderr)
        return False

def install_dependencies(cuda_version):
    """
    Install Python dependencies from requirements.txt into the virtual environment.
    
    Args:
        cuda_version (str or None): CUDA version for PyTorch installation
    
    Returns:
        bool: True if dependencies were installed successfully, False otherwise.
    """
    print_step(6, "Installing dependencies into virtual environment")
    
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
        
        # Try to read requirements.txt with different encodings
        requirements = []
        encodings_to_try = ['utf-8', 'utf-16', 'utf-8-sig', 'latin-1', 'cp1252']
        success = False
        
        for encoding in encodings_to_try:
            try:
                with open("requirements.txt", 'r', encoding=encoding) as f:
                    requirements = f.readlines()
                print_success(f"Successfully read requirements.txt using {encoding} encoding")
                success = True
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print_error(f"Error reading requirements.txt: {e}")
                return False
        
        if not success:
            print_error("Failed to read requirements.txt with any known encoding")
            print_warning("Please ensure the file is properly encoded (preferably as UTF-8)")
            return False
        
        # Create a modified requirements file without torch and torchvision
        temp_req_path = os.path.join(".tmp", "requirements_without_torch.txt")
        os.makedirs(os.path.dirname(temp_req_path), exist_ok=True)
        
        with open(temp_req_path, 'w', encoding='utf-8') as f:
            for line in requirements:
                # Skip comment lines that might be causing problems
                if line.strip().startswith("//"):
                    continue
                if not (line.startswith("torch") or line.startswith("torchvision")):
                    f.write(line)
        
        # Install other dependencies first
        print_warning("Installing other dependencies from requirements.txt...")
        other_deps = subprocess.run([pip_path, "install", "-r", temp_req_path], 
                                   capture_output=True, text=True, check=True)
        
        # Install PyTorch with appropriate CUDA support
        torch_installed = install_torch_with_cuda(pip_path, cuda_version)
        
        if not torch_installed:
            print_error("Failed to install PyTorch")
            return False
        
        # Installation completed successfully
        print_success("All dependencies installed successfully")
        return True
            
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

def verify_installations(python_path):
    """
    Verify that all necessary dependencies are installed correctly.
    
    Args:
        python_path (str): Path to the Python executable
        
    Returns:
        tuple: (bool for success, bool for GPU available)
    """
    print_step(7, "Verifying installations")
    
    # List of core dependencies to verify
    dependencies = [
        ("numpy", "NumPy (array processing)"),
        ("cv2", "OpenCV (image processing)"),
        ("torch", "PyTorch (deep learning)"),
        ("easyocr", "EasyOCR (optical character recognition)")
    ]
    
    all_successful = True
    gpu_available = False
    
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
            gpu_available = True
            
            # Check which CUDA version PyTorch is using
            cuda_ver_cmd = "import torch; print(torch.version.cuda)"
            cuda_ver_check = subprocess.run([python_path, "-c", cuda_ver_cmd], 
                                         capture_output=True, text=True, check=False)
            if cuda_ver_check.returncode == 0:
                cuda_ver = cuda_ver_check.stdout.strip()
                print_success(f"PyTorch is using CUDA version: {cuda_ver}")
        else:
            print_warning("GPU Acceleration - Not available (EasyOCR will run in CPU mode)")
    except Exception as e:
        print_warning(f"GPU Acceleration - Could not verify: {e}")
    
    # Return overall success status
    if all_successful:
        print_success("All core dependencies were installed successfully")
    else:
        print_error("Some dependencies failed to install correctly")
        print_warning("Try manually installing the missing packages or check for errors")
    
    return all_successful, gpu_available

def install_cuda_toolkit():
    """
    Install the CUDA Toolkit if not already installed.
    """
    print_step(4, "Installing CUDA Toolkit")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing CUDA Toolkit for Windows...")
            url = "https://developer.nvidia.com/cuda-downloads"
            print_warning(f"Please visit {url} to download and install the latest CUDA Toolkit.")
        except Exception as e:
            print_error(f"Failed to guide CUDA Toolkit installation: {e}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing CUDA Toolkit for Linux...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"], check=True)
            print_success("CUDA Toolkit installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install CUDA Toolkit: {e}")
        except Exception as e:
            print_error(f"Unexpected error during CUDA Toolkit installation: {e}")
    else:
        print_warning("CUDA Toolkit installation is not supported on this platform.")

def install_nvidia_drivers():
    """
    Install the latest NVIDIA drivers if not already installed.
    """
    print_step(5, "Installing NVIDIA drivers")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing NVIDIA drivers for Windows...")
            url = "https://www.nvidia.com/Download/index.aspx"
            print_warning(f"Please visit {url} to download and install the latest NVIDIA drivers.")
        except Exception as e:
            print_error(f"Failed to guide NVIDIA driver installation: {e}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing NVIDIA drivers for Linux...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-driver-470"], check=True)
            print_success("NVIDIA drivers installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install NVIDIA drivers: {e}")
        except Exception as e:
            print_error(f"Unexpected error during NVIDIA driver installation: {e}")
    else:
        print_warning("NVIDIA driver installation is not supported on this platform.")

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
    
    # Step 1: Create virtual environment
    venv_created = create_virtual_environment()
    if not venv_created:
        print_error("Failed to create virtual environment. Aborting setup.")
        return
    
    # Step 2: Create required directories
    create_required_directories()
    
    # Step 3: Check CUDA version before installing dependencies
    cuda_version = check_cuda_version()
    
    # Steps 4-5: Install CUDA Toolkit and NVIDIA drivers if necessary
    if not cuda_version:
        install_cuda_toolkit()
        install_nvidia_drivers()
        cuda_version = check_cuda_version()  # Recheck CUDA version after installation
    
    # Step 6: Install dependencies with the detected CUDA version
    deps_installed = False
    if venv_created:
        deps_installed = install_dependencies(cuda_version)
        if not deps_installed:
            print_error("Failed to install dependencies. Aborting setup.")
            return
    
    # Get the python path for verification
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
    
    # Step 7: Verify installations as a separate step
    all_successful = False
    gpu_available = False
    if deps_installed:
        all_successful, gpu_available = verify_installations(python_path)
    
    # Print summary
    print("\n" + "="*60)
    print(f"{BOLD}Setup Summary:{RESET}")
    print("="*60)
    print(f"Virtual Environment: {'✓' if venv_created else '✗'}")
    print(f"Dependencies: {'✓' if deps_installed else '✗'}")
    print(f"Verification: {'✓' if all_successful else '✗'}")
    print(f"GPU Acceleration: {'✓' if gpu_available else '⚠ (CPU mode)'}")
    
    # Print next steps
    if venv_created and deps_installed and all_successful:
        print_next_steps()
    else:
        print_error("Setup completed with errors. Please fix the issues and try again.")

if __name__ == "__main__":
    main()