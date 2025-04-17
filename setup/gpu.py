import os
import re
import platform
import subprocess
from typing import Any

from .utilities import print_step, print_success, print_warning, print_error, print_info, print_debug

def check_cuda_version(step_num=3, debug=False) -> str | Any | None:
    """
    Check the installed CUDA version on the system.
    
    Args:
        step_num (int): Step number to display in the console output
        debug (bool): Whether to show debug output
    
    Returns:
        str or None: CUDA version (e.g. '12.6', '12.4', '11.8') or None if not found
    """
    print_step(step_num, "Checking CUDA version for PyTorch installation")
    
    cuda_version = None
    
    try:
        # Try nvidia-smi first (works on both Windows and Linux)
        print_debug("Attempting to check CUDA version using nvidia-smi", debug)
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                print_success(f"CUDA version {cuda_version} detected using nvidia-smi")
                print_debug(f"Full nvidia-smi output: {result.stdout}", debug)
                return cuda_version
    except Exception as e:
        print_warning(f"No nvidia-smi found, will try other methods")  # Changed from info to warning
        print_debug(f"nvidia-smi check error details: {e}", debug)
    
    # Try Windows-specific checks
    if platform.system() == "Windows":
        print_debug("Attempting Windows-specific CUDA detection methods", debug)
        try:
            # Try checking Windows registry
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA") as key:
                    cuda_version = winreg.QueryValueEx(key, "Version")[0]
                    print_success(f"CUDA version {cuda_version} detected from registry")
                    return cuda_version
            except Exception as reg_error:
                print_debug(f"Registry check failed: {reg_error}", debug)
                
            # Check common install locations
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                os.path.expanduser("~") + r"\NVIDIA GPU Computing Toolkit\CUDA"
            ]
            
            for base_path in cuda_paths:
                print_debug(f"Checking CUDA path: {base_path}", debug)
                if os.path.exists(base_path):
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v")]
                    if versions:
                        # Sort versions and take the latest
                        versions.sort(reverse=True)
                        cuda_version = versions[0][1:]  # Remove 'v' prefix
                        print_success(f"CUDA version {cuda_version} detected in {base_path}")
                        return cuda_version
        except Exception as e:
            print_info("Windows CUDA detection methods unsuccessful")
            print_debug(f"Windows CUDA check error details: {e}", debug)
    
    # Try Linux-specific checks
    elif platform.system() == "Linux":
        print_debug("Attempting Linux-specific CUDA detection methods", debug)
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
                print_debug(f"Checking Linux CUDA path: {cuda_path}", debug)
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
            print_info("Linux CUDA detection methods unsuccessful")
            print_debug(f"Linux CUDA check error details: {e}", debug)
    
    # If we get here, no CUDA version was found
    print_warning("No CUDA installation detected. Will install CPU-only version of PyTorch.")  # Changed from info to warning
    return None

def install_nvidia_drivers(step_num=4, debug=False):
    """
    Install the latest NVIDIA drivers if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to show detailed installation output
    """
    print_step(step_num, "Installing NVIDIA drivers")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing NVIDIA drivers for Windows...")
            url = "https://www.nvidia.com/Download/index.aspx"
            print_warning(f"Please visit {url} to download and install the latest NVIDIA drivers.")
        except Exception as e:
            print_error(f"Failed to guide NVIDIA driver installation")
            if debug:
                print_error(f"Error details: {e}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing NVIDIA drivers for Linux...")
            
            # Update package list
            print_warning("Updating package lists...")
            if debug:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
            else:
                subprocess.run(["sudo", "apt-get", "update"], check=True, 
                              capture_output=True, text=True)
            
            # Install the driver
            print_warning("Installing nvidia-driver-470...")
            if debug:
                subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-driver-470"], check=True)
            else:
                subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-driver-470"], 
                              check=True, capture_output=True, text=True)
                              
            print_success("NVIDIA drivers installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install NVIDIA drivers")
            if debug:
                print_error(f"Error details: {e}")
        except Exception as e:
            print_error(f"Unexpected error during NVIDIA driver installation")
            if debug:
                print_error(f"Error details: {e}")
    else:
        print_warning("NVIDIA driver installation is not supported on this platform.")

def install_cuda_toolkit(step_num=5, debug=False):
    """
    Install the CUDA Toolkit if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to show detailed installation output
    """
    print_step(step_num, "Installing CUDA Toolkit")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing CUDA Toolkit for Windows...")
            url = "https://developer.nvidia.com/cuda-downloads"
            print_warning(f"Please visit {url} to download and install the latest CUDA Toolkit.")
        except Exception as e:
            print_error(f"Failed to guide CUDA Toolkit installation")
            if debug:
                print_error(f"Error details: {e}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing CUDA Toolkit for Linux...")
            # Update repository information
            print_warning("Updating package lists...")
            if debug:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
            else:
                subprocess.run(["sudo", "apt-get", "update"], 
                              check=True, capture_output=True, text=True)
            print_success("Package lists updated")
            
            # Install CUDA toolkit packages individually
            cuda_packages = ["nvidia-cuda-toolkit"]
            for pkg in cuda_packages:
                print_warning(f"Installing {pkg}...")
                if debug:
                    result = subprocess.run(["sudo", "apt-get", "install", "-y", pkg], check=True)
                else:
                    result = subprocess.run(["sudo", "apt-get", "install", "-y", pkg], 
                                          check=True, capture_output=True, text=True)
                print_success(f"Successfully installed {pkg}")
                    
            print_success("CUDA Toolkit installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install CUDA Toolkit")
            if debug:
                print_error(f"Error details: {e}")
        except Exception as e:
            print_error(f"Unexpected error during CUDA Toolkit installation")
            if debug:
                print_error(f"Error details: {e}")
    else:
        print_warning("CUDA Toolkit installation is not supported on this platform.")
