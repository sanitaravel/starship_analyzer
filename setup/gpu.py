import os
import re
import platform
import subprocess
from typing import Any

from .utilities import print_step, print_success, print_warning, print_error

def check_cuda_version(step_num=3) -> str | Any | None:
    """
    Check the installed CUDA version on the system.
    
    Args:
        step_num (int): Step number to display in the console output
    
    Returns:
        str or None: CUDA version (e.g. '12.6', '12.4', '11.8') or None if not found
    """
    print_step(step_num, "Checking CUDA version for PyTorch installation")
    
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

def install_nvidia_drivers(step_num=4):
    """
    Install the latest NVIDIA drivers if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
    """
    print_step(step_num, "Installing NVIDIA drivers")
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

def install_cuda_toolkit(step_num=5):
    """
    Install the CUDA Toolkit if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
    """
    print_step(step_num, "Installing CUDA Toolkit")
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
