import os
import re
import platform
import subprocess
from typing import Any

from .utilities import print_step, print_success, print_warning, print_error, print_debug

def check_cuda_version(step_num=3, debug=False) -> str | Any | None:
    """
    Check the installed CUDA version on the system.
    
    Args:
        step_num (int): Step number to display in the console output
        debug (bool): Whether to enable debug output
    
    Returns:
        str or None: CUDA version (e.g. '12.6', '12.4', '11.8') or None if not found
    """
    print_step(step_num, "Checking CUDA version for PyTorch installation")
    
    cuda_version = None
    
    try:
        # Try nvidia-smi first (works on both Windows and Linux)
        print_debug("Trying to detect CUDA with nvidia-smi")
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print_debug("nvidia-smi command succeeded")
            if debug:
                print_debug(f"nvidia-smi output: {result.stdout}")
                
            # Extract CUDA version from nvidia-smi output
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                cuda_version = match.group(1)
                print_success(f"CUDA version {cuda_version} detected using nvidia-smi")
                return cuda_version
            else:
                print_debug("No CUDA version pattern found in nvidia-smi output")
        else:
            print_debug(f"nvidia-smi returned non-zero exit code: {result.returncode}")
    except Exception as e:
        print_warning(f"nvidia-smi check failed: {e}")
        print_debug(f"Exception details for nvidia-smi check: {str(e)}")
    
    # Try Windows-specific checks
    if platform.system() == "Windows":
        try:
            print_debug("Checking CUDA on Windows system")
            # Try checking Windows registry
            import winreg
            try:
                print_debug("Checking Windows registry for CUDA information")
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA") as key:
                    cuda_version = winreg.QueryValueEx(key, "Version")[0]
                    print_success(f"CUDA version {cuda_version} detected from registry")
                    return cuda_version
            except Exception as e:
                print_debug(f"Windows registry check failed: {e}")
                
            # Check common install locations
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                os.path.expanduser("~") + r"\NVIDIA GPU Computing Toolkit\CUDA"
            ]
            print_debug(f"Checking common CUDA installation paths: {cuda_paths}")
            
            for base_path in cuda_paths:
                if os.path.exists(base_path):
                    print_debug(f"Found CUDA base path: {base_path}")
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v")]
                    print_debug(f"Found version directories: {versions}")
                    if versions:
                        # Sort versions and take the latest
                        versions.sort(reverse=True)
                        cuda_version = versions[0][1:]  # Remove 'v' prefix
                        print_success(f"CUDA version {cuda_version} detected in {base_path}")
                        return cuda_version
        except Exception as e:
            print_warning(f"Windows CUDA check failed: {e}")
            print_debug(f"Exception during Windows CUDA checks: {str(e)}")
    
    # Try Linux-specific checks
    elif platform.system() == "Linux":
        try:
            print_debug("Checking CUDA on Linux system")
            # Check CUDA_PATH environment variable
            if "CUDA_PATH" in os.environ:
                path = os.environ["CUDA_PATH"]
                print_debug(f"Found CUDA_PATH environment variable: {path}")
                match = re.search(r"/cuda-(\d+\.\d+)", path)
                if match:
                    cuda_version = match.group(1)
                    print_success(f"CUDA version {cuda_version} detected from CUDA_PATH")
                    return cuda_version
            
            # Check common install locations on Linux
            print_debug("Checking common Linux CUDA locations")
            for cuda_path in ["/usr/local/cuda", "/usr/cuda"]:
                if os.path.exists(cuda_path):
                    print_debug(f"Found CUDA path: {cuda_path}")
                    if os.path.islink(cuda_path):
                        target = os.readlink(cuda_path)
                        print_debug(f"Symlink target: {target}")
                        match = re.search(r"cuda-(\d+\.\d+)", target)
                        if match:
                            cuda_version = match.group(1)
                            print_success(f"CUDA version {cuda_version} detected from {cuda_path}")
                            return cuda_version
                    elif os.path.isdir(cuda_path):
                        # Try to find version from cuda binary
                        nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                        print_debug(f"Checking for nvcc at: {nvcc_path}")
                        if os.path.exists(nvcc_path):
                            result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, check=False)
                            print_debug(f"nvcc --version output: {result.stdout}")
                            match = re.search(r"release (\d+\.\d+)", result.stdout)
                            if match:
                                cuda_version = match.group(1)
                                print_success(f"CUDA version {cuda_version} detected from nvcc")
                                return cuda_version
        except Exception as e:
            print_warning(f"Linux CUDA check failed: {e}")
            print_debug(f"Exception during Linux CUDA checks: {str(e)}")
    
    # If we get here, no CUDA version was found
    print_warning("No CUDA installation detected. Will install CPU-only version of PyTorch.")
    return None

def install_nvidia_drivers(step_num=4, debug=False):
    """
    Install the latest NVIDIA drivers if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to enable debug output
    """
    print_step(step_num, "Installing NVIDIA drivers")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing NVIDIA drivers for Windows...")
            url = "https://www.nvidia.com/Download/index.aspx"
            print_debug(f"Directing user to NVIDIA driver download page: {url}")
            print_warning(f"Please visit {url} to download and install the latest NVIDIA drivers.")
        except Exception as e:
            print_error(f"Failed to guide NVIDIA driver installation: {e}")
            print_debug(f"Exception during Windows NVIDIA driver guidance: {str(e)}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing NVIDIA drivers for Linux...")
            # Update package lists first
            cmd = ["sudo", "apt-get", "update"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            # Only show output in real-time if debug is enabled
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
            
            # Install NVIDIA drivers
            cmd = ["sudo", "apt-get", "install", "-y", "nvidia-driver-470"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            # Only show output in real-time if debug is enabled
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
                
            print_success("NVIDIA drivers installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install NVIDIA drivers: {e}")
            print_debug(f"CalledProcessError during NVIDIA driver installation: {e}")
        except Exception as e:
            print_error(f"Unexpected error during NVIDIA driver installation: {e}")
            print_debug(f"Exception during NVIDIA driver installation: {str(e)}")
    else:
        print_warning("NVIDIA driver installation is not supported on this platform.")
        print_debug(f"Unsupported platform for NVIDIA driver installation: {platform.system()}")

def install_cuda_toolkit(step_num=5, debug=False):
    """
    Install the CUDA Toolkit if not already installed.
    
    Args:
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to enable debug output
    """
    print_step(step_num, "Installing CUDA Toolkit")
    if platform.system() == "Windows":
        try:
            print_warning("Downloading and installing CUDA Toolkit for Windows...")
            url = "https://developer.nvidia.com/cuda-downloads"
            print_debug(f"Directing user to CUDA Toolkit download page: {url}")
            print_warning(f"Please visit {url} to download and install the latest CUDA Toolkit.")
        except Exception as e:
            print_error(f"Failed to guide CUDA Toolkit installation: {e}")
            print_debug(f"Exception during Windows CUDA Toolkit guidance: {str(e)}")
    elif platform.system() == "Linux":
        try:
            print_warning("Installing CUDA Toolkit for Linux...")
            # Update package lists first
            cmd = ["sudo", "apt-get", "update"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            # Only show output in real-time if debug is enabled
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
            
            # Install CUDA toolkit
            cmd = ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            # Only show output in real-time if debug is enabled
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
                
            print_success("CUDA Toolkit installed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install CUDA Toolkit: {e}")
            print_debug(f"CalledProcessError during CUDA Toolkit installation: {e}")
        except Exception as e:
            print_error(f"Unexpected error during CUDA Toolkit installation: {e}")
            print_debug(f"Exception during CUDA Toolkit installation: {str(e)}")
    else:
        print_warning("CUDA Toolkit installation is not supported on this platform.")
        print_debug(f"Unsupported platform for CUDA Toolkit installation: {platform.system()}")
