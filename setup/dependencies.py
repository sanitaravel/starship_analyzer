import os
import platform
import subprocess

from .utilities import print_step, print_success, print_warning, print_error, print_debug

def install_torch_with_cuda(pip_path, cuda_version, debug=False):
    """
    Install PyTorch with the appropriate CUDA support.
    
    Args:
        pip_path (str): Path to the pip executable
        cuda_version (str or None): CUDA version or None for CPU-only
        debug (bool): Whether to enable debug output
        
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
    
    print_debug(f"Available CUDA mappings: {cuda_map}")
    
    # Normalize CUDA version - take only major.minor
    if cuda_version:
        cuda_version = ".".join(cuda_version.split(".")[:2])
        print_debug(f"Normalized CUDA version: {cuda_version}")
    
    # Determine installation command
    if cuda_version and cuda_version in cuda_map:
        cuda_tag = cuda_map[cuda_version]
        print_warning(f"Installing PyTorch with CUDA {cuda_version} support ({cuda_tag})...")
        url = f"https://download.pytorch.org/whl/{cuda_tag}"
        print_debug(f"Using PyTorch URL: {url}")
        
        try:
            # Install PyTorch with specific CUDA support - allow output to stream to console
            cmd = [pip_path, "install", "torch", "torchvision", "--index-url", url]
            print_debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print_success(f"PyTorch installed with CUDA {cuda_version} support")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA {cuda_version} failed")
            print_debug(f"CalledProcessError during PyTorch installation: {e}")
            print_debug(f"Return code: {e.returncode}")
            print_warning("Falling back to CPU-only PyTorch installation")
    elif cuda_version:
        # CUDA is installed but version not in cuda_map - use cu118 as fallback
        print_warning(f"CUDA {cuda_version} detected but not in supported versions map")
        print_warning("Installing PyTorch with CUDA 11.8 support as fallback...")
        url = "https://download.pytorch.org/whl/cu118"
        print_debug(f"Using fallback PyTorch URL: {url}")
        
        try:
            # Install PyTorch with CUDA 11.8 support - allow output to stream to console
            cmd = [pip_path, "install", "torch", "torchvision", "--index-url", url]
            print_debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print_success(f"PyTorch installed with CUDA 11.8 support (fallback for CUDA {cuda_version})")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA 11.8 fallback failed")
            print_debug(f"CalledProcessError during PyTorch fallback installation: {e}")
            print_warning("Falling back to CPU-only PyTorch installation")
    else:
        print_warning("No CUDA detected")
    
    # Fall back to CPU-only installation
    try:
        print_warning("Installing CPU-only PyTorch...")
        cmd = [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
        print_debug(f"Running CPU-only command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print_success("CPU-only PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("CPU-only PyTorch installation failed")
        print_debug(f"CalledProcessError during CPU-only PyTorch installation: {e}")
        return False

def install_dependencies(cuda_version, step_num=6, force_cpu=False, debug=False):
    """
    Install Python dependencies from requirements.txt into the virtual environment.
    
    Args:
        cuda_version (str or None): CUDA version for PyTorch installation
        step_num (int or float): Step number to display in the console output
        force_cpu (bool): Whether to force CPU-only installation
        debug (bool): Whether to enable debug output
    
    Returns:
        bool: True if dependencies were installed successfully, False otherwise.
    """
    print_step(step_num, "Installing dependencies into virtual environment")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print_error("requirements.txt not found")
        print_debug(f"Looking for requirements.txt in: {os.path.abspath('requirements.txt')}")
        return False
    
    # Get the python/pip paths based on the OS
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
        pip_path = os.path.join("venv", "Scripts", "pip.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
        pip_path = os.path.join("venv", "bin", "pip")
    
    print_debug(f"Using Python path: {os.path.abspath(python_path)}")
    print_debug(f"Using pip path: {os.path.abspath(pip_path)}")
    
    # Install python3-tk on Linux systems
    if platform.system() == "Linux":
        print_warning("Installing Python Tkinter package for Linux...")
        try:
            # Show real-time output by not capturing it
            cmd = ["sudo", "apt-get", "install", "-y", "python3-tk"]
            print_debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print_success("Successfully installed python3-tk")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install python3-tk: {e}")
            print_debug(f"CalledProcessError during Tkinter installation: {e}")
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
        except Exception as e:
            print_warning(f"Could not verify or install python3-tk: {e}")
            print_debug(f"Exception during Tkinter installation: {e}")
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
    
    # Verify the virtual environment exists and has the necessary executables
    if not os.path.exists(python_path) or not os.path.exists(pip_path):
        print_error(f"Virtual environment executables not found at expected locations")
        print_debug(f"Python exists: {os.path.exists(python_path)}")
        print_debug(f"Pip exists: {os.path.exists(pip_path)}")
        print_warning("Make sure the virtual environment was created correctly")
        return False
    
    # Define Windows-only packages
    windows_only_packages = ["pywin32", "WMI", "wmi"]
    print_debug(f"Windows-only packages that will be skipped on non-Windows: {windows_only_packages}")
    
    try:
        # Upgrade pip first using the virtual environment's python - show real-time output
        print_warning("Upgrading pip in virtual environment...")
        cmd = [python_path, "-m", "pip", "install", "--upgrade", "pip"]
        print_debug(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Try to read requirements.txt with different encodings
        requirements = []
        encodings_to_try = ['utf-8', 'utf-16', 'utf-8-sig', 'latin-1', 'cp1252']
        success = False
        
        print_debug(f"Trying to read requirements.txt with encodings: {encodings_to_try}")
        
        for encoding in encodings_to_try:
            try:
                with open("requirements.txt", 'r', encoding=encoding) as f:
                    requirements = f.readlines()
                print_success(f"Successfully read requirements.txt using {encoding} encoding")
                print_debug(f"Read {len(requirements)} lines from requirements.txt")
                success = True
                break
            except UnicodeDecodeError:
                print_debug(f"Failed to read with encoding {encoding}: UnicodeDecodeError")
                continue
            except Exception as e:
                print_error(f"Error reading requirements.txt: {e}")
                print_debug(f"Exception during requirements.txt reading: {e}")
                return False
        
        if not success:
            print_error("Failed to read requirements.txt with any known encoding")
            print_warning("Please ensure the file is properly encoded (preferably as UTF-8)")
            return False
        
        # Create a modified requirements file without torch, torchvision, and OpenCV packages
        temp_req_path = os.path.join(".tmp", "requirements_modified.txt")
        os.makedirs(os.path.dirname(temp_req_path), exist_ok=True)
        print_debug(f"Creating modified requirements file at: {os.path.abspath(temp_req_path)}")
        
        with open(temp_req_path, 'w', encoding='utf-8') as f:
            for line in requirements:
                # Skip comment lines that might be causing problems
                if line.strip().startswith("//"):
                    print_debug(f"Skipping comment line: {line.strip()}")
                    continue
                
                # Skip torch, torchvision and OpenCV as they will be installed separately
                if (line.startswith("torch") or 
                    line.startswith("torchvision") or 
                    line.startswith("opencv-python") or 
                    line.startswith("opencv-python-headless")):
                    print_debug(f"Skipping line for separate installation: {line.strip()}")
                    continue
                    
                # Skip Windows-only packages on non-Windows platforms
                if platform.system() != "Windows":
                    package_name = line.split('==')[0].strip() if '==' in line else line.strip()
                    if package_name.lower() in [p.lower() for p in windows_only_packages]:
                        print_warning(f"Skipping Windows-only package: {package_name}")
                        print_debug(f"Skipping Windows-only package line: {line.strip()}")
                        continue
                
                f.write(line)
                print_debug(f"Added to modified requirements: {line.strip()}")
        
        # Install other dependencies first - allow output to stream to console
        print_warning("Installing dependencies from requirements.txt...")
        try:
            # Run without capture_output to show real-time installation progress
            cmd = [pip_path, "install", "-r", temp_req_path]
            print_debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print_success("Successfully installed dependencies from requirements.txt")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies from requirements file")
            print_debug(f"CalledProcessError during requirements installation: {e}")
            
            # Try to install packages one by one to identify problematic package
            print_warning("Attempting to install packages individually to identify problematic packages...")
            
            with open(temp_req_path, 'r', encoding='utf-8') as f:
                individual_packages = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            print_debug(f"Individual packages to install: {individual_packages}")
            
            for package in individual_packages:
                try:
                    # Show real-time output for individual package installation
                    print_warning(f"Installing {package}...")
                    cmd = [pip_path, "install", package]
                    print_debug(f"Running command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    print_success(f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    print_error(f"Failed to install {package}")
                    print_debug(f"CalledProcessError installing {package}: {e}")
            
            return False
        
        # Install PyTorch with appropriate CUDA support
        if force_cpu:
            print_warning("Forcing CPU-only PyTorch installation")
            print_debug("force_cpu=True, setting cuda_version=None")
            cuda_version = None
            
        torch_installed = install_torch_with_cuda(pip_path, cuda_version, debug)
        
        if not torch_installed:
            print_error("Failed to install PyTorch")
            return False
        
        # Installation completed successfully
        print_success("All dependencies installed successfully")
        return True
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print_debug(f"CalledProcessError during dependency installation: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error installing dependencies: {e}")
        print_debug(f"Unexpected exception during dependency installation: {e}")
        import traceback
        tb = traceback.format_exc()
        print_warning(tb)
        print_debug(f"Full traceback: {tb}")
        return False
