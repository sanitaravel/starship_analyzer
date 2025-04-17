import os
import platform
import subprocess

from .utilities import print_step, print_success, print_info, print_warning, print_error, print_debug

def install_torch_with_cuda(pip_path, cuda_version, debug=False):
    """
    Install PyTorch with the appropriate CUDA support.
    
    Args:
        pip_path (str): Path to the pip executable
        cuda_version (str or None): CUDA version or None for CPU-only
        debug (bool): Whether to show detailed installation output
        
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
    
    print_debug(f"Normalized CUDA version: {cuda_version}", debug)
    
    # Determine installation command
    if cuda_version and cuda_version in cuda_map:
        cuda_tag = cuda_map[cuda_version]
        print_info(f"Installing PyTorch with CUDA {cuda_version} support ({cuda_tag})...")
        url = f"https://download.pytorch.org/whl/{cuda_tag}"
        
        try:
            # Determine capture_output based on debug flag
            if debug:
                # Show real-time output in debug mode
                print_debug("Running with real-time output in debug mode", debug)
                subprocess.run(
                    [pip_path, "install", "torch", "torchvision", "--index-url", url],
                    check=True
                )
            else:
                # Suppress output in normal mode
                print_info("Installing PyTorch (this may take a while)...")
                result = subprocess.run(
                    [pip_path, "install", "torch", "torchvision", "--index-url", url],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
            print_success(f"PyTorch installed with CUDA {cuda_version} support")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA {cuda_version} failed")
            if debug and hasattr(e, 'stderr'):
                print_debug(f"Error details: {e.stderr}", True)
            print_warning("Falling back to CPU-only PyTorch installation")
    elif cuda_version:
        # CUDA is installed but version not in cuda_map - use cu118 as fallback
        print_info(f"CUDA {cuda_version} detected but not in supported versions map")
        print_info("Installing PyTorch with CUDA 11.8 support as fallback...")
        url = "https://download.pytorch.org/whl/cu118"
        
        try:
            if debug:
                print_debug("Running PyTorch fallback installation with real-time output", debug)
                subprocess.run(
                    [pip_path, "install", "torch", "torchvision", "--index-url", url],
                    check=True
                )
            else:
                print_info("Installing PyTorch (this may take a while)...")
                subprocess.run(
                    [pip_path, "install", "torch", "torchvision", "--index-url", url],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
            print_success(f"PyTorch installed with CUDA 11.8 support (fallback for CUDA {cuda_version})")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA 11.8 fallback failed")
            if debug and hasattr(e, 'stderr'):
                print_debug(f"Error details: {e.stderr}", True)
            print_warning("Falling back to CPU-only PyTorch installation")
    else:
        print_warning("No CUDA detected, will use CPU-only version of PyTorch")  # Changed from info to warning
    
    # Fall back to CPU-only installation
    try:
        print_info("Installing CPU-only PyTorch...")
        if debug:
            print_debug("Running CPU-only PyTorch installation with real-time output", debug)
            subprocess.run(
                [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"],
                check=True
            )
        else:
            print_info("Installing PyTorch CPU version (this may take a while)...")
            subprocess.run(
                [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"],
                check=True,
                capture_output=True,
                text=True
            )
            
        print_success("CPU-only PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("CPU-only PyTorch installation failed")
        if debug and hasattr(e, 'stderr'):
            print_debug(f"Error details: {e.stderr}", True)
        return False

def install_dependencies(cuda_version, step_num=6, force_cpu=False, debug=False):
    """
    Install Python dependencies from requirements.txt into the virtual environment.
    
    Args:
        cuda_version (str or None): CUDA version for PyTorch installation
        step_num (int or float): Step number to display in the console output
        force_cpu (bool): Whether to force CPU-only installation
        debug (bool): Whether to show detailed installation output
    
    Returns:
        bool: True if dependencies were installed successfully, False otherwise.
    """
    print_step(step_num, "Installing dependencies into virtual environment")
    
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
    
    # Install python3-tk on Linux systems
    if platform.system() == "Linux":
        print_info("Installing Python Tkinter package for Linux...")
        try:
            # Install Linux packages one by one
            linux_packages = ["python3-tk"]
            for pkg in linux_packages:
                print_info(f"Installing {pkg}...")
                if debug:
                    # Show output in debug mode
                    result = subprocess.run(["sudo", "apt-get", "install", "-y", pkg], check=True)
                else:
                    # Suppress output in normal mode
                    result = subprocess.run(
                        ["sudo", "apt-get", "install", "-y", pkg],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                print_success(f"Successfully installed {pkg}")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install python3-tk")
            if debug:
                print_debug(f"Error details: {e}", True)
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
        except Exception as e:
            print_warning(f"Could not verify or install python3-tk")
            if debug:
                print_debug(f"Error details: {e}", True)
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
    
    # Verify the virtual environment exists and has the necessary executables
    if not os.path.exists(python_path) or not os.path.exists(pip_path):
        print_error(f"Virtual environment executables not found at expected locations")
        print_warning("Make sure the virtual environment was created correctly")
        return False
    
    # Define Windows-only packages
    windows_only_packages = ["pywin32", "WMI", "wmi"]
    
    try:
        # Upgrade pip first using the virtual environment's python
        print_info("Upgrading pip in virtual environment...")
        if debug:
            subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        else:
            subprocess.run(
                [python_path, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True
            )
        print_success("Successfully upgraded pip")
        
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
        
        # Parse individual packages from requirements.txt
        individual_packages = []
        for line in requirements:
            line = line.strip()
            # Skip empty lines, comments, and torch/torchvision (handled separately)
            if (not line or line.startswith('#') or line.startswith('//') or 
                line.startswith("torch") or line.startswith("torchvision")):
                continue
                
            # Skip Windows-only packages on non-Windows platforms
            package_name = line.split('==')[0].strip() if '==' in line else line.strip()
            if platform.system() != "Windows" and package_name.lower() in [p.lower() for p in windows_only_packages]:
                print_info(f"Skipping Windows-only package: {package_name}")
                continue
                
            individual_packages.append(line)
        
        # Install packages one by one
        all_packages_success = True
        print_info(f"Installing {len(individual_packages)} packages...")
        for package in individual_packages:
            try:
                # Show minimal package name for installation message
                package_name = package.split("==")[0].strip() if "==" in package else package.strip()
                print_info(f"Installing {package_name}...")
                
                if debug:
                    # Show output in debug mode
                    subprocess.run([pip_path, "install", package], check=True)
                else:
                    # Suppress output in normal mode
                    subprocess.run(
                        [pip_path, "install", package],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                print_success(f"Successfully installed {package_name}")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to install {package_name}")
                if debug and hasattr(e, 'stderr'):
                    print_debug(f"Error details: {e.stderr}", True)
                all_packages_success = False
        
        # Install PyTorch with appropriate CUDA support
        if force_cpu:
            print_warning("Forcing CPU-only PyTorch installation")
            cuda_version = None
            
        torch_installed = install_torch_with_cuda(pip_path, cuda_version, debug=debug)
        
        if not torch_installed:
            print_error("Failed to install PyTorch")
            return False
        
        # Installation completed successfully
        if all_packages_success:
            print_success("All dependencies installed successfully")
            return True
        else:
            print_error("Some dependencies failed to install")
            return False
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies")
        if debug:
            print_debug(f"Error details: {e}", True)
        return False
    except Exception as e:
        print_error(f"Unexpected error installing dependencies")
        if debug:
            import traceback
            print_debug(traceback.format_exc(), True)
        return False
