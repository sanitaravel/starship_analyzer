import os
import platform
import subprocess

from .utilities import print_step, print_success, print_warning, print_error

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
            # Install PyTorch with specific CUDA support - allow output to stream to console
            subprocess.run(
                [pip_path, "install", "torch", "torchvision", "--index-url", url],
                check=True
            )
            print_success(f"PyTorch installed with CUDA {cuda_version} support")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA {cuda_version} failed")
            print_warning("Falling back to CPU-only PyTorch installation")
    elif cuda_version:
        # CUDA is installed but version not in cuda_map - use cu118 as fallback
        print_warning(f"CUDA {cuda_version} detected but not in supported versions map")
        print_warning("Installing PyTorch with CUDA 11.8 support as fallback...")
        url = "https://download.pytorch.org/whl/cu118"
        
        try:
            # Install PyTorch with CUDA 11.8 support - allow output to stream to console
            subprocess.run(
                [pip_path, "install", "torch", "torchvision", "--index-url", url],
                check=True
            )
            print_success(f"PyTorch installed with CUDA 11.8 support (fallback for CUDA {cuda_version})")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"PyTorch installation with CUDA 11.8 fallback failed")
            print_warning("Falling back to CPU-only PyTorch installation")
    else:
        print_warning("No CUDA detected")
    
    # Fall back to CPU-only installation
    try:
        print_warning("Installing CPU-only PyTorch...")
        subprocess.run(
            [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True
        )
        print_success("CPU-only PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("CPU-only PyTorch installation failed")
        return False

def install_dependencies(cuda_version, step_num=6, force_cpu=False):
    """
    Install Python dependencies from requirements.txt into the virtual environment.
    
    Args:
        cuda_version (str or None): CUDA version for PyTorch installation
        step_num (int or float): Step number to display in the console output
        force_cpu (bool): Whether to force CPU-only installation
    
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
        print_warning("Installing Python Tkinter package for Linux...")
        try:
            # Show real-time output by not capturing it
            subprocess.run(["sudo", "apt-get", "install", "-y", "python3-tk"], check=True)
            print_success("Successfully installed python3-tk")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install python3-tk: {e}")
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
        except Exception as e:
            print_warning(f"Could not verify or install python3-tk: {e}")
            print_warning("You may need to manually install Tkinter: sudo apt-get install python3-tk")
    
    # Verify the virtual environment exists and has the necessary executables
    if not os.path.exists(python_path) or not os.path.exists(pip_path):
        print_error(f"Virtual environment executables not found at expected locations")
        print_warning("Make sure the virtual environment was created correctly")
        return False
    
    # Define Windows-only packages
    windows_only_packages = ["pywin32", "WMI", "wmi"]
    
    try:
        # Upgrade pip first using the virtual environment's python - show real-time output
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
        
        # Remove the package display section and keep only the requirements file preparation
        
        # Create a modified requirements file without torch, torchvision, and Windows-only packages on non-Windows
        temp_req_path = os.path.join(".tmp", "requirements_modified.txt")
        os.makedirs(os.path.dirname(temp_req_path), exist_ok=True)
        
        with open(temp_req_path, 'w', encoding='utf-8') as f:
            for line in requirements:
                # Skip comment lines that might be causing problems
                if line.strip().startswith("//"):
                    continue
                
                # Skip torch and torchvision as they will be installed separately
                if line.startswith("torch") or line.startswith("torchvision"):
                    continue
                    
                # Skip Windows-only packages on non-Windows platforms
                if platform.system() != "Windows":
                    package_name = line.split('==')[0].strip() if '==' in line else line.strip()
                    if package_name.lower() in [p.lower() for p in windows_only_packages]:
                        print_warning(f"Skipping Windows-only package: {package_name}")
                        continue
                
                f.write(line)
        
        # Install other dependencies first - allow output to stream to console
        print_warning("Installing dependencies from requirements.txt...")
        try:
            # Run without capture_output to show real-time installation progress
            subprocess.run([pip_path, "install", "-r", temp_req_path], check=True)
            print_success("Successfully installed dependencies from requirements.txt")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies from requirements file")
            
            # Try to install packages one by one to identify problematic package
            print_warning("Attempting to install packages individually to identify problematic packages...")
            
            with open(temp_req_path, 'r', encoding='utf-8') as f:
                individual_packages = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            for package in individual_packages:
                try:
                    # Show real-time output for individual package installation
                    print_warning(f"Installing {package}...")
                    subprocess.run([pip_path, "install", package], check=True)
                    print_success(f"Successfully installed {package}")
                except subprocess.CalledProcessError:
                    print_error(f"Failed to install {package}")
            
            return False
        
        # Install PyTorch with appropriate CUDA support
        if force_cpu:
            print_warning("Forcing CPU-only PyTorch installation")
            cuda_version = None
            
        torch_installed = install_torch_with_cuda(pip_path, cuda_version)
        
        if not torch_installed:
            print_error("Failed to install PyTorch")
            return False
        
        # Installation completed successfully
        print_success("All dependencies installed successfully")
        return True
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error installing dependencies: {e}")
        import traceback
        print_warning(traceback.format_exc())
        return False
