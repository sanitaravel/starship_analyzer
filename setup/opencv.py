import os
import platform
import subprocess
import sys
from pathlib import Path
import shutil
import re

from .utilities import print_step, print_success, print_warning, print_error, print_debug
from .gpu import check_cuda_version

def install_opencv_dependencies(pip_path, python_path, debug=False):
    """
    Install dependencies required for OpenCV compilation.
    
    Args:
        pip_path (str): Path to pip executable
        python_path (str): Path to python executable
        debug (bool): Whether to enable debug output
        
    Returns:
        bool: True if successful, False otherwise
    """
    print_warning("Installing OpenCV build dependencies...")
    
    try:
        # Install numpy first as it's required for OpenCV
        cmd = [pip_path, "install", "numpy"]
        print_debug(f"Running command: {' '.join(cmd)}")
        
        if debug:
            subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print_debug(f"Command output: {result.stdout}")
            print_debug(f"Command error: {result.stderr}")
        
        # Install build dependencies
        cmd = [pip_path, "install", "setuptools", "wheel", "cmake", "scikit-build"]
        print_debug(f"Running command: {' '.join(cmd)}")
        
        if debug:
            subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print_debug(f"Command output: {result.stdout}")
            print_debug(f"Command error: {result.stderr}")
        
        if platform.system() == "Windows":
            # Windows-specific dependencies
            cmd = [pip_path, "install", "ninja"]
            print_debug(f"Running Windows-specific command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        elif platform.system() == "Linux":
            # Linux-specific dependencies - need to use system package manager
            try:
                # Check if sudo is available
                sudo_check_cmd = ["which", "sudo"]
                print_debug(f"Checking if sudo is available: {' '.join(sudo_check_cmd)}")
                sudo_result = subprocess.run(sudo_check_cmd, capture_output=True, text=True, check=False)
                has_sudo = sudo_result.returncode == 0
                
                if not has_sudo:
                    print_warning("sudo is not available. System dependencies must be installed manually.")
                    print_warning("Please install the following packages with your system package manager:")
                    print_warning("build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev")
                    print_warning("libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev")
                    print_warning("libdc1394-22-dev python3-dev python3-numpy")
                    return True  # Continue without system dependencies
                
                # Update package lists first
                update_cmd = ["sudo", "apt-get", "update"]
                print_debug(f"Running command: {' '.join(update_cmd)}")
                
                try:
                    if debug:
                        subprocess.run(update_cmd, check=True)
                    else:
                        result = subprocess.run(update_cmd, check=True, capture_output=True, text=True)
                        print_debug(f"Command output: {result.stdout}")
                        print_debug(f"Command error: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    print_warning(f"Failed to update package lists: {e}")
                    print_debug(f"Return code: {e.returncode}")
                    print_warning("Continuing with installation anyway...")
                
                # Install required packages in groups to better identify issues
                dependency_groups = [
                    # Basic build tools
                    ["build-essential", "cmake", "git"],
                    # Libraries group 1
                    ["libgtk2.0-dev", "pkg-config", "libavcodec-dev", "libavformat-dev"],
                    # Libraries group 2
                    ["libswscale-dev", "libtbb2", "libtbb-dev"],
                    # Libraries group 3
                    ["libjpeg-dev", "libpng-dev", "libtiff-dev"],
                    # Libraries group 4
                    ["libdc1394-22-dev", "python3-dev", "python3-numpy"]
                ]
                
                all_installed = True
                for group in dependency_groups:
                    try:
                        install_cmd = ["sudo", "apt-get", "install", "-y"] + group
                        print_debug(f"Installing package group: {group}")
                        print_debug(f"Running command: {' '.join(install_cmd)}")
                        
                        if debug:
                            subprocess.run(install_cmd, check=True)
                        else:
                            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                            print_debug(f"Command output: {result.stdout}")
                            print_debug(f"Command error: {result.stderr}")
                            
                        print_success(f"Successfully installed packages: {', '.join(group)}")
                    except subprocess.CalledProcessError as e:
                        print_error(f"Failed to install packages {', '.join(group)}: {e}")
                        print_debug(f"Return code: {e.returncode}")
                        if e.returncode == 100:
                            print_warning("Error code 100 typically indicates a sudo permission problem or apt configuration issue")
                            print_warning("Try running the following commands manually:")
                            print_warning(f"  sudo apt-get update")
                            print_warning(f"  sudo apt-get install -y {' '.join(group)}")
                        elif e.returncode == 127:
                            print_warning("Error code 127 typically indicates that the command was not found")
                            print_warning("Make sure you have sudo and apt-get installed")
                        all_installed = False
                        
                if not all_installed:
                    print_warning("Some system dependencies could not be installed")
                    print_warning("You may need to install them manually using your system's package manager")
                    print_warning("OpenCV compilation may fail if required dependencies are missing")
                    # Continue anyway - don't return False here
            except Exception as e:
                print_error(f"Error during system dependencies installation: {e}")
                print_debug(f"Exception during Linux dependencies installation: {str(e)}")
                print_warning("You may need to manually install build dependencies")
                print_warning("See: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html")
        
        print_success("OpenCV build dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install OpenCV dependencies: {e}")
        print_debug(f"CalledProcessError during OpenCV dependencies installation: {e}")
        return False

def get_cuda_toolkit_path(cuda_version, debug=False):
    """
    Get the CUDA toolkit path for a specific CUDA version.
    
    Args:
        cuda_version (str): CUDA version string (e.g., "11.8", "12.4", "12.6")
        debug (bool): Whether to enable debug output
        
    Returns:
        str or None: Path to CUDA toolkit or None if not found
    """
    print_debug(f"Looking for CUDA toolkit path for version {cuda_version}")
    # Common paths for different platforms
    if platform.system() == "Windows":
        # Windows: Check Program Files and other common locations
        possible_paths = [
            f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_version}",
            os.path.expanduser(f"~\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_version}")
        ]
        
        print_debug(f"Checking Windows CUDA possible paths: {possible_paths}")
        
        for path in possible_paths:
            if os.path.exists(path):
                print_debug(f"Found CUDA path: {path}")
                return path
                
        # Check generic CUDA path that might be a symlink or the latest installation
        generic_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v" 
        if os.path.exists("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"):
            # Find all version directories
            try:
                print_debug("Checking generic CUDA path for version directories")
                cuda_dirs = [d for d in os.listdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA") 
                            if os.path.isdir(os.path.join("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA", d)) 
                            and d.startswith("v")]
                print_debug(f"Found CUDA directories: {cuda_dirs}")
                if cuda_dirs:
                    # Get the highest version that's compatible (same major version)
                    cuda_version_major = cuda_version.split(".")[0]
                    compatible_dirs = [d for d in cuda_dirs if d.startswith(f"v{cuda_version_major}")]
                    print_debug(f"Compatible CUDA directories: {compatible_dirs}")
                    if compatible_dirs:
                        compatible_dirs.sort(reverse=True)  # Sort in descending order
                        path = os.path.join("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA", compatible_dirs[0])
                        print_debug(f"Using highest compatible CUDA directory: {path}")
                        return path
            except Exception as e:
                print_warning(f"Error checking CUDA directories: {e}")
                print_debug(f"Exception during CUDA directory check: {str(e)}")
    else:
        # Linux: Check common locations
        possible_paths = [
            f"/usr/local/cuda-{cuda_version}",
            f"/usr/cuda-{cuda_version}",
            "/usr/local/cuda",  # This is often a symlink to the actual version
            "/usr/cuda"         # Another possible symlink
        ]
        
        print_debug(f"Checking Linux CUDA possible paths: {possible_paths}")
        
        for path in possible_paths:
            if os.path.exists(path):
                print_debug(f"Found path: {path}")
                # If it's a symlink and we're looking for a specific version,
                # check if it points to the version we want
                if os.path.islink(path) and cuda_version:
                    try:
                        target = os.readlink(path)
                        print_debug(f"Symlink target: {target}")
                        if f"cuda-{cuda_version}" in target or f"cuda{cuda_version}" in target:
                            print_debug(f"Target contains correct CUDA version {cuda_version}")
                            return path
                        else:
                            print_debug(f"Target is for a different CUDA version, skipping")
                            continue  # Skip if it's pointing to a different version
                    except Exception as e:
                        print_debug(f"Error reading symlink: {str(e)}")
                        pass  # If readlink fails, just use the path
                return path
    
    print_debug(f"No CUDA toolkit path found for version {cuda_version}")
    return None

def compile_opencv_from_source(pip_path, python_path, step_num, cuda_version=None, debug=False):
    """
    Compile OpenCV from source with the correct version and options.
    
    Args:
        pip_path (str): Path to pip executable
        python_path (str): Path to python executable
        step_num (int): Step number for display
        cuda_version (str, optional): CUDA version if available
        debug (bool): Whether to enable debug output
        
    Returns:
        bool: True if successful, False otherwise
    """
    print_step(step_num, "Building OpenCV from source")
    
    opencv_version = "4.11.0"  # Match the version in requirements.txt
    tmp_dir = Path(".tmp/opencv_build")
    os.makedirs(tmp_dir, exist_ok=True)
    
    print_debug(f"OpenCV version to build: {opencv_version}")
    print_debug(f"Using tmp directory: {tmp_dir.absolute()}")
    
    try:
        # First install build dependencies
        if not install_opencv_dependencies(pip_path, python_path, debug=debug):
            return False
        
        # Clone OpenCV repository
        print_warning(f"Cloning OpenCV {opencv_version}...")
        opencv_dir = tmp_dir / "opencv"
        if opencv_dir.exists():
            shutil.rmtree(opencv_dir)
        
        cmd = [
            "git", "clone", "--branch", opencv_version, "--depth", "1",
            "https://github.com/opencv/opencv.git", str(opencv_dir)
        ]
        print_debug(f"Running command: {' '.join(cmd)}")
        
        if debug:
            subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print_debug(f"Command output: {result.stdout}")
            print_debug(f"Command error: {result.stderr}")
        
        # Clone OpenCV contrib repository
        print_warning(f"Cloning OpenCV contrib {opencv_version}...")
        opencv_contrib_dir = tmp_dir / "opencv_contrib"
        if opencv_contrib_dir.exists():
            shutil.rmtree(opencv_contrib_dir)
            
        cmd = [
            "git", "clone", "--branch", opencv_version, "--depth", "1",
            "https://github.com/opencv/opencv_contrib.git", str(opencv_contrib_dir)
        ]
        print_debug(f"Running command: {' '.join(cmd)}")
        
        if debug:
            subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print_debug(f"Command output: {result.stdout}")
            print_debug(f"Command error: {result.stderr}")
        
        # Create build directory
        build_dir = opencv_dir / "build"
        os.makedirs(build_dir, exist_ok=True)
        
        # Get Python info for CMake
        py_executable = os.path.abspath(python_path)
        py_version = subprocess.check_output([python_path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                           text=True).strip()
        
        # Determine Python include and library paths
        if platform.system() == "Windows":
            # Windows paths
            py_include = subprocess.check_output([
                python_path, "-c", 
                "import sysconfig; print(sysconfig.get_path('include'))"
            ], text=True).strip()
            
            py_library = subprocess.check_output([
                python_path, "-c", 
                "import os, sysconfig; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or f'python{sysconfig.get_config_var('VERSION')}.dll'))"
            ], text=True).strip()
        else:
            # Linux paths
            py_include = subprocess.check_output([
                python_path, "-c", 
                "import sysconfig; print(sysconfig.get_path('include'))"
            ], text=True).strip()
            
            py_library = subprocess.check_output([
                python_path, "-c", 
                "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
            ], text=True).strip()
            
            # Find Python library
            py_library = os.path.join(py_library, f"libpython{py_version}.so")
            if not os.path.exists(py_library):
                py_library = os.path.join(py_library, f"libpython{py_version}m.so")
        
        # Configure with CMake
        print_warning("Configuring OpenCV build with CMake...")
        
        cmake_args = [
            "cmake",
            "-S", str(opencv_dir),
            "-B", str(build_dir),
            f"-DOPENCV_EXTRA_MODULES_PATH={str(opencv_contrib_dir)}/modules",
            "-DBUILD_opencv_world=ON",
            "-DBUILD_opencv_python3=ON",
            "-DBUILD_opencv_python2=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DWITH_IPP=OFF",
            "-DWITH_FFMPEG=ON",
            "-DWITH_TBB=ON",
            "-DWITH_OPENMP=ON",
            f"-DPYTHON3_EXECUTABLE={py_executable}",
            f"-DPYTHON3_INCLUDE_DIR={py_include}",
            f"-DPYTHON3_LIBRARY={py_library}",
            "-DOPENCV_ENABLE_NONFREE=ON"  # Enable non-free algorithms
        ]
        
        # Add CUDA support if available
        if cuda_version:
            print_warning(f"Enabling CUDA {cuda_version} support for OpenCV")
            
            # Get CUDA toolkit path
            cuda_path = get_cuda_toolkit_path(cuda_version, debug=debug)
            if not cuda_path:
                print_warning(f"Cannot find CUDA toolkit for version {cuda_version}. Using default path.")
                if platform.system() == "Windows":
                    cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_version}"
                else:
                    cuda_path = f"/usr/local/cuda-{cuda_version}"
            
            print_warning(f"Using CUDA toolkit at: {cuda_path}")
            
            # Convert CUDA version to compute capabilities
            # Define CUDA compute capabilities for different CUDA versions
            cuda_arch_bin = ""
            if cuda_version.startswith("12.6") or cuda_version.startswith("12.4"):
                # For CUDA 12.x series
                cuda_arch_bin = "5.0,6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0"
            elif cuda_version.startswith("11.8"):
                # For CUDA 11.8
                cuda_arch_bin = "3.5,5.0,5.2,6.0,6.1,7.0,7.5,8.0,8.6"
            else:
                # Default for other versions
                cuda_arch_bin = "3.5,5.0,6.0,6.1,7.0,7.5,8.0,8.6"
            
            cmake_args.extend([
                "-DWITH_CUDA=ON",
                "-DCUDA_FAST_MATH=ON",
                "-DWITH_CUBLAS=ON",
                f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}",
                "-DBUILD_opencv_cudacodec=OFF",  # Disable problematic CUDA modules
                f"-DCUDA_ARCH_BIN={cuda_arch_bin}",
                "-DCUDA_ARCH_PTX=",
                "-DWITH_NVCUVID=ON"
            ])
            
            # CUDA 12.x specific flags
            if cuda_version.startswith("12"):
                cmake_args.extend([
                    "-DOPENCV_DNN_CUDA=ON",
                    "-DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr"
                ])
            
            # Fix for CUDA 11.8 compilation issues
            if cuda_version.startswith("11.8"):
                cmake_args.extend([
                    "-DOPENCV_DNN_CUDA=OFF",  # Turn off DNN CUDA which can have issues with CUDA 11.x
                ])
        else:
            print_warning("Building OpenCV without CUDA support (no CUDA detected)")
            cmake_args.append("-DWITH_CUDA=OFF")
        
        # Filter out empty arguments
        cmake_args = [arg for arg in cmake_args if arg]
        
        print_debug(f"Running CMake with arguments: {' '.join(cmake_args)}")
        
        if debug:
            subprocess.run(cmake_args, check=True, cwd=str(build_dir))
        else:
            result = subprocess.run(cmake_args, check=True, capture_output=True, text=True, cwd=str(build_dir))
            print_debug(f"Command output: {result.stdout}")
            print_debug(f"Command error: {result.stderr}")
        
        # Build OpenCV
        print_warning("Building OpenCV (this may take a while)...")
        
        # Determine number of CPU cores for parallel build
        cpu_count = os.cpu_count() or 1
        
        if platform.system() == "Windows":
            cmd = [
                "cmake", "--build", ".", "--config", "Release", 
                "--parallel", str(cpu_count)
            ]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True, cwd=str(build_dir))
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(build_dir))
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        else:
            cmd = [
                "make", f"-j{cpu_count}"
            ]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True, cwd=str(build_dir))
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(build_dir))
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        
        # Install OpenCV
        print_warning("Installing OpenCV into virtual environment...")
        if platform.system() == "Windows":
            cmd = [
                "cmake", "--install", ".", "--prefix", 
                os.path.dirname(os.path.dirname(py_executable))
            ]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True, cwd=str(build_dir))
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(build_dir))
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        else:
            cmd = [
                "make", "install"
            ]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True, cwd=str(build_dir))
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(build_dir))
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
        
        # Verify installation
        try:
            cmd = [
                python_path, "-c", "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"
            ]
            print_debug(f"Running verification command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"OpenCV {opencv_version} installed successfully")
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
                
            print_success(f"OpenCV {opencv_version} successfully compiled and installed")
            
            # Check CUDA support in OpenCV
            if cuda_version:
                try:
                    cuda_support_cmd = (
                        "import cv2; "
                        "cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0; "
                        "print(f'CUDA support: {cuda_available}'); "
                        "print(f'CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}')"
                    )
                    cmd = [python_path, "-c", cuda_support_cmd]
                    print_debug(f"Running CUDA support check command: {' '.join(cmd)}")
                    cuda_check = subprocess.run(cmd, 
                                               capture_output=True, text=True, check=False)
                    print(cuda_check.stdout.strip())
                    if "CUDA support: True" in cuda_check.stdout:
                        print_success(f"OpenCV CUDA {cuda_version} support verified successfully")
                    else:
                        print_warning("OpenCV was compiled with CUDA but no CUDA devices are available or drivers not properly configured")
                except Exception as e:
                    print_warning(f"Could not verify OpenCV CUDA support: {e}")
                    print_debug(f"Exception during CUDA support verification: {str(e)}")
                
            return True
        except subprocess.CalledProcessError:
            print_error("OpenCV import verification failed")
            return False
        
    except subprocess.CalledProcessError as e:
        print_error(f"OpenCV compilation failed: {e}")
        print_debug(f"CalledProcessError during OpenCV compilation: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error during OpenCV compilation: {e}")
        import traceback
        print_warning(traceback.format_exc())
        print_debug(f"Exception during OpenCV compilation: {str(e)}")
        return False

def setup_opencv(pip_path, python_path, step_num=5.5, debug=False):
    """
    Install or compile OpenCV based on system capabilities.
    
    Args:
        pip_path (str): Path to pip executable
        python_path (str): Path to python executable
        step_num (int or float): Step number for display
        debug (bool): Whether to enable debug output
        
    Returns:
        bool: True if successful, False otherwise
    """
    print_step(step_num, "Setting up OpenCV")
    
    # Check for CUDA availability
    print_debug("Checking for CUDA availability")
    cuda_version = check_cuda_version(step_num=0, debug=debug)  # Use step_num=0 to avoid printing a step header
    
    # Validate CUDA version is in our supported list
    supported_versions = ['12.6', '12.4', '11.8']
    print_debug(f"Supported CUDA versions: {supported_versions}")
    if cuda_version:
        print_debug(f"Detected CUDA version: {cuda_version}")
        # Normalize to major.minor format
        cuda_version_norm = ".".join(cuda_version.split(".")[:2])
        print_debug(f"Normalized CUDA version: {cuda_version_norm}")
        
        if cuda_version_norm not in supported_versions:
            print_warning(f"CUDA {cuda_version} detected but not in supported list: {', '.join(supported_versions)}")
            print_warning(f"Will attempt to use the closest supported version for OpenCV compilation")
            
            # Find closest supported version
            major = cuda_version_norm.split('.')[0]
            if major == "12":
                if float(cuda_version_norm) >= 12.5:
                    cuda_version = "12.6"
                else:
                    cuda_version = "12.4"
            else:
                cuda_version = "11.8"  # Default to 11.8 for any other version
            
            print_warning(f"Using CUDA {cuda_version} compatibility for OpenCV")
            print_debug(f"Selected compatible CUDA version: {cuda_version}")
    
    # First try installing pre-built binaries if no CUDA is available
    if not cuda_version:
        print_warning("Attempting to install pre-built OpenCV packages...")
        
        try:
            # Uninstall existing OpenCV packages first to avoid conflicts
            cmd = [pip_path, "uninstall", "-y", "opencv-python", "opencv-python-headless"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd, check=False)
            else:
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
            
            # Try to install the specific version required
            print_debug("Installing specific OpenCV version 4.11.0.86")
            cmd = [pip_path, "install", "opencv-python==4.11.0.86", "opencv-python-headless==4.11.0.86"]
            print_debug(f"Running command: {' '.join(cmd)}")
            
            if debug:
                subprocess.run(cmd)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print_debug(f"Command output: {result.stdout}")
                print_debug(f"Command error: {result.stderr}")
            
            # Check if installation was successful by importing cv2
            cmd = [python_path, "-c", "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"]
            print_debug(f"Running verification command: {' '.join(cmd)}")
            
            if debug:
                import_check = subprocess.run(cmd)
                if import_check.returncode == 0:
                    print_success("Pre-built OpenCV installed successfully")
                    return True
            else:
                import_check = subprocess.run(cmd, capture_output=True, text=True)
                if import_check.returncode == 0:
                    print_success(f"Pre-built OpenCV installed successfully: {import_check.stdout.strip()}")
                    return True
                else:
                    print_warning("Pre-built OpenCV installation failed or verification failed")
                    print_warning(f"Error: {import_check.stderr.strip()}")
                    print_warning("Falling back to compiling from source...")
                    print_debug(f"Verification failed with return code {import_check.returncode}")
                    print_debug(f"Error output: {import_check.stderr}")
        except Exception as e:
            print_warning(f"Error installing pre-built OpenCV: {e}")
            print_warning("Falling back to compiling from source...")
            print_debug(f"Exception during pre-built OpenCV installation: {str(e)}")
    else:
        print_warning(f"CUDA {cuda_version} detected. Compiling OpenCV from source with CUDA support...")
    
    # Compile from source (either because CUDA is available or pre-built installation failed)
    return compile_opencv_from_source(pip_path, python_path, step_num + 0.1, cuda_version, debug)
