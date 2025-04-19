import subprocess

from .utilities import print_step, print_success, print_info, print_warning, print_error, print_debug

def verify_installations(python_path, step_num=7, debug=False):
    """
    Verify that all necessary dependencies are installed correctly.
    
    Args:
        python_path (str): Path to the Python executable
        step_num (int or float): Step number to display in the console output
        debug (bool): Whether to show detailed debug output
        
    Returns:
        tuple: (bool for success, bool for GPU available)
    """
    print_step(step_num, "Verifying installations")
    
    # List of core dependencies to verify
    dependencies = [
        ("numpy", "NumPy (array processing)"),
        ("cv2", "OpenCV (image processing)"),
        ("torch", "PyTorch (deep learning)"),
        ("easyocr", "EasyOCR (optical character recognition)")
    ]
    
    all_successful = True
    gpu_available = False
    
    print_info("Checking installed packages...")
    for module_name, description in dependencies:
        try:
            # Try to import the module
            cmd = f"import {module_name}; print('Success')"
            print_info(f"Verifying {description}...")
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
                
                # For PyTorch, check if it's CPU-only or CUDA version
                if module_name == "torch":
                    torch_info_cmd = "import torch; print(torch.__version__)"
                    torch_info = subprocess.run([python_path, "-c", torch_info_cmd],
                                            capture_output=True, text=True, check=False).stdout.strip()
                    
                    if "+cpu" in torch_info:
                        print_warning(f"{description} - Installed ({torch_info}) - CPU-only version")
                        print_debug(f"PyTorch is CPU-only despite possible CUDA installation", debug)
                    else:
                        print_success(f"{description} - Installed ({torch_info})")
                else:
                    print_success(f"{description} - Installed ({version})")
                
                print_debug(f"Full import details for {module_name}: Success with version {version}", debug)
            else:
                print_error(f"{description} - Failed to import")
                all_successful = False
                print_warning(f"Error: {result.stderr.strip()}")
                print_debug(f"Full import error for {module_name}: {result.stderr}", debug)
        except Exception as e:
            print_error(f"{description} - Error during verification: {e}")
            print_debug(f"Exception details: {repr(e)}", debug)
            all_successful = False
    
    # Special check for GPU support
    try:
        print_info("Checking GPU availability...")
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
            print_debug(f"GPU device information: {gpu_name}", debug)
            
            # Check which CUDA version PyTorch is using
            print_info("Checking CUDA version...")
            cuda_ver_cmd = "import torch; print(torch.version.cuda)"
            cuda_ver_check = subprocess.run([python_path, "-c", cuda_ver_cmd], 
                                         capture_output=True, text=True, check=False)
            if cuda_ver_check.returncode == 0:
                cuda_ver = cuda_ver_check.stdout.strip()
                print_success(f"PyTorch is using CUDA version: {cuda_ver}")
                print_debug(f"Additional CUDA info - PyTorch built with: {cuda_ver}", debug)
        else:
            # Additional check to understand why CUDA is not available
            cuda_issue_cmd = "import torch; print('CUDA Available:', torch.cuda.is_available()); print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') else 'Not available')"
            issue_check = subprocess.run([python_path, "-c", cuda_issue_cmd], 
                                      capture_output=True, text=True, check=False)
            
            print_warning("GPU Acceleration - Not available (EasyOCR will run in CPU mode)")
            print_warning(f"PyTorch CUDA details: {issue_check.stdout.strip()}")
            print_debug("No GPU detected: torch.cuda.is_available() returned False", debug)
    except Exception as e:
        print_warning(f"GPU Acceleration - Could not verify: {e}")
        print_debug(f"GPU verification exception details: {repr(e)}", debug)
    
    # Return overall success status
    if all_successful:
        print_success("All core dependencies were installed successfully")
    else:
        print_error("Some dependencies failed to install correctly")
        print_warning("Try manually installing the missing packages or check for errors")
    
    return all_successful, gpu_available
