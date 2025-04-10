import subprocess

from .utilities import print_step, print_success, print_warning, print_error

def verify_installations(python_path, step_num=7):
    """
    Verify that all necessary dependencies are installed correctly.
    
    Args:
        python_path (str): Path to the Python executable
        step_num (int or float): Step number to display in the console output
        
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
