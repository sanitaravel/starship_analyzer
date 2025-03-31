import argparse

from .utilities import (
    print_step, print_success, print_warning, print_error,
    print_next_steps, BOLD, RESET
)
from .environment import create_virtual_environment, create_required_directories
from .gpu import check_cuda_version, install_nvidia_drivers, install_cuda_toolkit
from .dependencies import install_dependencies
from .verification import verify_installations

def parse_arguments():
    """Parse command line arguments for setup script."""
    parser = argparse.ArgumentParser(description="Setup script for Starship Analyzer")
    parser.add_argument("--unattended", action="store_true", help="Run in unattended mode without prompts")
    parser.add_argument("--recreate-venv", action="store_true", help="Recreate virtual environment if it exists")
    parser.add_argument("--keep-venv", action="store_true", help="Keep existing virtual environment if it exists")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only installation")
    parser.add_argument("--setup-gpu", action="store_true", help="Attempt GPU setup if CUDA not detected")
    return parser.parse_args()

def run_setup():
    """Main setup function."""
    args = parse_arguments()
    
    print("\n" + "="*60)
    print(f"{BOLD}Starship Analyzer Setup{RESET}")
    
    if args.unattended:
        print(f"{BOLD}[Unattended Mode]{RESET}")
    
    print("="*60 + "\n")
    
    # Step 1: Create virtual environment
    venv_created = create_virtual_environment(
        1, 
        unattended=args.unattended, 
        recreate=args.recreate_venv, 
        keep=args.keep_venv
    )
    
    if not venv_created:
        print_error("Failed to create virtual environment. Aborting setup.")
        return
    
    # Step 2: Create required directories
    create_required_directories(2)
    
    # Step 3: Check CUDA version before installing dependencies
    cuda_version = check_cuda_version(3)
    
    # Step 4-5: Optional GPU setup if CUDA not detected
    if not cuda_version and not args.force_cpu:
        print_warning("CUDA not detected. GPU acceleration requires NVIDIA drivers and CUDA toolkit.")
        
        setup_gpu = False
        if args.unattended:
            setup_gpu = args.setup_gpu
            if setup_gpu:
                print_warning("Unattended mode: Attempting GPU setup")
            else:
                print_warning("Unattended mode: Skipping GPU setup")
        else:
            setup_gpu = input("Would you like guidance on setting up GPU support? (y/n): ").lower().strip() == 'y'
            
        if setup_gpu:
            # Step 4: Install NVIDIA drivers
            install_nvidia_drivers(4)
            # Step 5: Install CUDA Toolkit
            install_cuda_toolkit(5)
            
            # Check CUDA version again after installation
            print_warning("Checking for CUDA again after installation...")
            cuda_version = check_cuda_version(5.5)  # Using 5.5 to indicate it's between steps 5 and 6
            
            if cuda_version:
                print_success(f"CUDA {cuda_version} successfully detected after installation!")
            else:
                print_warning("CUDA still not detected. Continuing with CPU-only installation.")
                
                continue_anyway = True
                if not args.unattended:
                    continue_anyway = input("Continue with CPU-only installation? (y/n): ").lower().strip() == 'y'
                    
                if not continue_anyway:
                    print_warning("Setup paused. Please ensure CUDA is properly installed and run this script again.")
                    return
    
    # Step 6: Install dependencies with the detected CUDA version
    deps_installed = False
    if venv_created:
        deps_installed = install_dependencies(cuda_version, 6, force_cpu=args.force_cpu)
        if not deps_installed:
            print_error("Failed to install dependencies. Aborting setup.")
            return
    
    import platform
    import os
    
    # Get the python path for verification
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
    
    # Step 7: Verify installations as a separate step
    all_successful = False
    gpu_available = False
    if deps_installed:
        all_successful, gpu_available = verify_installations(python_path, 7)
    
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
