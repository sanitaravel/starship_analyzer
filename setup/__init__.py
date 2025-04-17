import os
import sys
import platform
import argparse

from .environment import create_virtual_environment, create_required_directories
from .gpu import check_cuda_version, install_nvidia_drivers, install_cuda_toolkit
from .dependencies import install_dependencies
from .verification import verify_installations
from .utilities import print_step, print_success, print_info, print_warning, print_error, print_debug, print_next_steps

def run_setup(args=None):
    """
    Main entry point for the setup process.
    
    Args:
        args: Command line arguments from argparse (optional)
    """
    if args is None:
        # Parse command line arguments if not provided
        parser = argparse.ArgumentParser(description="Starship Analyzer Setup")
        parser.add_argument("--update", action="store_true", help="Update the application")
        parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only installation")
        parser.add_argument("--unattended", action="store_true", help="Run in unattended mode")
        parser.add_argument("--recreate", action="store_true", help="Recreate virtual environment")
        parser.add_argument("--keep", action="store_true", help="Keep existing virtual environment")
        parser.add_argument("--debug", action="store_true", help="Show detailed installation output")
        args = parser.parse_args()
    
    # If in update mode, skip environment creation and just update dependencies
    if args.update:
        run_update(args)
        return
    
    print("\n" + "="*60)
    print("   Starship Analyzer Setup")
    print("="*60 + "\n")
    
    print_debug("Debug mode enabled: Showing detailed output", args.debug)
    
    # Create virtual environment
    if not create_virtual_environment(
        step_num=1, 
        unattended=args.unattended, 
        recreate=args.recreate, 
        keep=args.keep,
        debug=args.debug
    ):
        return
    
    # Create required directories
    create_required_directories(step_num=2)
    
    # Check CUDA version for PyTorch installation
    cuda_version = check_cuda_version(step_num=3, debug=args.debug)
    
    # Get the python path based on the OS
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
    
    # Install dependencies
    install_success = install_dependencies(
        cuda_version,
        step_num=6,
        force_cpu=args.force_cpu,
        debug=args.debug
    )
    
    if not install_success:
        print_error("Setup failed at the dependency installation stage")
        return
    
    # Verify installations
    success, gpu_available = verify_installations(python_path, step_num=7, debug=args.debug)
    
    # Print next steps if installation was successful
    if success:
        print_next_steps()
    else:
        print_error("Setup completed with errors")

def run_update(args):
    """
    Run the application update process.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("   Starship Analyzer Update")
    print("="*60 + "\n")
    
    # Check if virtual environment exists
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")
    
    if not os.path.exists(python_path):
        print_error("Virtual environment not found. Please run the setup first.")
        return
    
    print_step(1, "Updating Starship Analyzer")
    
    # Check CUDA version
    cuda_version = check_cuda_version(step_num=2)
    
    # Update dependencies
    print_step(3, "Updating dependencies")
    install_success = install_dependencies(
        cuda_version,
        step_num=3,
        force_cpu=args.force_cpu,
        debug=args.debug
    )
    
    if not install_success:
        print_error("Update failed at the dependency installation stage")
        return
    
    # Verify installations
    success, gpu_available = verify_installations(python_path, step_num=4)
    
    if success:
        print("\n" + "="*60)
        print_success("Starship Analyzer has been successfully updated!")
        print("="*60 + "\n")
        
        # Print activation instructions
        if platform.system() == "Windows":
            activate_cmd = "venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print(f"To start using the updated application:")
        print(f"1. Activate the virtual environment if not already activated:")
        print_warning(activate_cmd)
        print(f"2. Run the application:")
        print_warning('python main.py')
        print("\n" + "="*60 + "\n")
    else:
        print_error("Update completed with errors")
