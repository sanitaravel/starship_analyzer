import platform

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"  # Blue for informational messages
CYAN = "\033[96m"  # Cyan for debug messages
BOLD = "\033[1m"
RESET = "\033[0m"

def print_step(step_num, message):
    """Print a step in the setup process."""
    print(f"{BOLD}Step {step_num}: {message}{RESET}")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_info(message):
    """Print an informational message."""
    print(f"{BLUE}ℹ {message}{RESET}")

def print_warning(message):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def print_debug(message, debug=False):
    """
    Print a debug message only if debug mode is enabled.
    """
    if debug:
        print(f"{CYAN}🔍 {message}{RESET}")

def print_next_steps():
    """Print instructions for the next steps."""
    print("\n" + "="*60)
    print(f"{BOLD}Setup Complete!{RESET}")
    print("="*60)
    
    # Activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"\n{BOLD}Next Steps:{RESET}")
    print(f"1. Activate the virtual environment:")
    print(f"   {YELLOW}{activate_cmd}{RESET}")
    print(f"2. You can download flight recordings directly through the app")
    print(f"   or manually place them in the 'flight_recordings' directory.")
    print(f"3. Run the application:")
    print(f"   {YELLOW}python main.py{RESET}")
    print("\n" + "="*60 + "\n")
