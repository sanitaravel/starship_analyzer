import platform

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
BLUE = "\033[94m"  # Added blue color for debug messages
RESET = "\033[0m"

# Global debug flag
DEBUG = False

def set_debug_mode(debug=False):
    """Set the global debug flag."""
    global DEBUG
    DEBUG = debug

def print_step(step_num, message):
    """Print a step in the setup process."""
    print(f"{BOLD}Step {step_num}: {message}{RESET}")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def print_debug(message):
    """Print a debug message if debug mode is enabled."""
    if DEBUG:
        print(f"{BLUE}DEBUG: {message}{RESET}")

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
    print(f"2. Place your flight recordings in the 'flight_recordings' directory")
    print(f"3. Run the application:")
    print(f"   {YELLOW}python main.py{RESET}")
    print("\n" + "="*60 + "\n")
