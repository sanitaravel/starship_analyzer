"""
Main entry point for the Starship Analyzer application.
"""
import logging
from utils.logger import start_new_session, get_logger, set_global_log_level
from utils.terminal import clear_screen
from ui import display_menu

# Initialize logger
logger = get_logger(__name__)

# Global debug state - kept in main as requested
DEBUG_MODE = False

def toggle_debug_mode():
    """Toggle debug mode on/off and set appropriate log levels."""
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    
    if DEBUG_MODE:
        logger.info("Debug mode enabled")
        set_global_log_level(logging.DEBUG)
    else:
        logger.info("Debug mode disabled")
        set_global_log_level(logging.INFO)
    
    return True

def main() -> None:
    """
    Main function to handle the step-by-step menu and run the application.
    """
    # Clear screen at startup
    clear_screen()
    
    # Start a new logging session
    root_logger = start_new_session()
    logger.info("Starting Starship Analyzer application")
    
    try:
        menu_result = True
        while menu_result:
            debug_status = "Enabled" if DEBUG_MODE else "Disabled"
            menu_result = display_menu(debug_status)
            
            # Special return value to toggle debug mode
            if menu_result == "TOGGLE_DEBUG":
                toggle_debug_mode()
                menu_result = True
                
        logger.info("Application exiting normally")
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
