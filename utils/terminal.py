"""
Terminal utility functions.
"""
import os
import platform

def clear_screen():
    """
    Clear the terminal screen based on the operating system.
    """
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')  # For Linux/Mac
