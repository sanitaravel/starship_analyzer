"""
UI helper functions.
"""

def separator(text):
    """
    Create a separator for the menu.
    
    Args:
        text (str): Text to display in the separator
        
    Returns:
        dict: Separator configuration for inquirer
    """
    return {'name': text, 'disabled': '──────────────'}
