"""
Input validation utilities.
"""
from inquirer import errors

def validate_number(_, current):
    """
    Validate that input is a number or empty (for default).
    
    Args:
        _: Unused parameter (required by inquirer)
        current: The current input value
        
    Returns:
        bool: True if valid, otherwise raises ValidationError
    """
    try:
        if current.strip() == "":  # Allow empty for default values
            return True
        _ = int(current)
        return True
    except ValueError:
        raise errors.ValidationError('', reason='Please enter a valid number')


def validate_positive_number(_, current):
    """
    Validate that input is a positive number or empty (for default).
    
    Args:
        _: Unused parameter (required by inquirer)
        current: The current input value
        
    Returns:
        bool: True if valid, otherwise raises ValidationError
    """
    try:
        if current.strip() == "":  # Allow empty for default values
            return True
        value = int(current)
        if value <= 0:
            raise ValueError("Value must be positive")
        return True
    except ValueError:
        raise errors.ValidationError('', reason='Please enter a valid positive number')
