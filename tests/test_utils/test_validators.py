"""
Tests for validator utility functions in utils/validators.py.
"""
import pytest
from inquirer import errors
from utils.validators import validate_number, validate_positive_number

class TestValidators:
    """Test suite for validation utility functions."""
    
    def test_validate_number(self):
        """Test that validate_number accepts valid integers and rejects non-integers."""
        # Valid inputs
        validate_number(None, "123")  # Should not raise exception
        validate_number(None, "-123")  # Should not raise exception
        validate_number(None, "0")  # Should not raise exception
        validate_number(None, "")
        
        # Invalid inputs should raise ValidationError
        with pytest.raises(errors.ValidationError):
            validate_number(None, "123.45")  # Floating-point numbers are not allowed
        
        with pytest.raises(errors.ValidationError):
            validate_number(None, "abc")
    
    def test_validate_positive_number(self):
        """Test that validate_positive_number accepts positive numbers and rejects others."""
        # Valid inputs
        validate_positive_number(None, "123")  # Should not raise exception
        validate_positive_number(None, "")
        
        # Invalid inputs should raise ValidationError
        with pytest.raises(errors.ValidationError):
            validate_positive_number(None, "0.5")
        
        with pytest.raises(errors.ValidationError):
            validate_positive_number(None, "-123")
        
        with pytest.raises(errors.ValidationError):
            validate_positive_number(None, "0")
        
        with pytest.raises(errors.ValidationError):
            validate_positive_number(None, "abc")
