"""
Tests for UI helper functions in utils/ui_helpers.py.
"""
import pytest
from utils.ui_helpers import separator

class TestUIHelpers:
    """Test suite for UI helper functions."""
    
    def test_separator(self):
        """Test that separator function returns correct dictionary format."""
        # Test with a simple string
        result = separator("Test Separator")
        
        # Verify the format of the returned dictionary
        assert isinstance(result, dict)
        assert "name" in result
        assert "disabled" in result
        assert result["name"] == "Test Separator"
        assert result["disabled"] == "──────────────"
        
        # Test with an empty string
        empty_result = separator("")
        assert empty_result["name"] == ""
        assert empty_result["disabled"] == "──────────────"
        
        # Test with special characters
        special_result = separator("👋 Hello 🌎 World!")
        assert special_result["name"] == "👋 Hello 🌎 World!"
        assert special_result["disabled"] == "──────────────"
