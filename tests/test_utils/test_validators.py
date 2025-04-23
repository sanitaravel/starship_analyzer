"""
Tests for validator utility functions in utils/validators.py.
"""
import pytest
from inquirer import errors
from utils.validators import validate_number, validate_positive_number, validate_url

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

class TestURLValidator:
    """Test cases for URL validation functions."""
    
    def test_valid_youtube_urls(self):
        """Test that valid YouTube URLs are accepted."""
        valid_youtube_urls = [
            "https://www.youtube.com/watch?v=-1wcilQ58hI",
            "https://youtube.com/watch?v=abcdefghijk",
            "http://www.youtube.com/watch?v=12345",
            "https://youtu.be/abcdefg",
            "http://youtu.be/12345"
        ]
        
        for url in valid_youtube_urls:
            assert validate_url(url) is True, f"URL should be valid: {url}"
    
    def test_valid_twitter_urls(self):
        """Test that valid Twitter/X URLs are accepted."""
        valid_twitter_urls = [
            "https://x.com/i/broadcasts/1OwGWNYrzZVKQ",
            "https://twitter.com/i/broadcasts/1OwGWNYrzZVKQ",
            "https://x.com/username/status/12345",
            "https://twitter.com/username/status/12345",
            "http://twitter.com/username"
        ]
        
        for url in valid_twitter_urls:
            assert validate_url(url) is True, f"URL should be valid: {url}"
    
    def test_invalid_urls(self):
        """Test that invalid URLs are rejected."""
        invalid_urls = [
            "https://example.com",
            "https://facebook.com/video",
            "ftp://youtube.com/watch?v=123",
            "youtube.com/watch?v=123",  # Missing protocol
            "https://twitch.tv/videos/123",
            "",  # Empty URL
            "not a url at all",
            "https://twitter.org/fake",  # Wrong TLD
            "https://ytube.com/watch?v=123"  # Typo in domain
        ]
        
        for url in invalid_urls:
            assert validate_url(url) is not True, f"URL should be invalid: {url}"

