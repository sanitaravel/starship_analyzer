"""
Tests for setup/utilities.py functions.
"""
import platform
from unittest.mock import patch

from setup.utilities import (
    print_step, print_success, print_info, print_warning, 
    print_error, print_debug, print_next_steps,
    GREEN, YELLOW, RED, BLUE, CYAN, BOLD, RESET
)

class TestUtilities:
    """Test suite for utilities module functions."""
    
    @patch('builtins.print')
    def test_print_step(self, mock_print):
        """Test print_step function formats step message correctly."""
        print_step(1, "Test Step")
        mock_print.assert_called_once_with(f"{BOLD}Step 1: Test Step{RESET}")
    
    @patch('builtins.print')
    def test_print_success(self, mock_print):
        """Test print_success function formats success message correctly."""
        print_success("Success message")
        mock_print.assert_called_once_with(f"{GREEN}✓ Success message{RESET}")
    
    @patch('builtins.print')
    def test_print_info(self, mock_print):
        """Test print_info function formats info message correctly."""
        print_info("Info message")
        mock_print.assert_called_once_with(f"{BLUE}ℹ Info message{RESET}")
    
    @patch('builtins.print')
    def test_print_warning(self, mock_print):
        """Test print_warning function formats warning message correctly."""
        print_warning("Warning message")
        mock_print.assert_called_once_with(f"{YELLOW}⚠ Warning message{RESET}")
    
    @patch('builtins.print')
    def test_print_error(self, mock_print):
        """Test print_error function formats error message correctly."""
        print_error("Error message")
        mock_print.assert_called_once_with(f"{RED}✗ Error message{RESET}")
    
    @patch('builtins.print')
    def test_print_debug_enabled(self, mock_print):
        """Test print_debug function prints when debug is enabled."""
        print_debug("Debug message", True)
        mock_print.assert_called_once_with(f"{CYAN}🔍 Debug message{RESET}")
    
    @patch('builtins.print')
    def test_print_debug_disabled(self, mock_print):
        """Test print_debug function does not print when debug is disabled."""
        print_debug("Debug message", False)
        mock_print.assert_not_called()
    
    @patch('builtins.print')
    @patch('platform.system')
    def test_print_next_steps_windows(self, mock_system, mock_print):
        """Test print_next_steps function on Windows."""
        mock_system.return_value = "Windows"
        print_next_steps()
        
        # Verify print was called multiple times
        assert mock_print.call_count > 5
        
        # Get all the arguments passed to print calls
        printed_lines = [args[0] for args, _ in mock_print.call_args_list]
        
        # Convert to string for easier searching
        printed_text = '\n'.join(printed_lines)
        
        # Check that Windows activation command appears in the output
        assert "venv\\Scripts\\activate" in printed_text
    
    @patch('builtins.print')
    @patch('platform.system')
    def test_print_next_steps_linux(self, mock_system, mock_print):
        """Test print_next_steps function on Linux/macOS."""
        mock_system.return_value = "Linux"
        print_next_steps()
        
        # Verify print was called multiple times
        assert mock_print.call_count > 5
        
        # Get all the arguments passed to print calls
        printed_lines = [args[0] for args, _ in mock_print.call_args_list]
        
        # Convert to string for easier searching
        printed_text = '\n'.join(printed_lines)
        
        # Check that Linux activation command appears in the output
        assert "source venv/bin/activate" in printed_text
