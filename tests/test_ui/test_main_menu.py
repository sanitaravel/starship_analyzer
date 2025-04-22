"""
Tests for the main_menu module.
"""
import pytest
from unittest.mock import patch, MagicMock
import inquirer

# Import the module to test
from ui.main_menu import display_menu

class TestMainMenu:
    """Tests for the main menu functions."""
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.video_processing_menu')
    @patch('ui.main_menu.clear_screen')
    def test_video_processing_option(self, mock_clear, mock_video_menu, mock_prompt):
        """Test selecting the video processing option."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Video Processing'}
        
        # Call function
        result = display_menu("Disabled")
        
        # Verify results
        assert result is True
        mock_clear.assert_called_once()
        mock_video_menu.assert_called_once()
        mock_prompt.assert_called_once()
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.visualization_menu')
    @patch('ui.main_menu.clear_screen')
    def test_visualization_option(self, mock_clear, mock_visualization_menu, mock_prompt):
        """Test selecting the data visualization option."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Data Visualization'}
        
        # Call function
        result = display_menu("Disabled")
        
        # Verify results
        assert result is True
        mock_clear.assert_called_once()
        mock_visualization_menu.assert_called_once()
        mock_prompt.assert_called_once()
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.download_media_menu')
    @patch('ui.main_menu.clear_screen')
    def test_download_media_option(self, mock_clear, mock_download_menu, mock_prompt):
        """Test selecting the download media option."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Download Media'}
        
        # Call function
        result = display_menu("Disabled")
        
        # Verify results
        assert result is True
        mock_clear.assert_called_once()
        mock_download_menu.assert_called_once()
        mock_prompt.assert_called_once()
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.clear_screen')
    def test_toggle_debug_option(self, mock_clear, mock_prompt):
        """Test selecting the toggle debug mode option."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Toggle Debug Mode (Currently: Disabled)'}
        
        # Call function
        result = display_menu("Disabled")
        
        # Verify results
        assert result == "TOGGLE_DEBUG"
        mock_clear.assert_called_once()
        mock_prompt.assert_called_once()
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.clear_screen')
    @patch('ui.main_menu.print')
    def test_exit_option(self, mock_print, mock_clear, mock_prompt):
        """Test selecting the exit option."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Exit'}
        
        # Call function
        result = display_menu("Disabled")
        
        # Verify results
        assert result is False
        # clear_screen is called twice: once at the start and once before exiting
        assert mock_clear.call_count == 2
        mock_print.assert_called_once_with("Exiting the program.")
        mock_prompt.assert_called_once()
    
    @patch('ui.main_menu.inquirer.prompt')
    @patch('ui.main_menu.logger')
    def test_logging(self, mock_logger, mock_prompt):
        """Test that user selections are logged."""
        # Setup mock response
        mock_prompt.return_value = {'action': 'Exit'}
        
        # Call function
        display_menu("Disabled")
        
        # Verify results
        mock_logger.debug.assert_called_once_with("Main menu: User selected: Exit")
    
    @patch('ui.main_menu.inquirer.prompt')
    def test_debug_status_display(self, mock_prompt):
        """Test that debug status is correctly displayed in the menu."""
        # Setup mock
        mock_prompt.return_value = {'action': 'Exit'}
        
        # Call function
        display_menu("Enabled")
        
        # Verify results
        args, _ = mock_prompt.call_args
        questions = args[0]
        choices = questions[0].choices
        
        # Check that the debug status is correctly displayed
        assert any('Toggle Debug Mode (Currently: Enabled)' in choice for choice in choices)
