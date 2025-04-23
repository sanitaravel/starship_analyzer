"""
Tests for the visualization_menu module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import inquirer

# Import the module to test
from ui.visualization_menu import (
    visualization_menu, 
    visualize_flight_data,
    compare_multiple_launches_menu,
    get_launch_folders,
    validate_available_launches,
    prompt_for_comparison_options,
    validate_selected_launches,
    execute_launch_comparison
)

class TestMainVisualizationMenu:
    """Tests for the main visualization menu functionality."""
    
    @patch('ui.visualization_menu.inquirer.prompt')
    @patch('ui.visualization_menu.visualize_flight_data')
    @patch('ui.visualization_menu.compare_multiple_launches_menu')
    @patch('ui.visualization_menu.clear_screen')
    def test_visualization_menu_flight_data(self, mock_clear, mock_compare, 
                                           mock_visualize, mock_prompt):
        """Test visualization menu when selecting to visualize flight data."""
        # Setup mock responses
        mock_prompt.side_effect = [
            {'action': 'Visualize flight data'},
            {'action': 'Back to main menu'}
        ]
        
        # Call function
        result = visualization_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        mock_visualize.assert_called_once()
        mock_compare.assert_not_called()
        assert mock_prompt.call_count == 2
    
    @patch('ui.visualization_menu.inquirer.prompt')
    @patch('ui.visualization_menu.visualize_flight_data')
    @patch('ui.visualization_menu.compare_multiple_launches_menu')
    @patch('ui.visualization_menu.clear_screen')
    def test_visualization_menu_compare_launches(self, mock_clear, mock_compare, 
                                               mock_visualize, mock_prompt):
        """Test visualization menu when selecting to compare multiple launches."""
        # Setup mock responses
        mock_prompt.side_effect = [
            {'action': 'Visualize multiple launches data'},
            {'action': 'Back to main menu'}
        ]
        
        # Call function
        result = visualization_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        mock_compare.assert_called_once()
        mock_visualize.assert_not_called()
        assert mock_prompt.call_count == 2
    
    @patch('ui.visualization_menu.inquirer.prompt')
    @patch('ui.visualization_menu.clear_screen')
    def test_visualization_menu_back(self, mock_clear, mock_prompt):
        """Test visualization menu when selecting to go back to main menu."""
        # Setup mock responses
        mock_prompt.return_value = {'action': 'Back to main menu'}
        
        # Call function
        result = visualization_menu()
        
        # Verify results
        assert result is True
        mock_clear.assert_called()
        assert mock_prompt.call_count == 1


class TestFlightDataVisualization:
    """Tests for flight data visualization functionality."""
    
    @patch('ui.visualization_menu.os.listdir')
    @patch('ui.visualization_menu.os.path.isdir')
    @patch('ui.visualization_menu.inquirer.prompt')
    @patch('ui.visualization_menu.plot_flight_data')
    @patch('ui.visualization_menu.input')  # To mock the "Press Enter to continue"
    @patch('ui.visualization_menu.clear_screen')
    def test_visualize_flight_data(self, mock_clear, mock_input, mock_plot, 
                                  mock_prompt, mock_isdir, mock_listdir):
        """Test visualize flight data functionality."""
        # Setup mocks
        mock_listdir.return_value = ['launch1', 'launch2', 'compare_launches']
        mock_isdir.return_value = True
        mock_prompt.return_value = {
            'launch_folder': 'launch1',
            'start_time': '10',
            'end_time': '100',
            'show_figures': True
        }
        
        # Call function
        result = visualize_flight_data()
        
        # Verify results
        assert result is True
        mock_plot.assert_called_once()
        mock_plot.assert_called_with(
            os.path.join('.', 'results', 'launch1', 'results.json'),
            10, 100, show_figures=True
        )
        mock_input.assert_called_once()
        mock_clear.assert_called()
    
    @patch('ui.visualization_menu.os.listdir')
    @patch('ui.visualization_menu.os.path.isdir')
    @patch('ui.visualization_menu.input')  # To mock the "Press Enter to continue"
    @patch('ui.visualization_menu.clear_screen')
    def test_visualize_flight_data_no_folders(self, mock_clear, mock_input, 
                                            mock_isdir, mock_listdir):
        """Test visualize flight data when no folders exist."""
        # Setup mocks
        mock_listdir.return_value = []
        mock_isdir.return_value = False
        
        # Call function
        result = visualize_flight_data()
        
        # Verify results
        assert result is True
        mock_input.assert_called_once()
        mock_clear.assert_called()


class TestLaunchFolders:
    """Tests for launch folder management."""
    
    @patch('ui.visualization_menu.os.listdir')
    @patch('ui.visualization_menu.os.path.isdir')
    def test_get_launch_folders(self, mock_isdir, mock_listdir):
        """Test getting launch folders."""
        # Setup mocks
        mock_listdir.return_value = ['launch1', 'launch2', 'compare_launches']
        mock_isdir.return_value = True
        
        # Call function
        result = get_launch_folders()
        
        # Verify results
        assert len(result) == 2
        assert 'launch1' in result
        assert 'launch2' in result
        assert 'compare_launches' not in result
    
    def test_validate_available_launches_sufficient(self):
        """Test validation when sufficient launch folders exist."""
        launch_folders = ['launch1', 'launch2']
        
        with patch('ui.visualization_menu.input'), patch('ui.visualization_menu.clear_screen'):
            result = validate_available_launches(launch_folders)
            assert result is True
    
    def test_validate_available_launches_insufficient(self):
        """Test validation when insufficient launch folders exist."""
        launch_folders = ['launch1']
        
        with patch('ui.visualization_menu.input') as mock_input, \
             patch('ui.visualization_menu.clear_screen') as mock_clear:
            result = validate_available_launches(launch_folders)
            assert result is False
            mock_input.assert_called_once()
            mock_clear.assert_called_once()


class TestLaunchComparison:
    """Tests for launch comparison functionality."""
    
    @patch('ui.visualization_menu.inquirer.prompt')
    def test_prompt_for_comparison_options(self, mock_prompt):
        """Test prompting for comparison options."""
        # Setup mock
        expected_result = {
            'launches': ['launch1', 'launch2'],
            'start_time': '10',
            'end_time': '100',
            'show_figures': True
        }
        mock_prompt.return_value = expected_result
        launch_folders = ['launch1', 'launch2', 'launch3']
        
        # Call function
        result = prompt_for_comparison_options(launch_folders)
        
        # Verify results
        assert result == expected_result
        mock_prompt.assert_called_once()
        # Verify that the questions list was passed to prompt
        args, _ = mock_prompt.call_args
        assert len(args[0]) == 4  # Should have 4 questions
        assert args[0][0].name == 'launches'
        assert args[0][0].choices == launch_folders
    
    def test_validate_selected_launches_sufficient(self):
        """Test validation when sufficient launches are selected."""
        selected_launches = ['launch1', 'launch2']
        
        result = validate_selected_launches(selected_launches)
        assert result is True
    
    def test_validate_selected_launches_insufficient(self):
        """Test validation when insufficient launches are selected."""
        selected_launches = ['launch1']
        
        with patch('ui.visualization_menu.input') as mock_input, \
             patch('ui.visualization_menu.clear_screen') as mock_clear:
            result = validate_selected_launches(selected_launches)
            assert result is False
            mock_input.assert_called_once()
            mock_clear.assert_called_once()
    
    @patch('ui.visualization_menu.compare_multiple_launches')
    def test_execute_launch_comparison(self, mock_compare):
        """Test execution of launch comparison."""
        # Setup
        launches = ['launch1', 'launch2']
        start_time = '10'
        end_time = '100'
        show_figures = True
        
        # Call function
        execute_launch_comparison(launches, start_time, end_time, show_figures)
        
        # Verify results
        mock_compare.assert_called_once()
        args, kwargs = mock_compare.call_args
        assert args[0] == 10  # start_time converted to int
        assert args[1] == 100  # end_time converted to int
        assert len(args) == 4  # start_time, end_time, and 2 json paths
        assert kwargs['show_figures'] is True


class TestComparisonMenu:
    """Tests for the launch comparison menu workflow."""
    
    @patch('ui.visualization_menu.get_launch_folders')
    @patch('ui.visualization_menu.validate_available_launches')
    @patch('ui.visualization_menu.prompt_for_comparison_options')
    @patch('ui.visualization_menu.validate_selected_launches')
    @patch('ui.visualization_menu.execute_launch_comparison')
    @patch('ui.visualization_menu.input')
    @patch('ui.visualization_menu.clear_screen')
    def test_compare_multiple_launches_menu_successful(self, mock_clear, mock_input, 
                                                     mock_execute, mock_validate_selected,
                                                     mock_prompt, mock_validate_available,
                                                     mock_get_folders):
        """Test compare launches menu with successful flow."""
        # Setup mocks
        mock_get_folders.return_value = ['launch1', 'launch2', 'launch3']
        mock_validate_available.return_value = True
        mock_prompt.return_value = {
            'launches': ['launch1', 'launch2'],
            'start_time': '10',
            'end_time': '100',
            'show_figures': True
        }
        mock_validate_selected.return_value = True
        
        # Call function
        result = compare_multiple_launches_menu()
        
        # Verify results
        assert result is True
        mock_get_folders.assert_called_once()
        mock_validate_available.assert_called_once_with(['launch1', 'launch2', 'launch3'])
        mock_prompt.assert_called_once_with(['launch1', 'launch2', 'launch3'])
        mock_validate_selected.assert_called_once_with(['launch1', 'launch2'])
        mock_execute.assert_called_once_with(['launch1', 'launch2'], '10', '100', True)
        mock_input.assert_called_once()
        # The function calls clear_screen twice - at the beginning and after user input
        assert mock_clear.call_count == 2
    
    @patch('ui.visualization_menu.get_launch_folders')
    @patch('ui.visualization_menu.validate_available_launches')
    def test_compare_launches_insufficient_available(self, mock_validate_available, 
                                                   mock_get_folders):
        """Test compare launches menu when insufficient folders are available."""
        # Setup mocks
        mock_get_folders.return_value = ['launch1']
        mock_validate_available.return_value = False
        
        # Call function
        result = compare_multiple_launches_menu()
        
        # Verify results
        assert result is True
        mock_get_folders.assert_called_once()
        mock_validate_available.assert_called_once_with(['launch1'])
    
    @patch('ui.visualization_menu.get_launch_folders')
    @patch('ui.visualization_menu.validate_available_launches')
    @patch('ui.visualization_menu.prompt_for_comparison_options')
    @patch('ui.visualization_menu.validate_selected_launches')
    def test_compare_launches_insufficient_selected(self, mock_validate_selected,
                                                 mock_prompt, mock_validate_available,
                                                 mock_get_folders):
        """Test compare launches menu when insufficient launches are selected."""
        # Setup mocks
        mock_get_folders.return_value = ['launch1', 'launch2', 'launch3']
        mock_validate_available.return_value = True
        mock_prompt.return_value = {
            'launches': ['launch1'],  # Only one selected
            'start_time': '10',
            'end_time': '100',
            'show_figures': True
        }
        mock_validate_selected.return_value = False
        
        # Call function
        result = compare_multiple_launches_menu()
        
        # Verify results
        assert result is True
        mock_get_folders.assert_called_once()
        mock_validate_available.assert_called_once_with(['launch1', 'launch2', 'launch3'])
        mock_prompt.assert_called_once_with(['launch1', 'launch2', 'launch3'])
        mock_validate_selected.assert_called_once_with(['launch1'])
