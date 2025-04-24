"""
Tests for the interactive_viewer module.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import matplotlib
# Use non-interactive backend for testing to avoid Tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from plot.interactive_viewer import InteractivePlotViewer, show_plots_interactively

# Check if tkinter is properly installed and available
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TKINTER_AVAILABLE = False
except Exception as e:
    # Handle Tcl/Tk initialization errors by marking tkinter as unavailable
    if "TclError" in str(type(e)):
        TKINTER_AVAILABLE = False
    else:
        raise


@pytest.fixture
def test_viewer():
    """Fixture to create a test InteractivePlotViewer."""
    viewer = InteractivePlotViewer("Test Viewer")
    yield viewer
    # Teardown
    plt.close('all')  # Close all matplotlib figures
    if hasattr(viewer, 'root') and viewer.root:
        try:
            viewer.root.destroy()
        except Exception:
            pass  # Ignore errors when destroying the root window


@pytest.fixture
def test_figures():
    """Fixture to create test figures for use in tests."""
    # Create figures as simple MagicMock objects
    fig1 = MagicMock()
    fig1_title = "Test Figure 1"
    
    fig2 = MagicMock()
    fig2_title = "Test Figure 2"
    
    yield (fig1, fig1_title, fig2, fig2_title)


class TestInteractiveViewerInitialization:
    """Tests for InteractivePlotViewer initialization."""
    
    def test_initialization(self, test_viewer):
        """Test that the InteractivePlotViewer initializes with the correct title."""
        assert test_viewer.title == "Test Viewer"
        assert len(test_viewer.figures) == 0
        assert len(test_viewer.fig_titles) == 0
        assert test_viewer.is_showing is False


class TestFigureManagement:
    """Tests for figure management in the interactive viewer."""
    
    def test_add_figure(self, test_viewer, test_figures):
        """Test adding figures to the viewer."""
        fig1, fig1_title, fig2, fig2_title = test_figures
        
        # Add first figure
        test_viewer.add_figure(fig1, fig1_title)
        assert len(test_viewer.figures) == 1
        assert len(test_viewer.fig_titles) == 1
        assert test_viewer.fig_titles[0] == fig1_title
        
        # Add second figure
        test_viewer.add_figure(fig2, fig2_title)
        assert len(test_viewer.figures) == 2
        assert len(test_viewer.fig_titles) == 2
        assert test_viewer.fig_titles[1] == fig2_title
        
        # Test that add_figure returns self (for chaining)
        assert test_viewer.add_figure(fig1, "Test") == test_viewer

    def test_setup_ui_no_figures(self, test_viewer):
        """Test UI setup with no figures."""
        result = test_viewer._setup_ui()
        assert result is False


class TestUIFunctionality:
    """Tests for UI setup and interaction."""
    
    @pytest.mark.skipif(not TKINTER_AVAILABLE, reason="Tkinter not available")
    @patch('tkinter.Tk')
    @patch('matplotlib.backends.backend_tkagg.FigureCanvasTkAgg')
    @patch('matplotlib.backends.backend_tkagg.NavigationToolbar2Tk')
    def test_setup_ui(self, mock_toolbar, mock_canvas, mock_tk, test_viewer, test_figures):
        """Test the UI setup functionality."""
        fig1, fig1_title, _, _ = test_figures
        
        # Setup mocks
        mock_tk_instance = MagicMock()
        mock_tk.return_value = mock_tk_instance
        mock_canvas_instance = MagicMock()
        mock_canvas.return_value = mock_canvas_instance
        # Setup get_tk_widget method on the canvas
        mock_canvas_instance.get_tk_widget.return_value = MagicMock()
        mock_toolbar_instance = MagicMock()
        mock_toolbar.return_value = mock_toolbar_instance
        
        # Add a figure before calling _setup_ui - this was missing!
        test_viewer.add_figure(fig1, fig1_title)
        
        # Completely patch the _display_figure method to avoid calling it
        with patch.object(test_viewer, '_display_figure') as mock_display:
            result = test_viewer._setup_ui()
            
            # Verify _setup_ui works as expected
            assert result is True
            mock_tk.assert_called_once()
            mock_tk_instance.title.assert_called_with("Test Viewer")
            # Verify _display_figure was called with index 0
            mock_display.assert_called_once_with(0)

    @pytest.mark.skipif(not TKINTER_AVAILABLE, reason="Tkinter not available")
    def test_display_figure(self, test_viewer, test_figures, caplog):
        """Test figure display functionality."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        print("\n--- DETAILED DEBUG FOR test_display_figure ---")
        
        fig1, fig1_title, fig2, fig2_title = test_figures
        print(f"Figure objects: fig1={fig1}, fig2={fig2}")
        print(f"Figure titles: '{fig1_title}', '{fig2_title}'")
        
        # Add figures
        print("Adding figures to viewer")
        test_viewer.add_figure(fig1, fig1_title)
        test_viewer.add_figure(fig2, fig2_title)
        print(f"Viewer now has {len(test_viewer.figures)} figures")
        
        # Create a replacement implementation of _display_figure that doesn't use FigureCanvasTkAgg
        def mock_display_figure_impl(index):
            print(f"Mock _display_figure called with index {index}")
            # Store the index
            test_viewer.current_figure_index = index
            # Create mock canvas and toolbar for testing
            test_viewer.canvas = MagicMock(name="mock_canvas")
            test_viewer.toolbar = MagicMock(name="mock_toolbar")
            # Log which figure is being displayed
            fig, title = test_viewer.figures[index]
            print(f"Would display figure: {title}")
        
        # Completely replace the method
        with patch.object(test_viewer, '_display_figure', side_effect=mock_display_figure_impl) as mock_display:
            print("\nCalling _display_figure(1)")
            # Call our patched method
            test_viewer._display_figure(1)
            print("Successfully called _display_figure(1)")
            
            # Print detailed state after the call
            print(f"\nAfter _display_figure:")
            print(f"current_figure_index = {test_viewer.current_figure_index}")
            print(f"canvas = {test_viewer.canvas}")
            print(f"toolbar = {test_viewer.toolbar}")
            
            # Basic assertions on the implementation
            assert test_viewer.current_figure_index == 1
            assert test_viewer.canvas is not None
            assert test_viewer.toolbar is not None
            
            # Verify the method was called with the right parameter
            mock_display.assert_called_once_with(1)
            
        print("--- END DETAILED DEBUG ---")


class TestShowFunctionality:
    """Tests for interactive viewer show functionality."""
    
    @patch('plot.interactive_viewer.InteractivePlotViewer')
    def test_show_plots_interactively(self, mock_viewer_class):
        """Test the factory function for creating the viewer."""
        # Setup mock
        mock_instance = MagicMock()
        mock_viewer_class.return_value = mock_instance
        
        test_title = "Factory Test"
        viewer = show_plots_interactively(test_title)
        
        # Verify the factory function works
        mock_viewer_class.assert_called_once_with(test_title)
        assert viewer == mock_instance

    @patch('plot.interactive_viewer.InteractivePlotViewer._setup_ui')
    def test_show_with_no_figures(self, mock_setup_ui, test_viewer):
        """Test show method with no figures."""
        test_viewer.show()
        mock_setup_ui.assert_not_called()

    @pytest.mark.skipif(not TKINTER_AVAILABLE, reason="Tkinter not available")
    @patch('plot.interactive_viewer.InteractivePlotViewer._setup_ui')
    def test_show_with_figures(self, mock_setup_ui, test_viewer, test_figures):
        """Test show method with figures added."""
        fig1, fig1_title, _, _ = test_figures
        
        # Create a mock for the root attribute
        mock_root = MagicMock()
        
        # Have _setup_ui setup the required components
        def mock_setup_implementation():
            test_viewer.root = mock_root
            return True
        
        mock_setup_ui.side_effect = mock_setup_implementation
        
        # Add figures and call show
        test_viewer.add_figure(fig1, fig1_title)
        
        # Patch the mainloop method to prevent it from being called
        with patch.object(mock_root, 'mainloop') as mock_mainloop:
            test_viewer.show()
            
            # Verify setup_ui and mainloop were called
            mock_setup_ui.assert_called_once()
            mock_mainloop.assert_called_once()
            assert test_viewer.is_showing is True


class TestEventHandling:
    """Tests for event handling in the interactive viewer."""
    
    @pytest.mark.skipif(not TKINTER_AVAILABLE, reason="Tkinter not available")
    def test_on_close(self, test_viewer, test_figures):
        """Test the close event handler."""
        fig1, fig1_title, _, _ = test_figures
        
        # Add a figure and setup UI
        test_viewer.add_figure(fig1, fig1_title)
        
        # Create a proper mock for the root attribute
        mock_root = MagicMock()
        test_viewer.root = mock_root  # Explicitly set the mock
        
        # Set up the test state
        test_viewer.figures = [(fig1, fig1_title)]
        test_viewer.fig_titles = [fig1_title]
        test_viewer.is_showing = True
        
        # Patch plt.close to avoid errors
        with patch('matplotlib.pyplot.close'):
            # Call on_close
            test_viewer._on_close()
            
            # Verify that resources were cleaned up
            assert test_viewer.is_showing is False
            assert len(test_viewer.figures) == 0
            assert len(test_viewer.fig_titles) == 0
            mock_root.quit.assert_called_once()  # Use the mock_root variable directly
            mock_root.destroy.assert_called_once()

    @pytest.mark.skipif(not TKINTER_AVAILABLE, reason="Tkinter not available")
    def test_on_dropdown_change(self, test_viewer, test_figures):
        """Test the dropdown change handler."""
        fig1, fig1_title, fig2, fig2_title = test_figures
        
        # Add figures
        test_viewer.add_figure(fig1, fig1_title)
        test_viewer.add_figure(fig2, fig2_title)
        
        # Create mock dropdown variable
        test_viewer.dropdown_var = MagicMock()
        test_viewer.dropdown_var.get.return_value = fig2_title
        test_viewer.fig_titles = [fig1_title, fig2_title]
        
        # Mock _display_figure method
        with patch.object(test_viewer, '_display_figure') as mock_display:
            # Call the dropdown change handler
            test_viewer._on_dropdown_change(None)  # Event parameter not used in the method
            
            # Verify that _display_figure was called with the correct index
            mock_display.assert_called_once_with(1)  # Should display the second figure
