import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class InteractivePlotViewer:
    """
    Interactive viewer for displaying multiple matplotlib figures in a single window.
    Allows switching between figures using a dropdown menu.
    """
    def __init__(self, title="Interactive Plot Viewer"):
        self.title = title
        self.figures = []  # List of (figure, title) tuples
        self.fig_titles = []  # List of figure titles for dropdown
        self.current_figure_index = 0
        self.root = None
        self.frame = None
        self.canvas = None
        self.toolbar = None
        self.dropdown = None
        self.is_showing = False
        logger.debug(f"Initialized InteractivePlotViewer with title: {title}")
    
    def add_figure(self, fig, title):
        """Add a matplotlib figure to the viewer."""
        self.figures.append((fig, title))
        self.fig_titles.append(title)
        logger.debug(f"Added figure: {title} - Total figures: {len(self.figures)}")
        return self
    
    def _setup_ui(self):
        """Set up the tkinter UI components."""
        if not self.figures:
            logger.warning("No figures to display!")
            return False
            
        # Create a new Tk root if one doesn't exist
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1200x800")  # Default size
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create controls frame at the top
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create sorted list of titles and corresponding figures
        sorted_pairs = sorted(zip(self.fig_titles, self.figures), key=lambda x: x[0])
        self.fig_titles = [title for title, _ in sorted_pairs]
        self.figures = [fig for _, fig in sorted_pairs]
        
        # Create dropdown for figure selection
        ttk.Label(controls_frame, text="Select Figure:").pack(side=tk.LEFT, padx=5, pady=5)
        self.dropdown_var = tk.StringVar(self.root)
        self.dropdown_var.set(self.fig_titles[0] if self.fig_titles else "")  # Set default value
        self.dropdown = ttk.Combobox(
            controls_frame, 
            textvariable=self.dropdown_var,
            values=self.fig_titles,
            width=60
        )
        self.dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self._on_dropdown_change)
        
        # Add close button
        close_button = ttk.Button(controls_frame, text="Close", command=self._on_close)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create frame for the figure
        self.frame = ttk.Frame(main_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Display the first figure
        if self.figures:
            self._display_figure(0)
            
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        return True
    
    def _display_figure(self, index):
        """Display the figure at the given index."""
        if not 0 <= index < len(self.figures):
            logger.error(f"Invalid figure index: {index}")
            return
            
        # Clear existing figure display if any
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()
            
        # Get the figure and display it
        fig, title = self.figures[index]
        self.current_figure_index = index
        
        # Embed the matplotlib figure in the tkinter window
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add the matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        logger.debug(f"Displayed figure: {title}")
    
    def _on_dropdown_change(self, event):
        """Handle dropdown selection changes."""
        selected_title = self.dropdown_var.get()
        for i, title in enumerate(self.fig_titles):
            if title == selected_title:
                self._display_figure(i)
                break
    
    def _on_close(self):
        """Clean up resources and destroy the window when closing."""
        logger.debug("Closing interactive viewer window")
        self.is_showing = False
        
        # Clean up matplotlib figures to prevent memory leaks
        for fig, _ in self.figures:
            plt.close(fig)
            
        # Clear references
        self.figures = []
        self.fig_titles = []
        
        # Destroy Tkinter window
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None
    
    def show(self):
        """Display the interactive viewer with all added figures."""
        if len(self.figures) == 0:
            logger.warning("No figures to display!")
            return
            
        logger.info(f"Showing interactive viewer with {len(self.figures)} figures")
        if self._setup_ui():
            self.is_showing = True
            self.root.mainloop()
            logger.debug("Interactive viewer mainloop exited")


def show_plots_interactively(title="Interactive Plot Viewer"):
    """
    Create and return an interactive plot viewer.
    
    Args:
        title (str): The title for the interactive viewer window
        
    Returns:
        InteractivePlotViewer: An interactive plot viewer instance
    """
    logger.info(f"Creating interactive plot viewer: {title}")
    return InteractivePlotViewer(title)
