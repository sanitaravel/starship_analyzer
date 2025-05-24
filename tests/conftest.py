"""
Global pytest configuration.
"""
import pytest

def pytest_configure(config):
    """Configure pytest to show cleaner output."""
    # Register a custom marker for slow tests
    config.addinivalue_line("markers", 
                           "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    # Register ui marker for UI tests
    config.addinivalue_line("markers",
                           "ui: marks tests for UI components")

def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their module."""
    for item in items:
        # Add utility marker for all tests in utils module
        if "test_utils" in item.nodeid:
            item.add_marker(pytest.mark.utils)
        # Add ui marker for all tests in UI module
        if "test_ui" in item.nodeid:
            item.add_marker(pytest.mark.ui)

# Track the current module and class for grouping output
_current_module = None
_current_class = None

def pytest_runtest_protocol(item, nextitem):
    """
    Hook implementation that runs before each test.
    Used to organize tests visually by module and class.
    """
    global _current_module, _current_class
    
    # Extract module name
    module_name = item.module.__name__
    
    # Format module name for display (remove 'tests.' prefix if present)
    display_module = module_name
    if display_module.startswith('tests.'):
        display_module = display_module[6:]
    
    # Replace 'test_' with 'Testing ' for better readability
    if "test_" in display_module:
        display_module = display_module.replace("test_", "Testing ")
    
    # Print module header if we're in a new module
    if module_name != _current_module:
        # Add decorative separator before new module (except for the very first one)
        if _current_module is not None:
            print("\n" + "─" * 80)
        
        print(f"\n\033[1;36m▶ {display_module}\033[0m")
        print("─" * 40)  # Shorter line under module name
        _current_module = module_name
        _current_class = None  # Reset class tracking when module changes
    
    # Extract class name if available
    class_name = None
    if hasattr(item, 'cls') and item.cls:
        class_name = item.cls.__name__
        
        # Format class name for display
        display_class = class_name.replace("Test", "Testing ")
        
        # Print class header if we're in a new class
        if class_name != _current_class:
            # Add spacing before class name (except for the first class in a module)
            if _current_class is not None:
                print("")
            
            print(f"  \033[1;33m→ {display_class}\033[0m")
            print("  " + "┄" * 30)  # Dotted line under class name
            _current_class = class_name
    
    # Return None to let pytest handle the actual test run
    return None

def pytest_runtest_logreport(report):
    """
    Hook to customize the appearance of test results.
    """
    if report.when == "call":
        # Skip further processing if no class (to preserve layout)
        if _current_class is None:
            return
        
        # Add indent for better test hierarchy visualization
        if report.outcome == "passed":
            print(f"    ✓ {report.nodeid.split('::')[-1]}")
        elif report.outcome == "failed":
            print(f"    ✗ {report.nodeid.split('::')[-1]}")
        elif report.outcome == "skipped":
            print(f"    ○ {report.nodeid.split('::')[-1]}")

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information at the end of test session."""
    if terminalreporter.stats.get('passed', None):
        terminalreporter.write_sep("=", "Starship Analyzer Test Summary", bold=True)
        
        # Count tests by category
        utils_tests = 0
        ui_tests = 0
        for item in terminalreporter.stats.get('passed', []):
            if hasattr(item, 'keywords'):
                if 'utils' in item.keywords:
                    utils_tests += 1
                if 'ui' in item.keywords:
                    ui_tests += 1
                
        terminalreporter.write_line(f"Utility Tests: {utils_tests} passed")
        terminalreporter.write_line(f"UI Tests: {ui_tests} passed")
