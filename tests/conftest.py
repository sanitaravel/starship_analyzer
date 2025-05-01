"""
Global pytest configuration.
"""
import pytest

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped by default)"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test that requires no timeout"
    )
    # Add missing marker registrations
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "utils: mark test as a utility test"
    )
    config.addinivalue_line(
        "markers", "ui: mark test as a UI test"
    )

def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their module."""
    for item in items:
        # Add utility marker for all tests in utils module
        if "test_utils" in item.nodeid:
            item.add_marker(pytest.mark.utils)
        # Add ui marker for all tests in UI module
        if "test_ui" in item.nodeid:
            item.add_marker(pytest.mark.ui)

        # Apply custom timeouts from markers if specified
        timeout_marker = item.get_closest_marker("timeout")
        if timeout_marker and timeout_marker.args:
            item.add_marker(pytest.mark.timeout(timeout_marker.args[0]))

@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    """Setup for test runs - disable timeout for performance tests."""
    if item.get_closest_marker("performance"):
        # More robust way to disable timeout for performance tests
        if hasattr(item.config, "option"):
            # Disable pytest-timeout if it's being used
            if hasattr(item.config.option, "timeout"):
                item.config.option.timeout = 0  # 0 means no timeout
                
            # Also set the timeout_method to None if available
            if hasattr(item.config.option, "timeout_method"):
                item.config.option.timeout_method = None
