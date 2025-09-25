import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# definition_12f8fc717d614a31a2e2d6cf56f765bc block - DO NOT REPLACE or REMOVE
# Assuming all helper functions (generate_synthetic_data, validate_data, etc.)
# that `interactive_analysis` internally calls are part of the same module.
from definition_12f8fc717d614a31a2e2d6cf56f765bc import interactive_analysis
# END definition_12f8fc717d614a31a2e2d6cf56f765bc block

# Fixture to mock matplotlib.pyplot.show() to prevent plots from appearing during tests.
# This makes the tests non-interactive and prevents them from blocking execution.
@pytest.fixture(autouse=True)
def mock_plot_show(mocker):
    """Mocks plt.show() to prevent plots from being displayed during tests."""
    mocker.patch('matplotlib.pyplot.show')

# Fixture to mock the built-in print function to avoid polluting test output
# and allow for potential assertion on print calls if needed (though not done here).
@pytest.fixture(autouse=True)
def mock_print(mocker):
    """Mocks the built-in print function."""
    mocker.patch('builtins.print')

@pytest.mark.parametrize("bias_factor, reweighting_factor, expected_exception", [
    # Test Case 1: Standard valid input (using values from the notebook's interactive example).
    (0.3, 0.2, None),
    # Test Case 2: Edge case - No initial bias (bias_factor=0.0).
    # Ensures the pipeline functions correctly when synthetic data has minimal or no bias.
    (0.0, 0.2, None),
    # Test Case 3: Edge case - No reweighting applied (reweighting_factor=0.0).
    # Ensures the pipeline functions correctly when mitigation step is effectively skipped.
    (0.3, 0.0, None),
    # Test Case 4: Edge case - Maximum bias and maximum reweighting allowed by the sliders in the notebook specification.
    (0.5, 1.0, None),
    # Test Case 5: Invalid input type for bias_factor.
    # The internal functions like `generate_synthetic_data` expect numeric types.
    ('invalid_type', 0.2, TypeError),
])
def test_interactive_analysis(bias_factor, reweighting_factor, expected_exception):
    """
    Tests the interactive_analysis function with various bias and reweighting factors,
    including valid inputs and an invalid type edge case.
    """
    if expected_exception:
        # If an exception is expected, assert that the correct exception type is raised.
        with pytest.raises(expected_exception):
            interactive_analysis(bias_factor, reweighting_factor)
    else:
        # For valid inputs, assert that the function runs without raising any unexpected exception.
        # Since `interactive_analysis` has no return value (it prints and plots),
        # successful execution without errors is the primary assertion.
        try:
            interactive_analysis(bias_factor, reweighting_factor)
        except Exception as e:
            pytest.fail(f"interactive_analysis raised an unexpected exception for valid inputs: {e}")