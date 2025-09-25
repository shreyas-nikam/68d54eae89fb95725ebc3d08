import pytest
import pandas as pd
import numpy as np
from definition_cc1852ee51de4ab9b6a074f9a1cf1077 import statistical_parity_difference

@pytest.mark.parametrize(
    "df, group_col, outcome_col, privileged_group, expected",
    [
        # Test Case 1: Standard scenario, positive Statistical Parity Difference (SPD)
        # Privileged group ('Male') has a higher proportion of favorable outcomes (1/2 vs 0/2).
        (
            pd.DataFrame({
                'gender': ['Male', 'Male', 'Female', 'Female'],
                'loan_approval': [1, 0, 0, 0]
            }),
            'gender', 'loan_approval', 'Male', 0.5
        ),
        # Test Case 2: Standard scenario, negative SPD
        # Privileged group ('Male') has a lower proportion of favorable outcomes (0/2 vs 1/2).
        (
            pd.DataFrame({
                'gender': ['Male', 'Male', 'Female', 'Female'],
                'loan_approval': [0, 0, 1, 0]
            }),
            'gender', 'loan_approval', 'Male', -0.5
        ),
        # Test Case 3: Zero SPD
        # Both groups have the same proportion of favorable outcomes.
        (
            pd.DataFrame({
                'gender': ['Male', 'Male', 'Female', 'Female'],
                'loan_approval': [1, 0, 1, 0]
            }),
            'gender', 'loan_approval', 'Male', 0.0
        ),
        # Test Case 4: Invalid column name (edge case for input validation)
        # Expects a KeyError if 'group_col' or 'outcome_col' does not exist.
        (
            pd.DataFrame({
                'sex': ['Male', 'Female'],
                'result': [1, 0]
            }),
            'gender_col', 'result', 'Male', KeyError
        ),
        # Test Case 5: Privileged group absent in data (edge case)
        # If the privileged group is not present, its mean outcome will be NaN.
        # This propagates to the SPD calculation.
        (
            pd.DataFrame({
                'gender': ['Female', 'Female'],
                'loan_approval': [1, 0]
            }),
            'gender', 'loan_approval', 'Male', 'NAN_EXPECTED' # Custom marker for expected NaN
        ),
    ]
)
def test_statistical_parity_difference(df, group_col, outcome_col, privileged_group, expected):
    if expected == 'NAN_EXPECTED':
        # Special handling for expected NaN return values
        result = statistical_parity_difference(df, group_col, outcome_col, privileged_group)
        assert pd.isna(result)
        return

    try:
        result = statistical_parity_difference(df, group_col, outcome_col, privileged_group)
        # Using pytest.approx for floating-point comparisons
        assert result == pytest.approx(expected)
    except Exception as e:
        # For expected exceptions, check if the raised exception is of the expected type
        assert isinstance(e, expected)