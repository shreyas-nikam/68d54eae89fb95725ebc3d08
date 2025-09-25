import pytest
import pandas as pd
import numpy as np
import math
from definition_ffee8267027c4218ac3d938e002a6abf import equal_opportunity_difference

@pytest.mark.parametrize("df_data, group_col, outcome_col, privileged_group, expected_result", [
    # Test Case 1: Both groups have favorable outcomes (1), expecting a difference of 0.0
    # P(outcome=1 | A, actual=1) = 1.0 (mean of [1])
    # P(outcome=1 | B, actual=1) = 1.0 (mean of [1])
    # EOD = 1.0 - 1.0 = 0.0
    (pd.DataFrame({'group_col': ['A', 'A', 'B', 'B'], 'outcome_col': [1, 0, 1, 0]}), 'group_col', 'outcome_col', 'A', 0.0),

    # Test Case 2: Only the privileged group (A) has favorable outcomes (1), expecting NaN
    # P(outcome=1 | A, actual=1) = 1.0 (mean of [1])
    # P(outcome=1 | B, actual=1) = NaN (mean of empty series)
    # EOD = 1.0 - NaN = NaN
    (pd.DataFrame({'group_col': ['A', 'A', 'B', 'B'], 'outcome_col': [1, 0, 0, 0]}), 'group_col', 'outcome_col', 'A', np.nan),

    # Test Case 3: Only the unprivileged group (B) has favorable outcomes (1), expecting NaN
    # P(outcome=1 | A, actual=1) = NaN (mean of empty series)
    # P(outcome=1 | B, actual=1) = 1.0 (mean of [1])
    # EOD = NaN - 1.0 = NaN
    (pd.DataFrame({'group_col': ['A', 'A', 'B', 'B'], 'outcome_col': [0, 0, 1, 0]}), 'group_col', 'outcome_col', 'A', np.nan),

    # Test Case 4: Neither group has favorable outcomes (1), expecting NaN
    # P(outcome=1 | A, actual=1) = NaN (mean of empty series)
    # P(outcome=1 | B, actual=1) = NaN (mean of empty series)
    # EOD = NaN - NaN = NaN
    (pd.DataFrame({'group_col': ['A', 'A', 'B', 'B'], 'outcome_col': [0, 0, 0, 0]}), 'group_col', 'outcome_col', 'A', np.nan),

    # Test Case 5: Missing a required column (group_col), expecting KeyError
    (pd.DataFrame({'incorrect_group_col': ['A', 'B'], 'outcome_col': [1, 1]}), 'group_col', 'outcome_col', 'A', KeyError),
])
def test_equal_opportunity_difference(df_data, group_col, outcome_col, privileged_group, expected_result):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            equal_opportunity_difference(df_data, group_col, outcome_col, privileged_group)
    else:
        result = equal_opportunity_difference(df_data, group_col, outcome_col, privileged_group)
        if math.isnan(expected_result):
            assert math.isnan(result)
        else:
            assert result == expected_result