import pytest
import pandas as pd
import numpy as np
from definition_8830b5564b5a47b9a13d79e146bc1caa import validate_data

@pytest.mark.parametrize("df_input, expected_columns_input, expected_return_value, expected_output_part", [
    # Test Case 1: Valid DataFrame - Expected functionality
    (
        pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'location': ['Urban', 'Suburban', 'Rural'],
            'gender': ['Male', 'Female', 'Male'],
            'loan_approval': [1, 0, 1]
        }),
        ['age', 'income', 'location', 'gender', 'loan_approval'],
        True,
        "Summary statistics for numeric columns:"
    ),
    # Test Case 2: Missing Column - Edge case (Structural error)
    (
        pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'location': ['Urban', 'Suburban', 'Rural'],
            'loan_approval': [1, 0, 1] # 'gender' column is missing
        }),
        ['age', 'income', 'location', 'gender', 'loan_approval'],
        False,
        "Validation Error: Missing expected columns"
    ),
    # Test Case 3: Incorrect Data Type for 'age' - Edge case (Type error)
    (
        pd.DataFrame({
            'age': [25.0, 30.0, 35.0], # float64, not int64
            'income': [50000, 60000, 70000],
            'location': ['Urban', 'Suburban', 'Rural'],
            'gender': ['Male', 'Female', 'Male'],
            'loan_approval': [1, 0, 1]
        }),
        ['age', 'income', 'location', 'gender', 'loan_approval'],
        False,
        "Type Error: Incorrect data type for 'age' column"
    ),
    # Test Case 4: Missing Values in Critical Columns ('age', 'income', 'gender') - Edge case (Data quality error)
    (
        pd.DataFrame({
            'age': [25, np.nan, 35], # NaN in age
            'income': [50000, 60000, 70000],
            'location': ['Urban', 'Suburban', 'Rural'],
            'gender': ['Male', 'Female', 'Male'],
            'loan_approval': [1, 0, 1]
        }),
        ['age', 'income', 'location', 'gender', 'loan_approval'],
        False,
        "Validation Error: Missing values found in critical columns"
    ),
    # Test Case 5: Empty DataFrame - Edge case (Extreme structural error)
    (
        pd.DataFrame(), # An empty DataFrame has no columns
        ['age', 'income', 'location', 'gender', 'loan_approval'],
        False,
        "Validation Error: Missing expected columns"
    ),
])
def test_validate_data(df_input, expected_columns_input, expected_return_value, expected_output_part, capsys):
    # Call the function
    result = validate_data(df_input, expected_columns_input)

    # Assert the return value
    assert result == expected_return_value

    # Capture stdout and assert part of the expected message
    captured = capsys.readouterr()
    assert expected_output_part in captured.out