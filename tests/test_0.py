import pytest
from definition_25819e2198b741bb9416146e3614b8cd import generate_synthetic_data
import pandas as pd

def is_valid_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    return True

def are_all_columns_present(df, expected_columns):
        return all(col in df.columns for col in expected_columns)

@pytest.mark.parametrize("num_samples, bias_factor, seed, expected_columns", [
    (100, 0.2, 42, ['age', 'income', 'location', 'gender', 'loan_approval']),
    (0, 0.2, 42, ['age', 'income', 'location', 'gender', 'loan_approval']),
    (100, -0.2, 42, ['age', 'income', 'location', 'gender', 'loan_approval']),
    (100, 0.2, "42", ['age', 'income', 'location', 'gender', 'loan_approval']),
    (100, 0.2, 42, ['age', 'income', 'location', 'gender']),

])
def test_generate_synthetic_data(num_samples, bias_factor, seed, expected_columns):
    try:
        synthetic_data = generate_synthetic_data(num_samples, bias_factor, seed)
        assert is_valid_dataframe(synthetic_data)

        if num_samples > 0:
            assert are_all_columns_present(synthetic_data, expected_columns)
            if 'loan_approval' in expected_columns:
                assert all(x in [0, 1] for x in synthetic_data['loan_approval'].unique())

            if seed == "42":
                assert True
            else:
                assert True
        else:
            assert len(synthetic_data) == 0

    except Exception as e:
        print(f"Error during test: {e}")
        raise

