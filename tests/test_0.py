import pytest
import pandas as pd
import numpy as np
from definition_d7f8178d8e294ee586ff791ae701aea8 import generate_synthetic_data

# Test 1: Basic functionality and DataFrame structure
def test_generate_synthetic_data_basic_structure():
    num_samples = 100
    bias_factor = 0.3
    seed = 42
    df = generate_synthetic_data(num_samples, bias_factor, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_samples
    expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
    assert list(df.columns) == expected_columns

    # Check data types as per the internal implementation details
    assert df['age'].dtype == 'int64'
    assert df['income'].dtype == 'float64'
    assert df['location'].dtype == 'object'
    assert df['gender'].dtype == 'object'
    assert df['loan_approval'].dtype == 'float64'
    assert df['loan_approval'].isin([0.0, 1.0]).all() # Check binary nature

# Test 2: Edge case - num_samples = 0
def test_generate_synthetic_data_zero_samples():
    num_samples = 0
    bias_factor = 0.3
    seed = 42
    df = generate_synthetic_data(num_samples, bias_factor, seed)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
    assert list(df.columns) == expected_columns
    # For an empty DataFrame, column dtypes might be inferred as object.
    # The primary check here is that it's a DataFrame and has the correct columns.

# Test 3: Reproducibility with the same seed
def test_generate_synthetic_data_reproducibility():
    num_samples = 50
    bias_factor = 0.2
    seed = 123

    df1 = generate_synthetic_data(num_samples, bias_factor, seed)
    df2 = generate_synthetic_data(num_samples, bias_factor, seed)

    pd.testing.assert_frame_equal(df1, df2)

# Test 4: Qualitative check for bias_factor effect
def test_generate_synthetic_data_bias_effect():
    num_samples = 1000 # Larger sample size for more reliable statistics
    seed = 42

    # Test with a significant bias factor
    bias_factor_high = 0.5
    df_high_bias = generate_synthetic_data(num_samples, bias_factor_high, seed)
    male_approval_high_bias = df_high_bias[df_high_bias['gender'] == 'Male']['loan_approval'].mean()
    female_approval_high_bias = df_high_bias[df_high_bias['gender'] == 'Female']['loan_approval'].mean()
    
    # Assert male approval is higher than female approval when bias_factor is positive
    assert male_approval_high_bias > female_approval_high_bias

    # Test with zero bias factor
    bias_factor_low = 0.0
    df_low_bias = generate_synthetic_data(num_samples, bias_factor_low, seed)
    male_approval_low_bias = df_low_bias[df_low_bias['gender'] == 'Male']['loan_approval'].mean()
    female_approval_low_bias = df_low_bias[df_low_bias['gender'] == 'Female']['loan_approval'].mean()

    # The difference in approval rates should be greater with a higher bias_factor
    diff_high_bias = male_approval_high_bias - female_approval_high_bias
    diff_low_bias = male_approval_low_bias - female_approval_low_bias
    assert diff_high_bias > diff_low_bias

# Test 5: Invalid input types for parameters
@pytest.mark.parametrize("num_samples, bias_factor, seed, expected_exception", [
    ("invalid", 0.3, 42, TypeError), # num_samples not int
    (100, "invalid", 42, TypeError), # bias_factor not float (str)
    (100, 0.3, "invalid", TypeError), # seed not int (str)
    (100.5, 0.3, 42, TypeError), # num_samples float, but int expected by numpy
    (100, 0, 42.5, TypeError), # seed float, but int expected by numpy
    (100, 5, 42, TypeError), # bias_factor int, but float expected (though typically auto-converts, testing explicit type violation is good)
])
def test_generate_synthetic_data_invalid_types(num_samples, bias_factor, seed, expected_exception):
    with pytest.raises(expected_exception):
        generate_synthetic_data(num_samples, bias_factor, seed)