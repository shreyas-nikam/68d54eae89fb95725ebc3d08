import pytest
import pandas as pd
from definition_dbd90194e49d4f6b92a5549b93f53844 import equal_opportunity_difference

def test_equal_opportunity_difference_basic():
    df = pd.DataFrame({
        'gender': ['Male', 'Male', 'Female', 'Female'],
        'loan_approval': [1, 0, 1, 0]
    })
    result = equal_opportunity_difference(df, 'gender', 'loan_approval', 'Male')
    assert result == 0.0  # Equal opportunity for both groups

def test_equal_opportunity_difference_privileged_group():
    df = pd.DataFrame({
        'gender': ['Male', 'Male', 'Female', 'Female'],
        'loan_approval': [1, 1, 0, 0]
    })
    result = equal_opportunity_difference(df, 'gender', 'loan_approval', 'Male')
    assert result == 0.5  # Males have a higher approval rate

def test_equal_opportunity_difference_no_approval():
    df = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'loan_approval': [0, 0, 0, 0]
    })
    result = equal_opportunity_difference(df, 'gender', 'loan_approval', 'Male')
    assert result == 0.0  # No approvals for either group

def test_equal_opportunity_difference_empty_dataframe():
    df = pd.DataFrame(columns=['gender', 'loan_approval'])
    result = equal_opportunity_difference(df, 'gender', 'loan_approval', 'Male')
    assert result == 0.0  # No data should return 0 difference

def test_equal_opportunity_difference_single_group():
    df = pd.DataFrame({
        'gender': ['Male', 'Male', 'Male'],
        'loan_approval': [1, 1, 1]
    })
    result = equal_opportunity_difference(df, 'gender', 'loan_approval', 'Male')
    assert result == 0.0  # Only one group present, no difference