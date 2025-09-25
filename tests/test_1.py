import pytest
import pandas as pd
from definition_1e1ce00c108644588b91e6cb923d7240 import validate_data

@pytest.fixture
def sample_dataframe():
    data = {'age': [30, 40, 25],
            'income': [50000, 60000, 45000],
            'location': ['Urban', 'Suburban', 'Rural'],
            'gender': ['Male', 'Female', 'Male'],
            'loan_approval': [1, 0, 1]}
    return pd.DataFrame(data)

def test_validate_data_valid(sample_dataframe):
    expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
    assert validate_data(sample_dataframe.copy(), expected_columns) == True

def test_validate_data_missing_column(sample_dataframe):
    expected_columns = ['age', 'income', 'location', 'gender']
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_data(sample_dataframe.copy(), expected_columns)

def test_validate_data_incorrect_dtype(sample_dataframe):
    sample_dataframe['age'] = sample_dataframe['age'].astype(str)
    expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
    with pytest.raises(TypeError, match="Incorrect data type for 'age' column"):
        validate_data(sample_dataframe.copy(), expected_columns)

def test_validate_data_missing_values(sample_dataframe):
    sample_dataframe.loc[0, 'income'] = None
    expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
    with pytest.raises(ValueError, match="Missing values found in critical columns"):
        validate_data(sample_dataframe.copy(), expected_columns)