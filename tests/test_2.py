import pytest
import pandas as pd
from definition_77c574c37f0f4d7196f5230c196997e2 import statistical_parity_difference

@pytest.fixture
def sample_dataframe():
    data = {'group': [0, 0, 1, 1, 0, 1], 'outcome': [1, 0, 1, 0, 1, 1]}
    return pd.DataFrame(data)

def test_statistical_parity_difference_basic(sample_dataframe):
    spd = statistical_parity_difference(sample_dataframe, 'group', 'outcome', 1)
    assert spd == (2/3) - (2/3)

def test_statistical_parity_difference_no_privileged(sample_dataframe):
    df = sample_dataframe[sample_dataframe['group'] != 1]
    spd = statistical_parity_difference(df, 'group', 'outcome', 1)
    assert spd == 0 - (2/3)

def test_statistical_parity_difference_no_unprivileged(sample_dataframe):
    df = sample_dataframe[sample_dataframe['group'] != 0]
    spd = statistical_parity_difference(df, 'group', 'outcome', 1)
    assert spd == (2/3) - 0

def test_statistical_parity_difference_empty_dataframe():
    df = pd.DataFrame({'group': [], 'outcome': []})
    spd = statistical_parity_difference(df, 'group', 'outcome', 1)
    assert spd == 0

def test_statistical_parity_difference_all_same_outcome(sample_dataframe):
    sample_dataframe['outcome'] = 1
    spd = statistical_parity_difference(sample_dataframe, 'group', 'outcome', 1)
    assert spd == 1 - 1
