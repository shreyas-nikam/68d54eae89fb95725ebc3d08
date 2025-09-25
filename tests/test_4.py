import pytest
import pandas as pd
from definition_c1eaeba5d8724729b2f65af1b63e7cd8 import reweight

@pytest.fixture
def sample_dataframe():
    data = {'group': [0, 0, 1, 1, 0],
            'outcome': [0, 1, 0, 1, 0]}
    return pd.DataFrame(data)

def test_reweight_increases_size(sample_dataframe):
    reweighted_df = reweight(sample_dataframe, 'group', 'outcome', 1, 0.5)
    assert len(reweighted_df) > len(sample_dataframe)

def test_reweight_weight_zero(sample_dataframe):
    reweighted_df = reweight(sample_dataframe, 'group', 'outcome', 1, 0)
    assert len(reweighted_df) == len(sample_dataframe)

def test_reweight_invalid_group_col(sample_dataframe):
    with pytest.raises(KeyError):
        reweight(sample_dataframe, 'invalid_group', 'outcome', 1, 0.5)

def test_reweight_invalid_outcome_col(sample_dataframe):
    with pytest.raises(KeyError):
        reweight(sample_dataframe, 'group', 'invalid_outcome', 1, 0.5)

def test_reweight_negative_weight(sample_dataframe):
    reweighted_df = reweight(sample_dataframe, 'group', 'outcome', 1, -0.5)
    assert len(reweighted_df) == len(sample_dataframe)
