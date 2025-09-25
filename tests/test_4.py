import pytest
import pandas as pd
import numpy as np

from definition_63bd17d977c944a38b8c0712f074984b import reweight

@pytest.fixture
def sample_dataframe():
    """
    A DataFrame with a clear underrepresented group (gender=0, e.g., Female)
    80 privileged (1), 20 non-privileged (0)
    """
    data = {
        'group_col': [1]*80 + [0]*20,
        'outcome_col': [1]*70 + [0]*10 + [1]*15 + [0]*5,
        'other_col': np.random.rand(100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def balanced_dataframe():
    """
    A DataFrame with perfectly balanced groups (50/50 split).
    """
    data = {
        'group_col': [1]*50 + [0]*50,
        'outcome_col': np.random.randint(0, 2, 100),
        'other_col': np.random.rand(100)
    }
    return pd.DataFrame(data)

# Test 1: Basic functionality - underrepresented group gets duplicated
def test_reweight_underrepresented_group_increases(sample_dataframe):
    df = sample_dataframe.copy()
    initial_len = len(df) # 100
    initial_group_0_count = df[df['group_col'] == 0].shape[0] # 20 (non-privileged)

    reweighted_df = reweight(df, 'group_col', 'outcome_col', privileged_group=1, weight=0.5)

    # Expected calculation based on the function's internal logic:
    # df_len = 100
    # privileged_group = 1, non-privileged group = 0
    # group_A_size (privileged=1) = 80, group_B_size (non-privileged=0) = 20
    # `group_A_size < group_B_size` (80 < 20) is False.
    # So `else` branch: underrepresented_group = 0, overrepresented_group = 1.
    # underrepresented_group_df (group_col=0) has 20 rows.
    # reweighted_size = int(0.5 * 100) = 50.
    # num_rows_to_add = min(reweighted_size, group_B_size) = min(50, 20) = 20.
    # actual_rows_to_duplicate = min(num_rows_to_add, len(underrepresented_group_df)) = min(20, 20) = 20.
    expected_added_rows = 20
    expected_total_len = initial_len + expected_added_rows
    expected_group_0_count = initial_group_0_count + expected_added_rows
    expected_group_1_count = df[df['group_col'] == 1].shape[0] # Should remain unchanged (80)

    assert len(reweighted_df) == expected_total_len
    assert reweighted_df[reweighted_df['group_col'] == 0].shape[0] == expected_group_0_count
    assert reweighted_df[reweighted_df['group_col'] == 1].shape[0] == expected_group_1_count
    assert not reweighted_df.equals(df) # Ensure it's not the exact same DataFrame

# Test 2: Weight is 0, no rows should be added, DataFrame should be identical
def test_reweight_zero_weight_no_change(sample_dataframe):
    df = sample_dataframe.copy()
    reweighted_df = reweight(df, 'group_col', 'outcome_col', privileged_group=1, weight=0.0)

    # With weight=0, reweighted_size will be 0, num_rows_to_add will be 0,
    # and actual_rows_to_duplicate will be 0. So no rows should be duplicated.
    assert len(reweighted_df) == len(df)
    pd.testing.assert_frame_equal(reweighted_df, df) # Ensure content is also identical

# Test 3: Groups are initially balanced, should still reweight one group based on internal tie-breaking logic
def test_reweight_balanced_groups(balanced_dataframe):
    df = balanced_dataframe.copy()
    initial_len = len(df) # 100
    # privileged_group = 1, non-privileged group = 0
    # group_A_size (privileged=1) = 50, group_B_size (non-privileged=0) = 50
    # Condition `group_A_size < group_B_size` (50 < 50) is False.
    # So `else` branch: underrepresented_group = 0, overrepresented_group = 1.
    initial_group_0_count = df[df['group_col'] == 0].shape[0] # 50 (non-privileged, now identified as underrepresented)

    reweighted_df = reweight(df, 'group_col', 'outcome_col', privileged_group=1, weight=0.6)

    # Expected calculation:
    # df_len = 100
    # underrepresented_group = 0
    # underrepresented_group_df (group_col=0) has 50 rows.
    # reweighted_size = int(0.6 * 100) = 60.
    # num_rows_to_add = min(reweighted_size, group_B_size) = min(60, 50) = 50.
    # actual_rows_to_duplicate = min(num_rows_to_add, len(underrepresented_group_df)) = min(50, 50) = 50.
    expected_added_rows = 50
    expected_total_len = initial_len + expected_added_rows
    expected_group_0_count = initial_group_0_count + expected_added_rows
    expected_group_1_count = df[df['group_col'] == 1].shape[0] # Should remain unchanged (50)

    assert len(reweighted_df) == expected_total_len
    assert reweighted_df[reweighted_df['group_col'] == 0].shape[0] == expected_group_0_count
    assert reweighted_df[reweighted_df['group_col'] == 1].shape[0] == expected_group_1_count
    assert not reweighted_df.equals(df)

# Test 4: Empty DataFrame should return an empty DataFrame
def test_reweight_empty_dataframe():
    df = pd.DataFrame(columns=['group_col', 'outcome_col', 'other_col'])
    reweighted_df = reweight(df, 'group_col', 'outcome_col', privileged_group=1, weight=0.5)

    assert len(reweighted_df) == 0
    pd.testing.assert_frame_equal(reweighted_df, df) # Ensure columns are preserved for empty DF

# Test 5: Invalid column name should raise KeyError
def test_reweight_invalid_column_name(sample_dataframe):
    df = sample_dataframe.copy()
    with pytest.raises(KeyError):
        reweight(df, 'non_existent_group', 'outcome_col', privileged_group=1, weight=0.5)
    with pytest.raises(KeyError):
        reweight(df, 'group_col', 'non_existent_outcome', privileged_group=1, weight=0.5)