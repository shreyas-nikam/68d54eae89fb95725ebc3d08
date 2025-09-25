import pytest
import numpy as np
from definition_70726fa56a1d4914b8d2466d3958f1cf import generate_synthetic_time_series_data

@pytest.mark.parametrize(
    "n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale, trend_slope, threshold_for_label, expected_result_or_exception",
    [
        # Test Case 1: Standard valid input - check shapes, types, and binary labels
        (10, 50, 0.1, 0.2, 0.5, 0.05, 0.5, ((10, 50, 1), (10,))),
        # Test Case 2: Edge case: n_samples = 0 - check empty outputs with correct shapes
        (0, 50, 0.1, 0.2, 0.5, 0.05, 0.5, ((0, 50, 1), (0,))),
        # Test Case 3: Edge case: timesteps = 0 - check empty time series dimension, correct labels dimension
        (10, 0, 0.1, 0.2, 0.5, 0.05, 0.5, ((10, 0, 1), (10,))),
        # Test Case 4: Edge case: All noise, slope, and frequency are zero - check stability under these inputs
        (5, 10, 0.0, 0.0, 0.0, 0.0, 0.5, ((5, 10, 1), (5,))),
        # Test Case 5: Invalid input type for n_samples - expect TypeError
        ("invalid", 50, 0.1, 0.2, 0.5, 0.05, 0.5, TypeError),
    ]
)
def test_generate_synthetic_time_series_data(
    n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale,
    trend_slope, threshold_for_label, expected_result_or_exception
):
    try:
        X, y = generate_synthetic_time_series_data(
            n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale,
            trend_slope, threshold_for_label
        )
        # If execution reaches here, no exception occurred.
        # Assert that an exception was NOT expected.
        assert not isinstance(expected_result_or_exception, type), \
            f"Expected {expected_result_or_exception.__name__} but no exception was raised."

        expected_X_shape, expected_y_shape = expected_result_or_exception

        assert isinstance(X, np.ndarray), "Features should be a numpy array."
        assert isinstance(y, np.ndarray), "Labels should be a numpy array."
        assert X.shape == expected_X_shape, f"Expected X shape {expected_X_shape}, but got {X.shape}."
        assert y.shape == expected_y_shape, f"Expected y shape {expected_y_shape}, but got {y.shape}."

        # Additional checks for standard cases (only when n_samples > 0)
        if n_samples > 0:
            # Check if labels are binary (0 or 1)
            assert np.all(((y == 0) | (y == 1))), "Labels should be binary (0 or 1)."
            # Check if the last dimension of X is 1 (as per spec, only meaningful if timesteps > 0)
            if timesteps > 0:
                assert X.shape[2] == 1, "The last dimension of X should be 1."

    except Exception as e:
        # If an exception occurred, ensure it's the expected type
        assert isinstance(expected_result_or_exception, type), \
            f"Unexpected exception {type(e).__name__} - {e}"
        assert isinstance(e, expected_result_or_exception), \
            f"Expected exception type {expected_result_or_exception.__name__}, but got {type(e).__name__}."
