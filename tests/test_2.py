import pytest
import numpy as np
from definition_6f528af54ff7476aa350300ca71ef276 import amplitude_scaling_augmentation

@pytest.mark.parametrize("sequences, scale_factor_range, expected", [
    # Test Case 1: Fixed positive scaling (deterministic, expected functionality)
    (np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32), (2.0, 2.0), 
     np.array([[[2.0], [4.0]], [[6.0], [8.0]]], dtype=np.float32)),
    # Test Case 2: Fixed negative scaling (deterministic, checks behavior with negative factors)
    (np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32), (-1.0, -1.0), 
     np.array([[[-1.0], [-2.0]], [[-3.0], [-4.0]]], dtype=np.float32)),
    # Test Case 3: Empty sequences (edge case)
    (np.array([], dtype=np.float32).reshape(0, 50, 1), (0.8, 1.2), 
     np.array([], dtype=np.float32).reshape(0, 50, 1)),
    # Test Case 4: Invalid scale factor range order (min > max, error handling)
    (np.array([[[1.0]]], dtype=np.float32), (1.5, 0.5), ValueError),
    # Test Case 5: Invalid sequences type (list instead of numpy array, error handling)
    ([[1.0, 2.0]], (0.8, 1.2), TypeError),
])
def test_amplitude_scaling_augmentation(sequences, scale_factor_range, expected):
    try:
        result = amplitude_scaling_augmentation(sequences, scale_factor_range)
        # For numpy arrays, use np.testing.assert_array_almost_equal for robust comparison
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_almost_equal(result, expected)
        else:
            # This branch should not be hit with current expected values if only testing numpy arrays
            assert result == expected 
    except Exception as e:
        assert isinstance(e, expected)