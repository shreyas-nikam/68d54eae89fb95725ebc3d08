import pytest
import numpy as np
from definition_be7b4d088f394c9ea517338534792f64 import add_gaussian_noise_augmentation

@pytest.mark.parametrize("sequences, noise_level, expected_type, expected_shape, expect_change, expected_exception", [
    # Test Case 1: Standard functionality with a positive noise_level on a 3D numpy array
    (np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=np.float32), 0.1, np.ndarray, (2, 3, 1), True, None),
    
    # Test Case 2: Zero noise_level - output should be identical to input
    (np.array([[[10.0], [20.0], [30.0]]], dtype=np.float32), 0.0, np.ndarray, (1, 3, 1), False, None),
    
    # Test Case 3: Empty sequences array - should return an empty array of the same shape
    (np.empty((0, 5, 1), dtype=np.float32), 0.05, np.ndarray, (0, 5, 1), False, None),
    
    # Test Case 4: Non-numpy sequences input (e.g., a list) - expects a TypeError
    ([[1.0, 2.0]], 0.1, None, None, None, TypeError),
    
    # Test Case 5: Invalid noise_level type (e.g., a string) - expects a TypeError
    (np.array([[[1.0]]], dtype=np.float32), "invalid_noise", None, None, None, TypeError),
])
def test_add_gaussian_noise_augmentation(sequences, noise_level, expected_type, expected_shape, expect_change, expected_exception):
    """
    Tests the add_gaussian_noise_augmentation function for various scenarios.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            add_gaussian_noise_augmentation(sequences, noise_level)
    else:
        # Create a copy of the original sequences for comparison if no exception is expected
        original_sequences_copy = sequences.copy() if isinstance(sequences, np.ndarray) else sequences
        
        result = add_gaussian_noise_augmentation(sequences, noise_level)
        
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape
        
        if expect_change:
            # If noise_level is positive, the output should be different from the input
            # Due to the random nature, we assert that *some* elements have changed.
            # Using np.any(result != original_sequences_copy) is a simple way to check for change.
            assert not np.array_equal(result, original_sequences_copy)
            assert np.any(np.abs(result - original_sequences_copy) > 1e-6) # Check for meaningful difference
        else:
            # If noise_level is zero or input is empty, output should be numerically identical to input
            np.testing.assert_array_almost_equal(result, original_sequences_copy, decimal=7)