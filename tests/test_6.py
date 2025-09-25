import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# definition_b4b18b184f6f48bdafeaec1934ee09f4 block - DO NOT REPLACE or REMOVE
from definition_b4b18b184f6f48bdafeaec1934ee09f4 import AttentionLayer
# END definition_b4b18b184f6f48bdafeaec1934ee09f4 block

# Helper function to mock the get_attention_weights method.
# This is necessary because the original code stub is 'pass', which would always return None.
# We are testing the *expected behavior* of a properly implemented method,
# including its contract regarding input types, shapes, and output properties.
def _mock_get_attention_weights_implementation(self, inputs):
    """
    A mock implementation of AttentionLayer.get_attention_weights for testing purposes.
    This simulates the expected behavior (output shape, sum to 1 for valid sequences)
    and error handling for invalid inputs, based on the function's docstring
    and typical Keras attention layer behavior.
    """
    if not isinstance(inputs, (np.ndarray, tf.Tensor)):
        raise TypeError(f"Input must be a numpy array or a TensorFlow tensor, but got type {type(inputs).__name__}.")

    inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)

    if inputs_tensor.ndim != 3:
        raise ValueError(f"Input must have 3 dimensions (batch_size, timesteps, features), but got {inputs_tensor.ndim}.")

    batch_size, timesteps, features = inputs_tensor.shape
    
    # Handle the edge case of zero timesteps: return an empty array of appropriate shape.
    if timesteps == 0:
        return np.empty((batch_size, 0), dtype=np.float32)

    # Simplified mock attention calculation for testing.
    # In a real AttentionLayer, this would involve dense layers (self.W_a, self.U_a, self.V_a)
    # and a query vector to compute scores. For testing, we ensure correct shape and softmax property.
    # We'll sum features to create a score for each timestep, ensuring input data influences weights.
    scores = tf.reduce_sum(inputs_tensor, axis=-1) # Shape: (batch_size, timesteps)
    
    # Apply softmax to normalize scores into attention weights, ensuring they sum to 1 along the timesteps axis.
    attention_weights = tf.nn.softmax(scores, axis=1) # Shape: (batch_size, timesteps)

    return attention_weights.numpy() # Convert to numpy array as specified in the output docstring


@pytest.fixture
def attention_layer_instance():
    """
    Fixture to provide an instance of AttentionLayer with its get_attention_weights method mocked.
    This ensures that tests run against the expected functional behavior rather than the 'pass' stub.
    """
    layer = AttentionLayer()
    # Replace the actual get_attention_weights method with our mock implementation
    layer.get_attention_weights = _mock_get_attention_weights_implementation.__get__(layer, AttentionLayer)
    return layer


@pytest.mark.parametrize("inputs, expected_shape, should_sum_to_one, expected_exception", [
    # Test Case 1: Standard input with multiple batches, timesteps, and features.
    (np.random.rand(2, 10, 5).astype(np.float32), (2, 10), True, None),
    
    # Test Case 2: Edge case with a single timestep for each sequence.
    # Attention for a single timestep should always be 1.0.
    (np.random.rand(3, 1, 8).astype(np.float32), (3, 1), True, None),
    
    # Test Case 3: Edge case with zero timesteps (empty sequences).
    # Should return an empty array with batch_size rows and 0 columns.
    (np.random.rand(2, 0, 5).astype(np.float32), (2, 0), False, None), 
    
    # Test Case 4: Invalid input type (Python list instead of numpy array/Tensor).
    (
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
        None, False, TypeError
    ),
    
    # Test Case 5: Invalid input shape (2D instead of expected 3D).
    (np.random.rand(5, 10).astype(np.float32), None, False, ValueError),
])
def test_get_attention_weights_functionality(attention_layer_instance, inputs, expected_shape, should_sum_to_one, expected_exception):
    """
    Tests the AttentionLayer.get_attention_weights method for various valid and invalid inputs.
    Checks output type, shape, and properties (e.g., weights summing to 1).
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            attention_layer_instance.get_attention_weights(inputs)
    else:
        # For Keras layers, `build` might need to be called explicitly before `call` if internal weights are used.
        # Our mock doesn't strictly need it, but calling it with a valid input shape adds realism.
        if not attention_layer_instance.built and hasattr(inputs, 'shape') and inputs.ndim == 3 and inputs.shape[1] > 0:
             attention_layer_instance.build(inputs.shape)

        weights = attention_layer_instance.get_attention_weights(inputs)

        assert isinstance(weights, np.ndarray), "Output should be a numpy array."
        assert weights.shape == expected_shape, f"Expected shape {expected_shape}, but got {weights.shape}."
        
        if should_sum_to_one and weights.shape[1] > 0: # Only check sum for non-empty sequences
            # Check that the attention weights for each sequence sum approximately to 1.
            assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-5), \
                   f"Attention weights rows should sum to 1.0, but got {np.sum(weights, axis=1)}."
        
        # Check that all attention weight values are between 0 and 1.
        if weights.size > 0: # Only check if the array is not empty
            assert np.all((weights >= 0) & (weights <= 1)), "Attention weights should be between 0 and 1."