import pytest
import tensorflow as tf
import numpy as np

# definition_2667f312e1504211967a6263c184c7fe block START
from definition_2667f312e1504211967a6263c184c7fe import AttentionLayer
# definition_2667f312e1504211967a6263c184c7fe block END

@pytest.mark.parametrize("inputs, expected_output, expected_exception", [
    # Test case 1: Standard valid input (batch_size, timesteps, features)
    (tf.zeros((2, 10, 5), dtype=tf.float32), {'context_shape': (2, 5), 'attention_shape': (2, 10), 'dtype': tf.float32}, None),
    
    # Test case 2: Edge case - Single timestep input
    (tf.zeros((3, 1, 8), dtype=tf.float64), {'context_shape': (3, 8), 'attention_shape': (3, 1), 'dtype': tf.float64}, None),
    
    # Test case 3: Edge case - Single batch item input
    (tf.ones((1, 20, 4), dtype=tf.float32), {'context_shape': (1, 4), 'attention_shape': (1, 20), 'dtype': tf.float32}, None),
    
    # Test case 4: Invalid input - Incorrect tensor rank (e.g., 2D instead of 3D)
    (tf.zeros((10, 5), dtype=tf.float32), None, ValueError),
    
    # Test case 5: Invalid input - Non-tensor input (e.g., numpy array)
    # Keras layers typically expect tf.Tensor inputs. Direct numpy arrays usually result in TypeError when tf.shape is called.
    (np.zeros((2, 10, 5), dtype=np.float32), None, TypeError),
])
def test_attention_layer_call(inputs, expected_output, expected_exception):
    attention_layer = AttentionLayer() # Instantiate the layer for each test

    try:
        context_vector, attention_weights = attention_layer.call(inputs)

        # Assertions for successful calls
        assert expected_exception is None, "Expected an exception but none was raised."
        
        assert isinstance(context_vector, tf.Tensor), "Context vector is not a tf.Tensor"
        assert isinstance(attention_weights, tf.Tensor), "Attention weights is not a tf.Tensor"
        
        assert context_vector.shape == expected_output['context_shape'], \
            f"Context vector shape mismatch. Expected {expected_output['context_shape']}, got {context_vector.shape}"
        assert attention_weights.shape == expected_output['attention_shape'], \
            f"Attention weights shape mismatch. Expected {expected_output['attention_shape']}, got {attention_weights.shape}"
        
        assert context_vector.dtype == expected_output['dtype'], \
            f"Context vector dtype mismatch. Expected {expected_output['dtype']}, got {context_vector.dtype}"
        assert attention_weights.dtype == expected_output['dtype'], \
            f"Attention weights dtype mismatch. Expected {expected_output['dtype']}, got {attention_weights.dtype}"

    except Exception as e:
        # Assertions for expected exceptions
        assert expected_exception is not None, f"Unexpected exception raised: {type(e).__name__}"
        assert isinstance(e, expected_exception), \
            f"Expected exception type {expected_exception.__name__}, but got {type(e).__name__}"