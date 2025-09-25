import pytest
import tensorflow as tf
from tensorflow import keras
from definition_f497a63ce2ee4f4d87d34d0d7428f1bb import build_lstm_attention_model, AttentionLayer # Assuming AttentionLayer is defined in the same module

@pytest.mark.parametrize("input_shape, num_classes, expected_exception", [
    # Test Case 1: Valid inputs for binary classification
    ((50, 1), 1, None),
    # Test Case 2: Invalid input_shape - integer instead of a tuple/list
    (10, 1, (TypeError, ValueError)),
    # Test Case 3: Invalid input_shape - tuple with incorrect dimensions for LSTM (e.g., (timesteps,) instead of (timesteps, features))
    ((50,), 1, (ValueError, TypeError)),
    # Test Case 4: Invalid num_classes - non-positive integer
    ((50, 1), 0, ValueError),
    # Test Case 5: Valid inputs for binary classification with different timesteps
    ((10, 1), 1, None),
])
def test_build_lstm_attention_model(input_shape, num_classes, expected_exception):
    if expected_exception:
        # Test cases expected to raise an exception
        with pytest.raises(expected_exception):
            build_lstm_attention_model(input_shape, num_classes)
    else:
        # Test cases for expected functionality
        model = build_lstm_attention_model(input_shape, num_classes)

        # Assert model is a Keras Model instance
        assert isinstance(model, keras.Model), "Function should return a Keras Model"

        # Assert there are at least 3 layers: LSTM, Attention, Dense
        assert len(model.layers) >= 3, "Model should have at least an LSTM, AttentionLayer, and Dense layer"

        # Assert specific layer types are present and in the correct order
        lstm_idx, attention_idx, dense_idx = -1, -1, -1
        for i, layer in enumerate(model.layers):
            if lstm_idx == -1 and issubclass(type(layer), keras.layers.LSTM):
                lstm_idx = i
            elif attention_idx == -1 and issubclass(type(layer), AttentionLayer):
                attention_idx = i
            elif dense_idx == -1 and issubclass(type(layer), keras.layers.Dense):
                dense_idx = i
        
        assert lstm_idx != -1, "Model must contain an LSTM layer"
        assert attention_idx != -1, "Model must contain an AttentionLayer"
        assert dense_idx != -1, "Model must contain a Dense output layer"

        assert lstm_idx < attention_idx, "LSTM layer should precede AttentionLayer"
        assert attention_idx < dense_idx, "AttentionLayer should precede Dense layer"

        # Assert final output shape (number of units in the Dense layer)
        try:
            # Create a dummy input tensor to ensure the model is built and get output shape
            dummy_input = tf.zeros((1, *input_shape), dtype=tf.float32) # Batch size 1
            output = model(dummy_input)
            assert output.shape[-1] == num_classes, \
                f"Model output units should be {num_classes}, but got {output.shape[-1]}"
        except Exception as e:
            pytest.fail(f"Could not verify model output shape due to: {e}")