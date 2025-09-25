import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from definition_b9a7613d295f4cd2842df8da3f210da4 import build_lstm_model

@pytest.mark.parametrize("input_shape, num_classes, expected_exception, expected_output_units, expected_output_activation", [
    # Test Case 1: Basic functionality with expected inputs for binary classification
    # Asserts model creation, layer types, output configuration, and compilation.
    ((10, 1), 1, None, 1, 'sigmoid'),

    # Test Case 2: Different valid input_shape to ensure flexibility and correct input handling
    # Asserts model adapts to different sequence lengths and feature dimensions.
    ((50, 5), 1, None, 1, 'sigmoid'),

    # Test Case 3: Invalid num_classes (0) - Keras Dense layer units must be > 0.
    # Expects a ValueError as a Dense layer cannot have 0 units.
    ((10, 1), 0, ValueError, None, None),

    # Test Case 4: Invalid num_classes (>1) for a "binary classification" model with "sigmoid" output.
    # The docstring explicitly states "binary classification" and "sigmoid" activation,
    # which implies num_classes must be 1. This tests the semantic contract of the function.
    # A well-designed function should raise a ValueError if num_classes is not 1 for binary classification.
    ((10, 1), 2, ValueError, None, None),

    # Test Case 5: Invalid input_shape type (e.g., string)
    # Keras LSTM layer's `input_shape` parameter expects a tuple of integers.
    # Expects a TypeError when an incompatible type is provided.
    ("invalid_shape", 1, TypeError, None, None),
])
def test_build_lstm_model(input_shape, num_classes, expected_exception, expected_output_units, expected_output_activation):
    if expected_exception:
        with pytest.raises(expected_exception):
            build_lstm_model(input_shape, num_classes)
    else:
        model = build_lstm_model(input_shape, num_classes)

        # Assert return type
        assert isinstance(model, keras.Model)

        # Assert number of layers (LSTM + Dense).
        # A Sequential model initialized with `input_shape` in the first layer usually results in 2 layers internally (LSTM, Dense),
        # as the InputLayer is implicitly created and attached.
        assert len(model.layers) == 2

        # Assert layer types
        assert isinstance(model.layers[0], layers.LSTM)
        assert isinstance(model.layers[1], layers.Dense)

        # Assert input shape of the first layer (LSTM)
        # The input_shape property of a layer will include the batch dimension as None.
        assert model.layers[0].input_shape == (None,) + input_shape

        # Assert output layer configuration
        dense_layer = model.layers[1]
        assert dense_layer.units == expected_output_units
        assert dense_layer.activation.__name__ == expected_output_activation

        # Assert model is compiled with appropriate settings for binary classification
        assert model.optimizer is not None
        assert model.loss == 'binary_crossentropy'
        assert 'accuracy' in model.metrics_names