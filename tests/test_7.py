import pytest
import numpy as np
from unittest.mock import MagicMock
from definition_bf5f446abd1d44aa900f42b616fff73b import train_model

# Mock tensorflow.keras.callbacks.History
class MockHistory:
    """A mock object to simulate tf.keras.callbacks.History."""
    def __init__(self, history_data=None):
        self.history = history_data if history_data is not None else {'loss': [0.5], 'accuracy': [0.8]}

@pytest.fixture
def mock_keras_model():
    """Fixture for a MagicMock Keras model that simulates a tf.keras.Model."""
    mock_model = MagicMock()
    # Configure the mock model's methods for expected behavior
    mock_model.compile.return_value = None # Keras compile method doesn't return anything
    mock_model.fit.return_value = MockHistory() # Default history for successful fit
    mock_model.evaluate.return_value = (0.2, 0.9) # Example loss, accuracy from evaluate
    return mock_model

# Test Case 1: Successful training with valid inputs
def test_train_model_success(mock_keras_model):
    """
    Tests the successful training of a model with valid input data and parameters.
    Verifies that model.compile and model.fit are called with correct arguments,
    and a History object is returned.
    """
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.randint(0, 2, (100,))
    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.randint(0, 2, (20,))
    epochs = 5
    batch_size = 32

    history = train_model(mock_keras_model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Assert model.compile was called with specific parameters as per the docstring
    mock_keras_model.compile.assert_called_once_with(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Assert model.fit was called with specific parameters
    mock_keras_model.fit.assert_called_once_with(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0 # Assuming verbose=0 for automated training
    )
    # Assert a History object is returned and contains expected data
    assert isinstance(history, MockHistory)
    assert 'loss' in history.history
    assert 'accuracy' in history.history

# Test Case 2: Training with zero epochs
def test_train_model_zero_epochs(mock_keras_model):
    """
    Tests the behavior when training with zero epochs.
    The model should still be compiled, but fit should be called with epochs=0,
    and an empty or minimal history is expected.
    """
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.randint(0, 2, (100,))
    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.randint(0, 2, (20,))
    epochs = 0 # Zero epochs
    batch_size = 32

    # Configure mock.fit to return an empty history for 0 epochs, as Keras does
    mock_keras_model.fit.return_value = MockHistory(history_data={})

    history = train_model(mock_keras_model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Model should still attempt to compile
    mock_keras_model.compile.assert_called_once()
    # Model.fit should be called with epochs=0
    mock_keras_model.fit.assert_called_once_with(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )
    assert isinstance(history, MockHistory)
    assert not history.history # Expect an empty history for 0 epochs

# Test Case 3: Invalid model object
@pytest.mark.parametrize("invalid_model, expected_exception", [
    (None, AttributeError),         # Calling .compile on None
    (123, AttributeError),          # Calling .compile on an integer
    (object(), AttributeError),     # Calling .compile on a generic object
    (MagicMock(spec_set=[]), AttributeError) # Mock without compile/fit methods
])
def test_train_model_invalid_model_object(invalid_model, expected_exception):
    """
    Tests `train_model` with various invalid objects passed as the `model` argument.
    Expects an AttributeError as these objects lack the `.compile` or `.fit` methods.
    """
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.randint(0, 2, (100,))
    X_val = np.random.rand(20, 10, 1)
    y_val = np.random.randint(0, 2, (20,))
    epochs = 5
    batch_size = 32

    with pytest.raises(expected_exception):
        train_model(invalid_model, X_train, y_train, X_val, y_val, epochs, batch_size)

# Test Case 4: Mismatched data shapes (training or validation)
@pytest.mark.parametrize(
    "X_train_shape, y_train_shape, X_val_shape, y_val_shape, expected_exception_match",
    [
        # Mismatched training data sample count
        ((100, 10, 1), (99,), (20, 10, 1), (20,), "X_train and y_train must have the same number of samples."),
        # Mismatched validation data sample count
        ((100, 10, 1), (100,), (20, 10, 1), (19,), "X_val and y_val must have the same number of samples."),
    ]
)
def test_train_model_mismatched_data_shapes(
    mock_keras_model, X_train_shape, y_train_shape, X_val_shape, y_val_shape, expected_exception_match
):
    """
    Tests `train_model` with input data (X_train/y_train or X_val/y_val) having mismatched
    number of samples, expecting a ValueError.
    """
    X_train = np.random.rand(*X_train_shape)
    y_train = np.random.randint(0, 2, y_train_shape)
    X_val = np.random.rand(*X_val_shape)
    y_val = np.random.randint(0, 2, y_val_shape)
    epochs = 5
    batch_size = 32

    with pytest.raises(ValueError, match=expected_exception_match):
        train_model(mock_keras_model, X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Assert that compile and fit were not called if error occurs during data validation
    mock_keras_model.compile.assert_not_called()
    mock_keras_model.fit.assert_not_called()

# Test Case 5: Empty or None data
@pytest.mark.parametrize(
    "X_train_data, y_train_data, X_val_data, y_val_data, expected_exception, expected_match",
    [
        # Empty training data arrays
        (np.array([]).reshape(0,10,1), np.array([]).reshape(0,), np.random.rand(20,10,1), np.random.randint(0,2,(20,)), ValueError, "Input arrays should not be empty."),
        # X_train is None
        (None, np.random.randint(0,2,(100,)), np.random.rand(20,10,1), np.random.randint(0,2,(20,)), ValueError, "Training data cannot be None."),
        # y_train is None
        (np.random.rand(100,10,1), None, np.random.rand(20,10,1), np.random.randint(0,2,(20,)), ValueError, "Training data cannot be None."),
        # X_val is None
        (np.random.rand(100,10,1), np.random.randint(0,2,(100,)), None, np.random.randint(0,2,(20,)), ValueError, "Validation data cannot be None."),
        # y_val is None
        (np.random.rand(100,10,1), np.random.randint(0,2,(100,)), np.random.rand(20,10,1), None, ValueError, "Validation data cannot be None."),
    ]
)
def test_train_model_empty_or_none_data(
    mock_keras_model, X_train_data, y_train_data, X_val_data, y_val_data, expected_exception, expected_match
):
    """
    Tests `train_model` when training or validation data is empty (numpy array) or None.
    Expects a ValueError, either from `train_model` itself or from the underlying Keras `.fit` call.
    """
    epochs = 5
    batch_size = 32

    # If empty numpy array, mock `fit` method to raise ValueError, mimicking Keras behavior
    if isinstance(X_train_data, np.ndarray) and X_train_data.size == 0:
        mock_keras_model.fit.side_effect = ValueError(expected_match)
    
    with pytest.raises(expected_exception, match=expected_match):
        train_model(mock_keras_model, X_train_data, y_train_data, X_val_data, y_val_data, epochs, batch_size)

    # Assert that compile and fit were not called if error occurs during data validation
    # (i.e., if the error is due to None data or validation errors before calling fit)
    if expected_match not in ["Input arrays should not be empty."]: # Error specific to model.fit
        mock_keras_model.compile.assert_not_called()
        mock_keras_model.fit.assert_not_called()