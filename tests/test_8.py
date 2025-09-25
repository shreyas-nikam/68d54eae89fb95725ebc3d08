import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# This block must remain as is, do not replace or remove.
from definition_1efa163b55ac4380985a41987694ae98 import preprocess_and_split
# End of placeholder block.

# Helper function to generate dummy data for tests
def _generate_data(n_samples, n_features):
    np.random.seed(42) # For reproducibility
    X = np.random.rand(n_samples, n_features) * 100
    y = np.random.randint(0, 2, n_samples)
    return X, y

def test_preprocess_and_split_standard_case():
    # Test with valid inputs, checking shapes, types, and scaling properties.
    n_samples, n_features = 100, 5
    X, y = _generate_data(n_samples, n_features)
    test_size = 0.2
    random_state = 42

    X_scaled_all, X_train, X_test, y_train, y_test = preprocess_and_split(X, y, test_size, random_state)

    # Assert shapes
    assert X_scaled_all.shape == (n_samples, n_features)
    assert X_train.shape[0] == int(n_samples * (1 - test_size))
    assert X_test.shape[0] == int(n_samples * test_size)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # Assert types
    assert isinstance(X_scaled_all, np.ndarray)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # Assert scaling properties: mean ~0 and std ~1 for each feature in X_scaled_all
    # This assumes StandardScaler is fit on the full X and then transforms it to X_scaled_all
    assert np.allclose(X_scaled_all.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_scaled_all.std(axis=0), 1.0, atol=1e-7)

def test_preprocess_and_split_empty_inputs_raises_error():
    # Test behavior with empty input arrays, expecting an error from scikit-learn.
    X = np.array([]).reshape(0, 5) # 0 samples, 5 features
    y = np.array([])
    test_size = 0.2
    random_state = 42

    # StandardScaler typically raises ValueError if n_samples = 0
    with pytest.raises(ValueError, match="Expected n_samples >= 1"):
        preprocess_and_split(X, y, test_size, random_state)

def test_preprocess_and_split_single_sample_raises_error():
    # Test behavior with a single sample when a non-trivial split is requested.
    X = np.array([[10.0, 20.0, 30.0]])
    y = np.array([0])
    test_size = 0.2 # Cannot split 1 sample into 80/20
    random_state = 42

    # train_test_split requires at least 2 samples if test_size is not 0 or 1.0.
    with pytest.raises(ValueError, match="Can't have a test size of 0.2 with one sample."):
        preprocess_and_split(X, y, test_size, random_state)

def test_preprocess_and_split_test_size_zero():
    # Test when test_size is 0, implying all data goes to the training set and no test set.
    n_samples, n_features = 10, 3
    X, y = _generate_data(n_samples, n_features)
    test_size = 0.0
    random_state = 42

    X_scaled_all, X_train, X_test, y_train, y_test = preprocess_and_split(X, y, test_size, random_state)

    # All samples should be in the training set
    assert X_train.shape[0] == n_samples
    assert y_train.shape[0] == n_samples
    assert X_test.shape == (0, n_features)
    assert y_test.shape == (0,)

    # Check scaling properties (mean ~0, std ~1 for X_scaled_all)
    assert np.allclose(X_scaled_all.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_scaled_all.std(axis=0), 1.0, atol=1e-7)
    # X_train should be identical to X_scaled_all as no test split happened
    assert np.allclose(X_train, X_scaled_all)

def test_preprocess_and_split_invalid_X_type_raises_type_error():
    # Test with X being a non-numpy array type, expecting a TypeError for robust input validation.
    X = [[1, 2], [3, 4]] # List of lists, not a numpy array
    y = np.array([0, 1])
    test_size = 0.5
    random_state = 42

    # A senior architect's implementation should explicitly validate input types.
    with pytest.raises(TypeError, match="X must be a numpy.ndarray"):
        preprocess_and_split(X, y, test_size, random_state)
