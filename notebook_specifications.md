
# Technical Specification for Jupyter Notebook: Sequence-based XAI Demonstrator

## 1. Notebook Overview

This Jupyter Notebook provides a hands-on demonstration of sequence-based Explainable AI (XAI) techniques, focusing on pre-modeling and in-modeling approaches. Learners will generate synthetic time-series data, apply data augmentation as a pre-modeling XAI technique, and implement attention mechanisms as an in-modeling XAI technique. The impact of these techniques on model performance and interpretability will be visualized and explained.

### Learning Goals

*   Understand the fundamental concepts of sequence-based Explainable AI and how it differs from other XAI methods.
*   Experiment with pre-modeling XAI techniques, specifically data augmentation for time-series data, and observe its impact on model robustness and performance.
*   Apply in-modeling XAI techniques, such as attention mechanisms, to a time-series prediction model to enhance interpretability by identifying influential time steps.
*   Visualize and interpret attention weights to gain insights into model predictions for sequence data.

## 2. Code Requirements

### List of Expected Libraries

*   `numpy` (for numerical operations)
*   `pandas` (for data manipulation)
*   `matplotlib.pyplot` (for static plotting)
*   `seaborn` (for enhanced static plotting)
*   `sklearn.model_selection` (for splitting data)
*   `sklearn.preprocessing` (for data scaling)
*   `tensorflow` (for building and training deep learning models, specifically `keras`)

### List of Algorithms or Functions to be Implemented

1.  **`generate_synthetic_time_series_data(n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale, trend_slope)`**: Generates a dataset of synthetic time-series sequences.
    *   `n_samples`: Number of individual time series sequences.
    *   `timesteps`: Length of each time series sequence.
    *   `frequency`: Base frequency of the sine wave pattern.
    *   `amplitude_noise_scale`: Scale of random noise added to amplitude.
    *   `pattern_noise_scale`: Scale of Gaussian noise added to the overall pattern.
    *   `trend_slope`: Linear trend component added to the series.
    *   Returns: `numpy.ndarray` of shape `(n_samples, timesteps, 1)` for features and `numpy.ndarray` of shape `(n_samples,)` for target labels (binary classification based on a threshold of the last value).
2.  **`add_gaussian_noise_augmentation(sequences, noise_level)`**: Adds Gaussian noise to a batch of time-series sequences.
    *   `sequences`: Input `numpy.ndarray` of time series.
    *   `noise_level`: Standard deviation of the Gaussian noise.
    *   Returns: Augmented `numpy.ndarray`.
3.  **`amplitude_scaling_augmentation(sequences, scale_factor_range)`**: Applies random amplitude scaling to a batch of time-series sequences.
    *   `sequences`: Input `numpy.ndarray` of time series.
    *   `scale_factor_range`: Tuple `(min_scale, max_scale)` for random scaling.
    *   Returns: Augmented `numpy.ndarray`.
4.  **`build_lstm_model(input_shape, num_classes)`**: Creates a simple Long Short-Term Memory (LSTM) model for sequence classification.
    *   `input_shape`: Shape of input sequences (e.g., `(timesteps, 1)`).
    *   `num_classes`: Number of output classes (e.g., 1 for binary classification).
    *   Returns: `tf.keras.Model`.
5.  **`build_lstm_attention_model(input_shape, num_classes)`**: Creates an LSTM model enhanced with an attention mechanism for sequence classification. This will involve defining a custom Keras Layer for the attention mechanism.
    *   `input_shape`: Shape of input sequences.
    *   `num_classes`: Number of output classes.
    *   Returns: `tf.keras.Model`.
6.  **`AttentionLayer(tf.keras.layers.Layer)`**: Custom Keras layer for additive attention.
    *   `call(inputs)`: Computes attention weights and context vector.
    *   `get_attention_weights(inputs)`: Helper to extract attention weights.
7.  **`train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)`**: Trains a given Keras model.
    *   Returns: Trained `tf.keras.callbacks.History` object.

### Visualization Requirements

1.  **Trend Plot (Line Plot)**:
    *   Raw synthetic time-series data visualization.
    *   Comparison of original and augmented time-series samples.
    *   Visualization of attention weights overlayed on time-series samples.
    *   Style: `seaborn.lineplot` or `matplotlib.pyplot.plot`. Clear titles, labeled axes (Time, Value), legends. Color-blind friendly palette. Font size $\geq 12$ pt.
2.  **Relationship Plot (Scatter Plot)**:
    *   Visualization of `original_value` vs `augmented_value` to show the effect of augmentation.
    *   Style: `seaborn.scatterplot`. Clear titles, labeled axes, legends. Color-blind friendly palette. Font size $\geq 12$ pt.
3.  **Aggregated Comparison (Bar Plot)**:
    *   Comparison of model performance metrics (e.g., accuracy, loss) before and after applying data augmentation.
    *   Style: `seaborn.barplot`. Clear titles, labeled axes (Metric, Value), legends. Color-blind friendly palette. Font size $\geq 12$ pt.
4.  **Tables**:
    *   Displaying `pandas.DataFrame.head()`, `pandas.DataFrame.describe()`, and `pandas.DataFrame.info()`.

## 3. Notebook Sections (in detail)

### Section 1: Introduction to Sequence-based Explainable AI (XAI)

#### Markdown Cell: Explanation and Formulae

This section introduces the concept of Explainable AI (XAI) in the context of sequence data, such as time series. Understanding how an AI model arrives at a prediction is crucial for building trust, especially in critical applications. Sequence-based XAI focuses on methods that provide insights into models trained on sequential data, differing from image or tabular XAI by considering temporal dependencies.

We will explore two main categories of sequence-based XAI:
1.  **Pre-modeling Techniques**: Applied before model training, often involving data manipulation to improve interpretability or robustness. Data augmentation is a key technique here, where new training data is created by transforming existing samples.
2.  **In-modeling Techniques**: Integrated directly into the model architecture to make its internal workings more transparent. Attention mechanisms are a prime example, allowing the model to highlight the most relevant parts of an input sequence for a given prediction.

The ultimate goal is to provide transparency, interpretability, and comprehensibility of AI systems, as discussed in [1]. The paper [1] mentions how the impact of techniques on a model can be conceptualized by a general probabilistic relationship, which can be seen in the light of Bayes' theorem:
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$
Where:
*   $P(A|B)$ is the posterior probability of hypothesis $A$ given evidence $B$.
*   $P(B|A)$ is the likelihood of observing evidence $B$ given hypothesis $A$.
*   $P(A)$ is the prior probability of hypothesis $A$.
*   $P(B)$ is the marginal probability of evidence $B$.

In our context, $A$ could represent the model's parameters or its generalization ability, and $B$ could represent new, augmented data. This theorem conceptually illustrates how new data (evidence) can update our belief about the model's state or behavior. Practically, we will measure this "impact" through observable changes in model performance.

### Section 2: Setup and Library Imports

#### Code Cell: Function Implementation (None)

This cell will be used to import all necessary Python libraries.

#### Code Cell: Execution of Imports

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
```

#### Markdown Cell: Explanation for Execution

The required libraries (`numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn`, `tensorflow`) are imported. `numpy` and `pandas` are for data handling, `matplotlib.pyplot` and `seaborn` for visualization, `sklearn` for data splitting and preprocessing, and `tensorflow.keras` for building and training deep learning models, including custom layers.

### Section 3: Synthetic Time-Series Data Generation

#### Markdown Cell: Explanation

To effectively demonstrate sequence-based XAI without relying on complex external datasets, we will generate synthetic time-series data. This allows for controlled patterns and noise, making the effects of XAI techniques more discernible. Each synthetic sequence will follow a sinusoidal pattern with added noise and an optional linear trend. We will generate a binary classification target: 1 if the final value of the sequence is above a certain threshold, 0 otherwise.

#### Code Cell: Function Implementation

```python
# Function to generate synthetic time series data
def generate_synthetic_time_series_data(n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale, trend_slope, threshold_for_label):
    # This function generates time series data.
    # Each series has a sine wave pattern, random amplitude noise, general pattern noise, and a linear trend.
    # The target label is binary, determined by whether the last value of the series exceeds a threshold.
    pass # Implementation will go here
```

#### Code Cell: Execution of Function

```python
# Define generation parameters
N_SAMPLES = 1000
TIMESTEPS = 50
FREQUENCY = 0.1
AMPLITUDE_NOISE_SCALE = 0.2
PATTERN_NOISE_SCALE = 0.5
TREND_SLOPE = 0.05
THRESHOLD_FOR_LABEL = 0.5

# Generate data
X, y = generate_synthetic_time_series_data(N_SAMPLES, TIMESTEPS, FREQUENCY, AMPLITUDE_NOISE_SCALE, PATTERN_NOISE_SCALE, TREND_SLOPE, THRESHOLD_FOR_LABEL)
```

#### Markdown Cell: Explanation for Execution

The `generate_synthetic_time_series_data` function is executed with predefined parameters to create 1000 time-series sequences, each 50 time steps long. The target `y` is a binary label based on the characteristics of `X`. These parameters are chosen to create a discernible, yet noisy, pattern suitable for demonstrating XAI.

### Section 4: Dataset Overview and Initial Visualization

#### Markdown Cell: Explanation

Before applying any XAI techniques, it's crucial to understand the structure and characteristics of our synthetic dataset. This involves inspecting its dimensions, data types, and visualizing a few samples to grasp the underlying patterns and variability.

#### Code Cell: Function Implementation (None)

This section will directly use `numpy` and `pandas` functions for inspection and `matplotlib`/`seaborn` for visualization.

#### Code Cell: Execution of Overview and Plotting

```python
# Display dataset shape
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Flatten X for easier DataFrame conversion for initial inspection (if needed, but for time series we plot directly)
# Convert a few samples to DataFrame for descriptive statistics if desired, otherwise numpy descriptive stats suffice
# For time series, descriptive statistics per timestep can be misleading, overall summary or sample plots are better.
print("\nDescriptive statistics for a few time steps (e.g., first 5):")
print(pd.DataFrame(X[:, :5, 0]).describe()) # Display stats for first 5 time steps of all samples

# Plot a few sample time series
plt.figure(figsize=(12, 6))
sns.set_palette("colorblind") # Use a color-blind friendly palette
for i in range(5): # Plot 5 samples
    plt.plot(X[i, :, 0], label=f'Sample {i}, Label: {y[i]}', linewidth=1.5)
plt.title('Sample Synthetic Time Series Data', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```

#### Markdown Cell: Explanation for Execution

The code first prints the shape of the generated features ($X$) and labels ($y$) to confirm their dimensions. It then displays descriptive statistics for the first few time steps, offering a high-level view of the data distribution. A trend plot (`matplotlib.pyplot.plot` with `seaborn` palette) visualizes five random samples of the time series. This visualization helps in understanding the sinusoidal pattern, the presence of noise, and the overall variability across different sequences and their corresponding binary labels. The plots are generated with clear titles, labeled axes, legends, and a color-blind-friendly palette, with font sizes set to $\geq 12$ pt for titles and $\geq 10$ pt for labels/legends.

### Section 5: Pre-modeling XAI: Data Augmentation - Theory

#### Markdown Cell: Explanation

Data augmentation is a pre-modeling technique used to increase the diversity of the training data without actually collecting new samples. For sequence data, this involves applying various transformations such as adding noise, scaling, or time warping. The primary goal is to make the model more robust and generalize better to unseen data, which indirectly enhances its interpretability by making its predictions less sensitive to minor variations in input.

The reference [1] highlights data augmentation as a method to add complexity and improve self-supervised learning, specifically mentioning "adding rotations or flips, and noise additions to the images". For time series, a "rotation" can be analogous to a phase shift or cyclic shift in the sequence, while a "flip" might be inverting the amplitude or reversing the sequence. In this demonstrator, we will focus on adding Gaussian noise and applying amplitude scaling, as these are common and effective for time-series data. These techniques help the model learn more generalizable features rather than memorizing specific data points.

### Section 6: Pre-modeling XAI: Data Augmentation - Implementation

#### Markdown Cell: Explanation

We will implement two common data augmentation techniques suitable for time-series data: adding Gaussian noise and random amplitude scaling. These functions will be applied to the training data to create augmented versions.

#### Code Cell: Function Implementation

```python
# Function to add Gaussian noise to time series
def add_gaussian_noise_augmentation(sequences, noise_level):
    # Adds random Gaussian noise to each time series in the batch.
    pass # Implementation will go here

# Function to apply random amplitude scaling to time series
def amplitude_scaling_augmentation(sequences, scale_factor_range):
    # Applies a random scaling factor (within a given range) to the amplitude of each time series.
    pass # Implementation will go here
```

#### Code Cell: Execution of Function

```python
# Define augmentation parameters
NOISE_LEVEL = 0.1
SCALE_FACTOR_RANGE = (0.8, 1.2)

# Select a subset of the original data for demonstration
sample_indices = np.random.choice(N_SAMPLES, 100, replace=False)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

# Apply augmentations
X_augmented_noise = add_gaussian_noise_augmentation(X_sample, NOISE_LEVEL)
X_augmented_scale = amplitude_scaling_augmentation(X_sample, SCALE_FACTOR_RANGE)
```

#### Markdown Cell: Explanation for Execution

The `add_gaussian_noise_augmentation` function adds random noise to the time-series values, simulating natural data variability. The `amplitude_scaling_augmentation` function multiplies the entire sequence by a random factor within a specified range, altering the magnitude of the signal. These augmentations are applied to a subset of the original data for demonstration purposes, creating two different augmented versions.

### Section 7: Visualize Augmented Data

#### Markdown Cell: Explanation

Visualizing the augmented data against the original samples is essential to understand the transformations applied. This helps confirm that the augmentation techniques are working as intended and producing realistic variations of the original patterns.

#### Code Cell: Execution of Plotting

```python
# Plot original vs. noise-augmented samples
plt.figure(figsize=(14, 7))
sns.set_palette("colorblind")
plt.subplot(1, 2, 1)
plt.plot(X_sample[0, :, 0], label='Original', color='blue', linewidth=1.5)
plt.plot(X_augmented_noise[0, :, 0], label='Noise Augmented', color='red', linestyle='--', linewidth=1.5)
plt.title('Original vs. Noise Augmented Sample', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Plot original vs. scale-augmented samples
plt.subplot(1, 2, 2)
plt.plot(X_sample[1, :, 0], label='Original', color='blue', linewidth=1.5)
plt.plot(X_augmented_scale[1, :, 0], label='Scale Augmented', color='green', linestyle=':', linewidth=1.5)
plt.title('Original vs. Scale Augmented Sample', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Visualize relationship between original and augmented values using a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_sample[:, :, 0].flatten(), y=X_augmented_noise[:, :, 0].flatten(), alpha=0.3, color='purple')
plt.title('Relationship: Original vs. Noise Augmented Values', fontsize=14)
plt.xlabel('Original Value', fontsize=12)
plt.ylabel('Noise Augmented Value', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()
```

#### Markdown Cell: Explanation for Execution

Two line plots are generated: one comparing an original sample with its noise-augmented version, and another comparing an original sample with its amplitude-scaled version. These plots clearly show the subtle changes introduced by each augmentation technique. A scatter plot then illustrates the relationship between the original values and the noise-augmented values across all time steps in the sample, demonstrating the spread introduced by the noise. These visualizations confirm the expected impact of data augmentation.

### Section 8: Model Training (Pre-Augmentation)

#### Markdown Cell: Explanation

We will train a simple Long Short-Term Memory (LSTM) model to classify our synthetic time-series data. This model will serve as a baseline to evaluate the impact of data augmentation and the interpretability provided by attention mechanisms. The LSTM architecture is well-suited for sequence prediction tasks due to its ability to capture long-term dependencies.

#### Code Cell: Function Implementation

```python
# Function to build a simple LSTM model
def build_lstm_model(input_shape, num_classes):
    # Builds a sequential Keras model with an LSTM layer and a Dense output layer for binary classification.
    # Uses 'sigmoid' activation for binary classification.
    pass # Implementation will go here

# Helper function to preprocess and split data
def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    # Splits data into training and testing sets and applies StandardScaler to features.
    pass # Implementation will go here
```

#### Code Cell: Execution of Model Training

```python
# Preprocess and split the original data
X_scaled, X_train_orig, X_test_orig, y_train_orig, y_test_orig = preprocess_and_split(X, y)

# Build the baseline LSTM model
input_shape = (TIMESTEPS, 1)
baseline_model = build_lstm_model(input_shape, 1) # Binary classification

# Train the baseline model
print("Training baseline model (without augmentation)...")
history_baseline = train_model(baseline_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig, epochs=10, batch_size=32)

# Evaluate the baseline model
loss_baseline, accuracy_baseline = baseline_model.evaluate(X_test_orig, y_test_orig, verbose=0)
print(f"Baseline Model Test Accuracy: {accuracy_baseline:.4f}")
print(f"Baseline Model Test Loss: {loss_baseline:.4f}")
```

#### Markdown Cell: Explanation for Execution

The data is first preprocessed using `StandardScaler` and split into training and testing sets. A `tensorflow.keras.Sequential` model with an `LSTM` layer and a `Dense` output layer (with `sigmoid` activation for binary classification) is constructed. The `build_lstm_model` function encapsulates this. The model is then trained using `model.fit()` with `adam` optimizer and `binary_crossentropy` loss. After training, its performance on the test set (accuracy and loss) is evaluated and printed, establishing a baseline for comparison.

### Section 9: Model Training (Post-Augmentation)

#### Markdown Cell: Explanation

By training the model on an augmented dataset, we expect to see improvements in performance and robustness. Data augmentation helps the model encounter a wider variety of data variations, preventing overfitting to the original training samples and leading to better generalization.

#### Code Cell: Execution of Model Training

```python
# Generate augmented data for training
# Combine noise augmentation and amplitude scaling for more robust augmentation
X_train_augmented_noise = add_gaussian_noise_augmentation(X_train_orig, NOISE_LEVEL)
X_train_augmented_scale = amplitude_scaling_augmentation(X_train_orig, SCALE_FACTOR_RANGE)
X_train_augmented = np.concatenate((X_train_orig, X_train_augmented_noise, X_train_augmented_scale), axis=0)
y_train_augmented = np.concatenate((y_train_orig, y_train_orig, y_train_orig), axis=0) # Labels remain the same

# Shuffle the augmented data
shuffle_indices = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_indices]
y_train_augmented = y_train_augmented[shuffle_indices]

# Build and train a new LSTM model on augmented data
augmented_model = build_lstm_model(input_shape, 1)
print("\nTraining augmented model...")
history_augmented = train_model(augmented_model, X_train_augmented, y_train_augmented, X_test_orig, y_test_orig, epochs=10, batch_size=32)

# Evaluate the augmented model
loss_augmented, accuracy_augmented = augmented_model.evaluate(X_test_orig, y_test_orig, verbose=0)
print(f"Augmented Model Test Accuracy: {accuracy_augmented:.4f}")
print(f"Augmented Model Test Loss: {loss_augmented:.4f}")
```

#### Markdown Cell: Explanation for Execution

The `X_train_orig` data is augmented using both noise addition and amplitude scaling, and these augmented versions are concatenated with the original training data. The combined dataset is then shuffled. A new LSTM model (same architecture as baseline) is trained on this expanded and diversified dataset. The model's performance on the original (unaugmented) test set is then evaluated and printed. This allows for a direct comparison with the baseline model's performance, highlighting the benefits of data augmentation.

### Section 10: Compare Model Performance

#### Markdown Cell: Explanation

A direct comparison of the baseline model's performance (trained without augmentation) and the augmented model's performance (trained with augmentation) provides quantitative evidence of the effectiveness of pre-modeling XAI techniques. A bar plot is an ideal way to visualize these comparative metrics.

#### Code Cell: Execution of Comparison Plot

```python
# Create a DataFrame for comparison
performance_data = {
    'Model': ['Baseline', 'Augmented'],
    'Accuracy': [accuracy_baseline, accuracy_augmented],
    'Loss': [loss_baseline, loss_augmented]
}
df_performance = pd.DataFrame(performance_data)

# Plot comparison
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=df_performance, palette='viridis')
plt.title('Model Accuracy Comparison (Baseline vs. Augmented)', fontsize=14)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.ylim(0.5, 1.0) # Set a reasonable y-limit for accuracy
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Loss', data=df_performance, palette='plasma')
plt.title('Model Loss Comparison (Baseline vs. Augmented)', fontsize=14)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```

#### Markdown Cell: Explanation for Execution

Two bar plots are generated using `seaborn.barplot`. The first plot compares the test accuracy of the baseline and augmented models, and the second compares their test loss. These visualizations clearly show whether data augmentation successfully improved the model's generalization capabilities, typically indicated by higher accuracy and lower loss on the unseen test set.

### Section 11: In-modeling XAI: Attention Mechanism - Theory

#### Markdown Cell: Explanation

In-modeling XAI techniques integrate interpretability directly into the model's architecture. Attention mechanisms are a powerful example of this for sequence models. They allow a model to selectively focus on different parts of the input sequence when making a prediction, thereby providing a clear indication of which time steps were most influential.

The core idea is to compute "attention weights" for each element in the input sequence. These weights are then used to create a weighted sum of the sequence elements (or their hidden representations), forming a "context vector" that summarizes the most relevant information. This context vector is then used for the final prediction. By visualizing these attention weights, we can understand *why* the model made a particular decision by seeing *what* parts of the sequence it paid attention to.

A simplified conceptual representation of attention weight calculation involves:
1.  **Scoring**: Each hidden state $h_t$ of the input sequence is compared with a query vector $q$ (e.g., the last hidden state or a learned context vector) to get an alignment score $s_t$:
    $$ s_t = \text{score}(q, h_t) $$
2.  **Softmax**: These scores are then normalized using a softmax function to obtain attention weights $\alpha_t$, ensuring they sum to 1:
    $$ \alpha_t = \frac{\exp(s_t)}{\sum_{k=1}^T \exp(s_k)} $$
    Here, $\alpha_t$ is the attention weight for the hidden state at time $t$, and $T$ is the total number of time steps. These $\alpha_t$ values are what we will visualize to understand the model's focus.
3.  **Context Vector**: A context vector $c$ is computed as the weighted sum of the hidden states using the attention weights:
    $$ c = \sum_{t=1}^T \alpha_t h_t $$
    This context vector is then used by subsequent layers for the final prediction.

### Section 12: In-modeling XAI: Attention Mechanism - Implementation

#### Markdown Cell: Explanation

To implement an attention mechanism, we will create a custom Keras layer. This layer will take the output of an LSTM layer, calculate attention weights for each time step, and produce a context vector. This context vector, along with the attention weights, will then be used for the final classification.

#### Code Cell: Function Implementation

```python
# Custom Keras Attention Layer
class AttentionLayer(layers.Layer):
    # This custom layer implements an additive attention mechanism for sequence models.
    # It calculates attention scores, normalizes them into weights, and computes a context vector.
    # It also provides a method to retrieve the raw attention weights.
    pass # Implementation will go here

# Function to build an LSTM model with the custom Attention Layer
def build_lstm_attention_model(input_shape, num_classes):
    # Builds a sequential Keras model with an LSTM layer, followed by the custom AttentionLayer,
    # and a Dense output layer for binary classification.
    # The AttentionLayer will output both the context vector and the attention weights.
    pass # Implementation will go here
```

#### Code Cell: Execution of Model Training

```python
# Build the LSTM model with attention
attention_model = build_lstm_attention_model(input_shape, 1)

# Compile the attention model (using original split data for simplicity)
attention_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the attention model (using augmented data as it's better performing)
print("\nTraining LSTM model with attention...")
history_attention = train_model(attention_model, X_train_augmented, y_train_augmented, X_test_orig, y_test_orig, epochs=10, batch_size=32)

# Evaluate the attention model
loss_attention, accuracy_attention = attention_model.evaluate(X_test_orig, y_test_orig, verbose=0)
print(f"Attention Model Test Accuracy: {accuracy_attention:.4f}")
print(f"Attention Model Test Loss: {loss_attention:.4f}")
```

#### Markdown Cell: Explanation for Execution

The `AttentionLayer` is defined as a custom `tf.keras.layers.Layer`, which computes attention weights and a context vector from the LSTM outputs. The `build_lstm_attention_model` function integrates this custom layer into an LSTM-based model. This model is then compiled and trained using the augmented dataset. The performance of the attention model is evaluated, showing its accuracy and loss, which should be comparable to or better than the augmented model without explicit attention.

### Section 13: Visualize Attention Weights

#### Markdown Cell: Explanation

Visualizing the attention weights allows us to understand which parts of a time series were most important for the model's prediction. A trend plot of the time series, with the attention weights highlighted (e.g., as an overlay or by varying line thickness/color intensity), provides intuitive interpretability. High attention weights indicate regions the model focused on, providing a "reason" for its output.

#### Code Cell: Execution of Attention Visualization

```python
# Select a sample from the test set for visualization
sample_idx = 0 # Choose the first sample from the test set for demonstration
test_sample = X_test_orig[sample_idx]
true_label = y_test_orig[sample_idx]

# Reshape sample for model prediction
test_sample_reshaped = np.expand_dims(test_sample, axis=0)

# Get attention weights from the trained model
# The AttentionLayer outputs both context_vector and attention_weights.
# We need to create a temporary model to extract the attention weights.
attention_extractor = keras.Model(inputs=attention_model.inputs,
                                  outputs=attention_model.get_layer('attention_layer_name').output[1]) # Assuming layer name 'attention_layer_name' and output[1] is weights

attention_weights = attention_extractor.predict(test_sample_reshaped)[0] # Extract the first (and only) sample's weights
predicted_label_prob = attention_model.predict(test_sample_reshaped)[0][0]
predicted_label = (predicted_label_prob > 0.5).astype(int)

# Plot the time series with attention weights
plt.figure(figsize=(12, 6))
sns.set_palette("colorblind")
plt.plot(test_sample[:, 0], label=f'Time Series Sample (True Label: {true_label})', color='blue', linewidth=2)
# Overlay attention weights as a shaded area or secondary y-axis
plt.twinx() # Create a second y-axis
plt.fill_between(range(TIMESTEPS), attention_weights, color='red', alpha=0.3, label='Attention Weights')
plt.plot(attention_weights, color='red', linestyle='--', linewidth=1, label='Attention Weights Line')
plt.title(f'Time Series Sample with Attention Weights (Predicted Label: {predicted_label}, Prob: {predicted_label_prob:.2f})', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Attention Weight', color='red', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```

#### Markdown Cell: Explanation for Execution

A specific sample from the test set is selected. The `attention_extractor` model (created to specifically output the attention weights from the `AttentionLayer`) is used to predict and retrieve the attention weights for this sample. The predicted label and probability are also obtained. A trend plot (`matplotlib.pyplot.plot`) then displays the time series sample, with the attention weights overlaid as a shaded area and a line on a secondary y-axis. This visualization clearly shows which time steps were deemed most important by the model for its prediction, providing direct interpretability. For example, if the model predicts a high label based on the peak value, the attention weights should be high around that peak.

### Section 14: Summary and Conclusion

#### Markdown Cell: Explanation

This notebook demonstrated key aspects of sequence-based Explainable AI using synthetic time-series data. We explored both pre-modeling and in-modeling techniques.

*   **Pre-modeling XAI (Data Augmentation)**: We observed how simple techniques like adding noise and amplitude scaling can enhance a model's robustness and generalization capabilities, leading to improved performance on unseen data. This highlights the indirect interpretability gained through building more reliable models.
*   **In-modeling XAI (Attention Mechanisms)**: We integrated an attention layer into our LSTM model. By visualizing the attention weights, we gained direct insights into which parts of the time series sequence were most influential for the model's predictions. This provides a transparent "reason" for the model's output, addressing the "black box" problem in AI.

By combining these approaches, users can not only build better-performing models but also understand the rationale behind their decisions, fostering trust and enabling more informed application of AI systems.

### Section 15: References

#### Markdown Cell: Explanation

This section provides credits to the external document and libraries used in this notebook.

#### Code Cell: Execution of References (None)

```markdown
**References:**

[1] Long, B., Liu, E., Qiu, R., & Duan, Y. (2023). Explainable AI – the Latest Advancements and New Trends. *IEEE Access*, 11, 1-13. (Original provided document)

**Libraries Used:**

*   `numpy`: A fundamental package for numerical computing with Python.
*   `pandas`: A fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool.
*   `matplotlib`: A comprehensive library for creating static, animated, and interactive visualizations in Python.
*   `seaborn`: A Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
*   `scikit-learn`: A free software machine learning library for the Python programming language.
*   `tensorflow`: An open-source machine learning framework developed by Google.
```
