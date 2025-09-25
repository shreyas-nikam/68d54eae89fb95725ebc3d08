
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will provide an interactive demonstration of sequence-based Explainable AI (XAI) techniques, specifically focusing on pre-modeling (data augmentation) and in-modeling (attention mechanisms) approaches. Users will be able to generate synthetic time-series data, apply augmentation, train and compare models, and visualize attention weights for interpretability.

**Learning Goals:**
*   Understand the fundamental concepts of sequence-based Explainable AI and how it differs from other XAI methods.
*   Experiment with pre-modeling XAI techniques, specifically data augmentation for time-series data, and observe its impact on model robustness and performance.
*   Apply in-modeling XAI techniques, such as attention mechanisms, to a time-series prediction model to enhance interpretability by identifying influential time steps.
*   Visualize and interpret attention weights to gain insights into model predictions for sequence data.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will use a two-column layout:
*   **Sidebar (`st.sidebar`):** Dedicated to input widgets for data generation, augmentation, and model training parameters.
*   **Main Content Area:** Displays narrative markdown, data visualizations, model performance metrics, and attention mechanism visualizations. Sections will be organized clearly with headers.

### Input Widgets and Controls
All interactive controls will reside in the sidebar.

1.  **Synthetic Time-Series Data Generation Parameters:**
    *   `n_samples`: `st.slider` (e.g., 100 to 5000, default 1000)
    *   `timesteps`: `st.slider` (e.g., 20 to 100, default 50)
    *   `frequency`: `st.number_input` (e.g., 0.01 to 0.5, step 0.01, default 0.1)
    *   `amplitude_noise_scale`: `st.number_input` (e.g., 0.0 to 1.0, step 0.05, default 0.2)
    *   `pattern_noise_scale`: `st.number_input` (e.g., 0.0 to 1.0, step 0.05, default 0.5)
    *   `trend_slope`: `st.number_input` (e.g., -0.1 to 0.1, step 0.01, default 0.05)
    *   `threshold_for_label`: `st.number_input` (e.g., -2.0 to 2.0, step 0.1, default 0.5)
    *   **Action:** `st.button("Generate Synthetic Data")`

2.  **Pre-modeling XAI: Data Augmentation Parameters:**
    *   `noise_level`: `st.number_input` (e.g., 0.0 to 0.5, step 0.01, default 0.1)
    *   `scale_factor_range`: Two `st.number_input` for (min_scale, max_scale) (e.g., min: 0.5 to 1.0, max: 1.0 to 1.5, defaults 0.8, 1.2)
    *   A selector for choosing a sample index to visualize augmented data: `st.slider` or `st.selectbox`.

3.  **Model Training Parameters:**
    *   `epochs`: `st.number_input` (e.g., 5 to 50, step 1, default 10)
    *   `batch_size`: `st.number_input` (e.g., 16 to 128, step 16, default 32)
    *   **Action:** `st.button("Train Baseline Model")`
    *   **Action:** `st.button("Train Augmented Model")`
    *   **Action:** `st.button("Train Attention Model")`

4.  **Attention Mechanism Visualization:**
    *   A selector for choosing a sample index to visualize attention: `st.slider` or `st.selectbox`.

### Visualization Components
All visualizations will be displayed in the main content area.

1.  **Dataset Overview and Initial Visualization:**
    *   **Plot:** Line plot showing 5 sample synthetic time series (matplotlib/seaborn).
        *   **Title:** 'Sample Synthetic Time Series Data' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Time Step' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Value' (fontsize $\geq 12$ pt)
        *   **Legend:** Displaying sample index and label (fontsize $\geq 10$ pt)
        *   **Style:** Color-blind-friendly palette.
    *   **Text:** Display of features and labels shapes, and descriptive statistics for initial time steps.

2.  **Visualize Augmented Data:**
    *   **Plot 1:** Line plot comparing an original sample with its noise-augmented version.
        *   **Title:** 'Original vs. Noise Augmented Sample' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Time Step' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Value' (fontsize $\geq 12$ pt)
        *   **Legend:** 'Original', 'Noise Augmented' (fontsize $\geq 10$ pt)
        *   **Style:** Color-blind-friendly palette.
    *   **Plot 2:** Line plot comparing an original sample with its amplitude-scaled version.
        *   **Title:** 'Original vs. Scale Augmented Sample' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Time Step' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Value' (fontsize $\geq 12$ pt)
        *   **Legend:** 'Original', 'Scale Augmented' (fontsize $\geq 10$ pt)
        *   **Style:** Color-blind-friendly palette.
    *   **Plot 3:** Scatter plot showing the relationship between original and noise-augmented values.
        *   **Title:** 'Relationship: Original vs. Noise Augmented Values' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Original Value' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Noise Augmented Value' (fontsize $\geq 12$ pt)
        *   **Style:** Color-blind-friendly palette.

3.  **Compare Model Performance:**
    *   **Plot 1:** Bar plot comparing Baseline and Augmented Model Test Accuracy.
        *   **Title:** 'Model Accuracy Comparison (Baseline vs. Augmented)' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Model Type' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Test Accuracy' (fontsize $\geq 12$ pt)
        *   **Y-limit:** 0.5 to 1.0.
        *   **Style:** Color-blind-friendly palette.
    *   **Plot 2:** Bar plot comparing Baseline and Augmented Model Test Loss.
        *   **Title:** 'Model Loss Comparison (Baseline vs. Augmented)' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Model Type' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Test Loss' (fontsize $\geq 12$ pt)
        *   **Style:** Color-blind-friendly palette.

4.  **Attention Mechanism Visualization:**
    *   **Plot:** Line plot of a selected sample time series, with attention weights visualized (e.g., as a heatmap overlay or color intensity on the line plot).
        *   **Title:** 'Attention Weights for a Sample Prediction' (fontsize $\geq 14$ pt)
        *   **X-axis:** 'Time Step' (fontsize $\geq 12$ pt)
        *   **Y-axis:** 'Value / Attention Weight' (fontsize $\geq 12$ pt)
        *   **Legend:** 'Time Series Value', 'Attention Weights' (fontsize $\geq 10$ pt)
        *   **Style:** Color-blind-friendly palette.

### Interactive Elements and Feedback Mechanisms
*   All plots should be interactive where possible (e.g., using `st.pyplot` with `matplotlib` or `altair` if preferred for interactivity, though `matplotlib` is used in notebook).
*   Progress indicators (`st.spinner`, `st.progress`) will be used during data generation and model training.
*   Informative messages (`st.info`, `st.success`, `st.error`) will be displayed for user feedback (e.g., "Data generated successfully!", "Training complete.").
*   Display of model evaluation metrics (accuracy, loss) after training.

## 3. Additional Requirements

*   **Annotation and Tooltip Specifications:**
    *   Each input widget in the sidebar will include inline help text (`st.info` or `st.help`) or tooltips (`st.sidebar.info` with a descriptive message) to explain its purpose.
    *   Markdown sections will include narrative descriptions as provided in the Jupyter Notebook to explain concepts and what is being demonstrated.

*   **Save the states of the fields properly so that changes are not lost:**
    *   Streamlit's `st.session_state` will be utilized to store the values of all input widgets (sliders, number inputs) and results of computations (generated data, trained models, performance metrics). This ensures that user selections persist across reruns and interactions.

## 4. Notebook Content and Code Requirements

This section extracts the narrative and code stubs from the Jupyter Notebook, outlining how they will be integrated into the Streamlit application.

### Markdown: Introduction
```python
import streamlit as st

st.title("Sequence-based XAI Demonstrator")

st.header("1. Introduction to Sequence-based Explainable AI (XAI)")
st.markdown("""
This Streamlit application provides a hands-on demonstration of sequence-based Explainable AI (XAI) techniques, focusing on pre-modeling and in-modeling approaches. Understanding how an AI model arrives at a prediction is crucial for building trust, especially in critical applications. Sequence-based XAI focuses on methods that provide insights into models trained on sequential data, differing from image or tabular XAI by considering temporal dependencies.

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
""")

st.subheader("Learning Goals")
st.markdown("""
*   Understand the fundamental concepts of sequence-based Explainable AI and how it differs from other XAI methods.
*   Experiment with pre-modeling XAI techniques, specifically data augmentation for time-series data, and observe its impact on model robustness and performance.
*   Apply in-modeling XAI techniques, such as attention mechanisms, to a time-series prediction model to enhance interpretability by identifying influential time steps.
*   Visualize and interpret attention weights to gain insights into model predictions for sequence data.
""")
```

### Markdown: Setup and Library Imports
```python
st.header("2. Setup and Library Imports")
st.markdown("""
This section handles the installation of required libraries and then imports them for use throughout the notebook.
""")

st.subheader("Code for Library Imports")
st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf # Required for custom AttentionLayer
""", language="python")

st.markdown("""
The required libraries (`numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn`, `tensorflow`) are imported. `numpy` and `pandas` are for data handling, `matplotlib.pyplot` and `seaborn` for visualization, `sklearn` for data splitting and preprocessing, and `tensorflow.keras` for building and training deep learning models, including custom layers.
""")
```

### Markdown: Synthetic Time-Series Data Generation
```python
st.header("3. Synthetic Time-Series Data Generation")
st.markdown("""
To effectively demonstrate sequence-based XAI without relying on complex external datasets, we will generate synthetic time-series data. This allows for controlled patterns and noise, making the effects of XAI techniques more discernible. Each synthetic sequence will follow a sinusoidal pattern with added noise and an optional linear trend. We will generate a binary classification target: 1 if the final value of the sequence is above a certain threshold, 0 otherwise.
""")

st.subheader("Code for Data Generation Function")
st.code("""
def generate_synthetic_time_series_data(n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale, trend_slope, threshold_for_label):
    if not isinstance(n_samples, int) or n_samples < 0:
        raise TypeError("n_samples must be a non-negative integer.")
    if not isinstance(timesteps, int) or timesteps < 0:
        raise TypeError("timesteps must be a non-negative integer.")
    
    if n_samples == 0:
        return np.empty((0, timesteps, 1), dtype=np.float64), np.empty((0,), dtype=int)

    X = np.empty((n_samples, timesteps, 1), dtype=np.float64)
    y = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        if timesteps == 0:
            y[i] = 0
            continue 

        t = np.arange(timesteps, dtype=np.float64)
        base_sin_wave = np.sin(2 * np.pi * frequency * t)
        current_amplitude = 1.0 + np.random.normal(0, amplitude_noise_scale)
        amplitude_modulated_wave = current_amplitude * base_sin_wave
        trend = trend_slope * t
        series = amplitude_modulated_wave + trend
        series += np.random.normal(0, pattern_noise_scale, timesteps)
        X[i, :, 0] = series
        y[i] = 1 if series[-1] > threshold_for_label else 0
    
    return X, y
""", language="python")

# Streamlit UI for parameters (in sidebar)
st.sidebar.header("Synthetic Data Generation")
N_SAMPLES = st.sidebar.slider("Number of Samples", 100, 5000, 1000, key='n_samples_gen')
TIMESTEPS = st.sidebar.slider("Number of Timesteps", 20, 100, 50, key='timesteps_gen')
FREQUENCY = st.sidebar.number_input("Frequency", 0.01, 0.5, 0.1, step=0.01, key='frequency_gen')
AMPLITUDE_NOISE_SCALE = st.sidebar.number_input("Amplitude Noise Scale", 0.0, 1.0, 0.2, step=0.05, key='amp_noise_scale_gen')
PATTERN_NOISE_SCALE = st.sidebar.number_input("Pattern Noise Scale", 0.0, 1.0, 0.5, step=0.05, key='pat_noise_scale_gen')
TREND_SLOPE = st.sidebar.number_input("Trend Slope", -0.1, 0.1, 0.05, step=0.01, key='trend_slope_gen')
THRESHOLD_FOR_LABEL = st.sidebar.number_input("Threshold for Label", -2.0, 2.0, 0.5, step=0.1, key='threshold_gen')

if st.sidebar.button("Generate Synthetic Data", key='generate_data_button'):
    with st.spinner("Generating data..."):
        X, y = generate_synthetic_time_series_data(N_SAMPLES, TIMESTEPS, FREQUENCY, AMPLITUDE_NOISE_SCALE, PATTERN_NOISE_SCALE, TREND_SLOPE, THRESHOLD_FOR_LABEL)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['TIMESTEPS'] = TIMESTEPS # Store TIMESTEPS for later model building
    st.success("Synthetic data generated!")
else:
    # Ensure X and y are initialized if button not pressed yet
    if 'X' not in st.session_state:
        X, y = generate_synthetic_time_series_data(N_SAMPLES, TIMESTEPS, FREQUENCY, AMPLITUDE_NOISE_SCALE, PATTERN_NOISE_SCALE, TREND_SLOPE, THRESHOLD_FOR_LABEL)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['TIMESTEPS'] = TIMESTEPS

st.markdown("""
The `generate_synthetic_time_series_data` function is executed with predefined parameters to create 1000 time-series sequences, each 50 time steps long. The target $y$ is a binary label based on the characteristics of $X$. These parameters are chosen to create a discernible, yet noisy, pattern suitable for demonstrating XAI.
""")
```

### Markdown: Dataset Overview and Initial Visualization
```python
st.header("4. Dataset Overview and Initial Visualization")
st.markdown("""
Before applying any XAI techniques, it's crucial to understand the structure and characteristics of our synthetic dataset. This involves inspecting its dimensions, data types, and visualizing a few samples to grasp the underlying patterns and variability.
""")

if 'X' in st.session_state and 'y' in st.session_state:
    X = st.session_state['X']
    y = st.session_state['y']
    
    st.subheader("Dataset Shape")
    st.write(f"Features shape: {X.shape}")
    st.write(f"Labels shape: {y.shape}")

    st.subheader("Descriptive Statistics")
    if X.shape[0] > 0 and X.shape[1] > 0:
        st.write("Descriptive statistics for a few time steps (e.g., first 5):")
        st.dataframe(pd.DataFrame(X[:, :min(5, X.shape[1]), 0]).describe())
    else:
        st.info("No data or timesteps to display descriptive statistics.")

    st.subheader("Sample Synthetic Time Series Data")
    if X.shape[0] > 0 and X.shape[1] > 0:
        plt.figure(figsize=(12, 6))
        sns.set_palette("colorblind")
        num_samples_to_plot = min(5, X.shape[0])
        for i in range(num_samples_to_plot):
            plt.plot(X[i, :, 0], label=f'Sample {i}, Label: {y[i]}', linewidth=1.5)
        plt.title('Sample Synthetic Time Series Data', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(plt)
    else:
        st.info("No synthetic data to plot. Please generate data first.")
else:
    st.info("Please generate synthetic data using the controls in the sidebar.")

st.markdown("""
The code first prints the shape of the generated features ($X$) and labels ($y$) to confirm their dimensions. It then displays descriptive statistics for the first few time steps, offering a high-level view of the data distribution. A trend plot (`matplotlib.pyplot.plot` with `seaborn` palette) visualizes five random samples of the time series. This visualization helps in understanding the sinusoidal pattern, the presence of noise, and the overall variability across different sequences and their corresponding binary labels. The plots are generated with clear titles, labeled axes, legends, and a color-blind-friendly palette, with font sizes set to $\geq 12$ pt for titles and $\geq 10$ pt for labels/legends.
""")
```

### Markdown: Pre-modeling XAI: Data Augmentation - Theory
```python
st.header("5. Pre-modeling XAI: Data Augmentation - Theory")
st.markdown("""
Data augmentation is a pre-modeling technique used to increase the diversity of the training data without actually collecting new samples. For sequence data, this involves applying various transformations such as adding noise, scaling, or time warping. The primary goal is to make the model more robust and generalize better to unseen data, which indirectly enhances its interpretability by making its predictions less sensitive to minor variations in input.

The reference [1] highlights data augmentation as a method to add complexity and improve self-supervised learning, specifically mentioning "adding rotations or flips, and noise additions to the images". For time series, a "rotation" can be analogous to a phase shift or cyclic shift in the sequence, while a "flip" might be inverting the amplitude or reversing the sequence. In this demonstrator, we will focus on adding Gaussian noise and applying amplitude scaling, as these are common and effective for time-series data. These techniques help the model learn more generalizable features rather than memorizing specific data points.
""")
```

### Markdown: Pre-modeling XAI: Data Augmentation - Implementation
```python
st.header("6. Pre-modeling XAI: Data Augmentation - Implementation")
st.markdown("""
We will implement two common data augmentation techniques suitable for time-series data: adding Gaussian noise and random amplitude scaling. These functions will be applied to the training data to create augmented versions.
""")

st.subheader("Code for Data Augmentation Functions")
st.code("""
def add_gaussian_noise_augmentation(sequences, noise_level):
    if not isinstance(sequences, np.ndarray):
        raise TypeError("Input 'sequences' must be a numpy.ndarray.")
    
    if not isinstance(noise_level, (int, float)):
        raise TypeError("Input 'noise_level' must be a numeric type (int or float).")
    
    noise = np.random.normal(loc=0.0, scale=noise_level, size=sequences.shape)
    augmented_sequences = sequences + noise.astype(sequences.dtype)
    
    return augmented_sequences

def amplitude_scaling_augmentation(sequences, scale_factor_range):
    if not isinstance(sequences, np.ndarray):
        raise TypeError("Input 'sequences' must be a numpy.ndarray.")

    if not (isinstance(scale_factor_range, tuple) and len(scale_factor_range) == 2):
        raise TypeError("'scale_factor_range' must be a tuple of two numeric values (min_scale, max_scale).")

    min_scale, max_scale = scale_factor_range
    if not (isinstance(min_scale, (int, float)) and isinstance(max_scale, (int, float))):
        raise TypeError("Elements of 'scale_factor_range' must be numeric (int or float).")

    if min_scale > max_scale:
        raise ValueError("min_scale cannot be greater than max_scale in 'scale_factor_range'.")

    if sequences.shape[0] == 0:
        return sequences

    batch_size = sequences.shape[0]
    scaling_factor_shape = (batch_size,) + (1,) * (sequences.ndim - 1)
    scaling_factors = np.random.uniform(min_scale, max_scale, size=scaling_factor_shape).astype(sequences.dtype)
    augmented_sequences = sequences * scaling_factors

    return augmented_sequences
""", language="python")

st.sidebar.header("Data Augmentation Parameters")
NOISE_LEVEL = st.sidebar.number_input("Gaussian Noise Level", 0.0, 0.5, 0.1, step=0.01, key='noise_level_aug')
MIN_SCALE_FACTOR = st.sidebar.number_input("Min Amplitude Scale Factor", 0.5, 1.0, 0.8, step=0.05, key='min_scale_factor_aug')
MAX_SCALE_FACTOR = st.sidebar.number_input("Max Amplitude Scale Factor", 1.0, 1.5, 1.2, step=0.05, key='max_scale_factor_aug')
SCALE_FACTOR_RANGE = (MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)

if 'X' in st.session_state:
    X_sample_indices = np.random.choice(st.session_state['X'].shape[0], min(st.session_state['X'].shape[0], 100), replace=False)
    X_sample_for_aug = st.session_state['X'][X_sample_indices]
    
    X_augmented_noise = add_gaussian_noise_augmentation(X_sample_for_aug, NOISE_LEVEL)
    X_augmented_scale = amplitude_scaling_augmentation(X_sample_for_aug, SCALE_FACTOR_RANGE)

    st.session_state['X_sample_for_aug'] = X_sample_for_aug
    st.session_state['X_augmented_noise'] = X_augmented_noise
    st.session_state['X_augmented_scale'] = X_augmented_scale

st.markdown("""
The `add_gaussian_noise_augmentation` function adds random noise to the time-series values, simulating natural data variability. The `amplitude_scaling_augmentation` function multiplies the entire sequence by a random factor within a specified range, altering the magnitude of the signal. These augmentations are applied to a subset of the original data for demonstration purposes, creating two different augmented versions.
""")
```

### Markdown: Visualize Augmented Data
```python
st.header("7. Visualize Augmented Data")
st.markdown("""
Visualizing the augmented data against the original samples is essential to understand the transformations applied. This helps confirm that the augmentation techniques are working as intended and producing realistic variations of the original patterns.
""")

if 'X_sample_for_aug' in st.session_state and st.session_state['X_sample_for_aug'].shape[0] > 0:
    sample_idx = st.sidebar.slider("Select Sample for Augmentation Visualization", 0, st.session_state['X_sample_for_aug'].shape[0] - 1, 0, key='aug_sample_idx')

    X_sample_val = st.session_state['X_sample_for_aug']
    X_augmented_noise_val = st.session_state['X_augmented_noise']
    X_augmented_scale_val = st.session_state['X_augmented_scale']

    st.subheader("Original vs. Augmented Samples")
    plt.figure(figsize=(14, 7))
    sns.set_palette("colorblind")
    
    plt.subplot(1, 2, 1)
    plt.plot(X_sample_val[sample_idx, :, 0], label='Original', color='blue', linewidth=1.5)
    plt.plot(X_augmented_noise_val[sample_idx, :, 0], label='Noise Augmented', color='red', linestyle='--', linewidth=1.5)
    plt.title('Original vs. Noise Augmented Sample', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.subplot(1, 2, 2)
    plt.plot(X_sample_val[sample_idx, :, 0], label='Original', color='blue', linewidth=1.5)
    plt.plot(X_augmented_scale_val[sample_idx, :, 0], label='Scale Augmented', color='green', linestyle=':', linewidth=1.5)
    plt.title('Original vs. Scale Augmented Sample', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("Relationship: Original vs. Noise Augmented Values")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_sample_val[:, :, 0].flatten(), y=X_augmented_noise_val[:, :, 0].flatten(), alpha=0.3, color='purple')
    plt.title('Relationship: Original vs. Noise Augmented Values', fontsize=14)
    plt.xlabel('Original Value', fontsize=12)
    plt.ylabel('Noise Augmented Value', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    st.pyplot(plt)
else:
    st.info("No augmented data to visualize. Please ensure synthetic data is generated.")

st.markdown("""
Two line plots are generated: one comparing an original sample with its noise-augmented version, and another comparing an original sample with its amplitude-scaled version. These plots clearly show the subtle changes introduced by each augmentation technique. A scatter plot then illustrates the relationship between the original values and the noise-augmented values across all time steps in the sample, demonstrating the spread introduced by the noise. These visualizations confirm the expected impact of data augmentation.
""")
```

### Markdown: Model Training (Pre-Augmentation)
```python
st.header("8. Model Training (Pre-Augmentation)")
st.markdown("""
We will train a simple Long Short-Term Memory (LSTM) model to classify our synthetic time-series data. This model will serve as a baseline to evaluate the impact of data augmentation and the interpretability provided by attention mechanisms. The LSTM architecture is well-suited for sequence prediction tasks due to its ability to capture long-term dependencies.
""")

st.subheader("Code for Model Building and Training Functions")
st.code("""
def build_lstm_model(input_shape, num_classes):
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(
            f"num_classes must be a positive integer, but got {num_classes}. "
            "For binary classification with sigmoid, num_classes should be 1."
        )
    if num_classes != 1:
        raise ValueError(
            f"This function is designed for binary classification with a 'sigmoid' output, "
            f"thus num_classes must be 1. Got {num_classes}."
        )

    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(units=num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray")

    scaler = StandardScaler()
    # Reshape X to 2D for StandardScaler, then back to 3D after scaling
    original_shape = X.shape
    X_reshaped_for_scaler = X.reshape(-1, original_shape[-1])
    X_scaled_all_reshaped = scaler.fit_transform(X_reshaped_for_scaler)
    X_scaled_all = X_scaled_all_reshaped.reshape(original_shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_all, y, test_size=test_size, random_state=random_state
    )
    return X_scaled_all, X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    if X_train is None or y_train is None:
        raise ValueError("Training data cannot be None.")
    if X_val is None or y_val is None:
        raise ValueError("Validation data cannot be None.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    if X_train.size == 0 or y_train.size == 0 or X_val.size == 0 or y_val.size == 0:
        raise ValueError("Input arrays should not be empty.")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val must have the same number of samples.")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history
""", language="python")

st.sidebar.header("Model Training Parameters")
EPOCHS = st.sidebar.number_input("Epochs", 5, 50, 10, step=1, key='epochs_train')
BATCH_SIZE = st.sidebar.number_input("Batch Size", 16, 128, 32, step=16, key='batch_size_train')

if st.sidebar.button("Train Baseline Model", key='train_baseline_button'):
    if 'X' in st.session_state and 'y' in st.session_state and 'TIMESTEPS' in st.session_state:
        with st.spinner("Preprocessing and training baseline model..."):
            X_scaled, X_train_orig, X_test_orig, y_train_orig, y_test_orig = preprocess_and_split(st.session_state['X'], st.session_state['y'])
            
            input_shape = (st.session_state['TIMESTEPS'], 1)
            baseline_model = build_lstm_model(input_shape, 1)

            history_baseline = train_model(baseline_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig, epochs=EPOCHS, batch_size=BATCH_SIZE)
            loss_baseline, accuracy_baseline = baseline_model.evaluate(X_test_orig, y_test_orig, verbose=0)
            
            st.session_state['baseline_model'] = baseline_model
            st.session_state['X_train_orig'] = X_train_orig
            st.session_state['X_test_orig'] = X_test_orig
            st.session_state['y_train_orig'] = y_train_orig
            st.session_state['y_test_orig'] = y_test_orig
            st.session_state['accuracy_baseline'] = accuracy_baseline
            st.session_state['loss_baseline'] = loss_baseline

        st.success("Baseline model training complete!")
        st.write(f"Baseline Model Test Accuracy: {st.session_state['accuracy_baseline']:.4f}")
        st.write(f"Baseline Model Test Loss: {st.session_state['loss_baseline']:.4f}")
    else:
        st.warning("Please generate synthetic data first.")

st.markdown("""
The data is first preprocessed using `StandardScaler` and split into training and testing sets. A `tensorflow.keras.Sequential` model with an `LSTM` layer and a `Dense` output layer (with `sigmoid` activation for binary classification) is constructed. The `build_lstm_model` function encapsulates this. The model is then trained using `model.fit()` with `adam` optimizer and `binary_crossentropy` loss. After training, its performance on the test set (accuracy and loss) is evaluated and printed, establishing a baseline for comparison.
""")
```

### Markdown: Model Training (Post-Augmentation)
```python
st.header("9. Model Training (Post-Augmentation)")
st.markdown("""
By training the model on an augmented dataset, we expect to see improvements in performance and robustness. Data augmentation helps the model encounter a wider variety of data variations, preventing overfitting to the original training samples and leading to better generalization.
""")

if st.sidebar.button("Train Augmented Model", key='train_augmented_button'):
    if 'X_train_orig' in st.session_state and 'y_train_orig' in st.session_state and 'X_test_orig' in st.session_state and 'y_test_orig' in st.session_state and 'TIMESTEPS' in st.session_state:
        with st.spinner("Generating augmented data and training model..."):
            X_train_orig = st.session_state['X_train_orig']
            y_train_orig = st.session_state['y_train_orig']
            X_test_orig = st.session_state['X_test_orig']
            y_test_orig = st.session_state['y_test_orig']

            X_train_augmented_noise = add_gaussian_noise_augmentation(X_train_orig, NOISE_LEVEL)
            X_train_augmented_scale = amplitude_scaling_augmentation(X_train_orig, SCALE_FACTOR_RANGE)
            X_train_augmented = np.concatenate((X_train_orig, X_train_augmented_noise, X_train_augmented_scale), axis=0)
            y_train_augmented = np.concatenate((y_train_orig, y_train_orig, y_train_orig), axis=0)

            shuffle_indices = np.random.permutation(len(X_train_augmented))
            X_train_augmented = X_train_augmented[shuffle_indices]
            y_train_augmented = y_train_augmented[shuffle_indices]

            input_shape = (st.session_state['TIMESTEPS'], 1)
            augmented_model = build_lstm_model(input_shape, 1)
            history_augmented = train_model(augmented_model, X_train_augmented, y_train_augmented, X_test_orig, y_test_orig, epochs=EPOCHS, batch_size=BATCH_SIZE)
            loss_augmented, accuracy_augmented = augmented_model.evaluate(X_test_orig, y_test_orig, verbose=0)

            st.session_state['augmented_model'] = augmented_model
            st.session_state['accuracy_augmented'] = accuracy_augmented
            st.session_state['loss_augmented'] = loss_augmented

        st.success("Augmented model training complete!")
        st.write(f"Augmented Model Test Accuracy: {st.session_state['accuracy_augmented']:.4f}")
        st.write(f"Augmented Model Test Loss: {st.session_state['loss_augmented']:.4f}")
    else:
        st.warning("Please train the baseline model first.")

st.markdown("""
The `X_train_orig` data is augmented using both noise addition and amplitude scaling, and these augmented versions are concatenated with the original training data. The combined dataset is then shuffled. A new LSTM model (same architecture as baseline) is trained on this expanded and diversified dataset. The model's performance on the original (unaugmented) test set is then evaluated and printed. This allows for a direct comparison with the baseline model's performance, highlighting the benefits of data augmentation.
""")
```

### Markdown: Compare Model Performance
```python
st.header("10. Compare Model Performance")
st.markdown("""
A direct comparison of the baseline model's performance (trained without augmentation) and the augmented model's performance (trained with augmentation) provides quantitative evidence of the effectiveness of pre-modeling XAI techniques. A bar plot is an ideal way to visualize these comparative metrics.
""")

if 'accuracy_baseline' in st.session_state and 'accuracy_augmented' in st.session_state:
    performance_data = {
        'Model': ['Baseline', 'Augmented'],
        'Accuracy': [st.session_state['accuracy_baseline'], st.session_state['accuracy_augmented']],
        'Loss': [st.session_state['loss_baseline'], st.session_state['loss_augmented']]
    }
    df_performance = pd.DataFrame(performance_data)

    st.subheader("Model Accuracy Comparison")
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Accuracy', data=df_performance, palette='viridis')
    plt.title('Model Accuracy Comparison (Baseline vs. Augmented)', fontsize=14)
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(plt)

    st.subheader("Model Loss Comparison")
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Loss', data=df_performance, palette='plasma')
    plt.title('Model Loss Comparison (Baseline vs. Augmented)', fontsize=14)
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(plt)
else:
    st.info("Please train both baseline and augmented models to compare performance.")
```

### Markdown: In-modeling XAI: Attention Mechanism - Theory
```python
st.header("11. In-modeling XAI: Attention Mechanism - Theory")
st.markdown("""
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
""")
```

### Markdown: In-modeling XAI: Attention Mechanism - Implementation
```python
st.header("12. In-modeling XAI: Attention Mechanism - Implementation")
st.markdown("""
To implement an attention mechanism, we will create a custom Keras layer. This layer will take the output of an LSTM layer, calculate attention weights for each time step, and produce a context vector. This context vector, along with the attention weights, will then be used for the final classification.
""")

st.subheader("Code for Custom Attention Layer and Attention Model Building")
st.code("""
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W_a = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Input to AttentionLayer must be 3D (batch_size, timesteps, features). "
                f"Received input shape: {input_shape}"
            )
        self.W_a = tf.keras.layers.Dense(1, use_bias=True, name="attention_score_dense")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        if inputs.shape.rank != 3:
            raise ValueError(
                f"Input tensor to AttentionLayer must have rank 3 (batch_size, timesteps, features). "
                f"Got input with rank {inputs.shape.rank} and shape {inputs.shape}."
            )
        scores = self.W_a(inputs)
        attention_weights_raw = tf.nn.softmax(scores, axis=1)
        attention_weights = tf.squeeze(attention_weights_raw, axis=-1)
        weighted_inputs = inputs * tf.expand_dims(attention_weights, axis=-1)
        context_vector = tf.reduce_sum(weighted_inputs, axis=1)

        return context_vector, attention_weights

    def get_attention_weights(self, inputs):
        if not isinstance(inputs, (np.ndarray, tf.Tensor)):
            raise TypeError(f"Input must be a numpy array or a TensorFlow tensor, but got type {type(inputs).__name__}.")

        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)

        if inputs_tensor.ndim != 3:
            raise ValueError(f"Input must have 3 dimensions (batch_size, timesteps, features), but got {inputs_tensor.ndim}.")

        batch_size, timesteps, features = inputs_tensor.shape

        if timesteps == 0:
            return np.empty((batch_size, 0), dtype=np.float32)

        # To get the attention scores, we need to apply W_a (if built).
        # This requires `self.W_a` to be built and called.
        # For simplicity and to match the notebook's `get_attention_weights` which uses `tf.reduce_sum`
        # in the provided stub (which might be a simplified mock for the example),
        # we'll adhere to that. A more accurate layer would use self.W_a here.
        if hasattr(self, 'W_a') and self.W_a.built:
             scores = self.W_a(inputs_tensor)
             scores = tf.squeeze(scores, axis=-1)
        else: # Fallback for unbuilt W_a or simplified stub
             scores = tf.reduce_sum(inputs_tensor, axis=-1)

        attention_weights = tf.nn.softmax(scores, axis=1)
        return attention_weights.numpy()


def build_lstm_attention_model(input_shape, num_classes):
    if not isinstance(input_shape, (tuple, list)):
        raise TypeError(f"input_shape must be a tuple or list, but got {type(input_shape)}")
    if len(input_shape) != 2:
        raise ValueError(
            f"input_shape must be a tuple of (timesteps, features), "
            f"but got {input_shape} with length {len(input_shape)}"
        )
    if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
        raise ValueError(
            f"Both timesteps and features in input_shape must be positive integers, "
            f"but got {input_shape}"
        )

    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, but got {num_classes}")

    inputs = keras.Input(shape=input_shape)
    lstm_output = layers.LSTM(units=64, return_sequences=True, name="lstm_layer")(inputs)
    attention_output, attention_weights = AttentionLayer(name="attention_layer")(lstm_output)
    outputs = layers.Dense(
        units=num_classes,
        activation="sigmoid" if num_classes == 1 else "softmax",
        name="output_layer"
    )(attention_output)
    
    model = keras.Model(inputs=inputs, outputs=outputs) # Build Functional Model to access intermediate layers
    
    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model
""", language="python")

if st.sidebar.button("Train Attention Model", key='train_attention_button'):
    if 'X_train_orig' in st.session_state and 'y_train_orig' in st.session_state and 'X_test_orig' in st.session_state and 'y_test_orig' in st.session_state and 'TIMESTEPS' in st.session_state:
        with st.spinner("Training attention model..."):
            X_train_orig = st.session_state['X_train_orig']
            y_train_orig = st.session_state['y_train_orig']
            X_test_orig = st.session_state['X_test_orig']
            y_test_orig = st.session_state['y_test_orig']

            input_shape = (st.session_state['TIMESTEPS'], 1)
            attention_model = build_lstm_attention_model(input_shape, 1)
            history_attention = train_model(attention_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig, epochs=EPOCHS, batch_size=BATCH_SIZE)
            loss_attention, accuracy_attention = attention_model.evaluate(X_test_orig, y_test_orig, verbose=0)

            st.session_state['attention_model'] = attention_model
            st.session_state['accuracy_attention'] = accuracy_attention
            st.session_state['loss_attention'] = loss_attention

        st.success("Attention model training complete!")
        st.write(f"Attention Model Test Accuracy: {st.session_state['accuracy_attention']:.4f}")
        st.write(f"Attention Model Test Loss: {st.session_state['loss_attention']:.4f}")
    else:
        st.warning("Please train the baseline model first to ensure data is prepared.")

st.subheader("Attention Mechanism Visualization")
if 'attention_model' in st.session_state and 'X_test_orig' in st.session_state:
    X_test_orig = st.session_state['X_test_orig']
    y_test_orig = st.session_state['y_test_orig']
    attention_model = st.session_state['attention_model']

    if X_test_orig.shape[0] > 0:
        sample_index_attention = st.slider("Select Sample for Attention Visualization", 0, X_test_orig.shape[0] - 1, 0, key='att_sample_idx')
        
        # To get attention weights, we need to access the output of the AttentionLayer.
        # Since build_lstm_attention_model returns a Functional Model, we can create
        # a sub-model that outputs the attention weights.
        attention_layer_output = attention_model.get_layer("attention_layer").output
        # The AttentionLayer returns (context_vector, attention_weights)
        attention_weights_output = attention_layer_output[1] 

        # Create a model that outputs LSTM hidden states and attention weights
        intermediate_model = keras.Model(
            inputs=attention_model.input,
            outputs=[attention_model.get_layer("lstm_layer").output, attention_weights_output]
        )
        
        lstm_hidden_states, attention_weights = intermediate_model.predict(X_test_orig[[sample_index_attention]])
        
        # Flatten attention_weights if necessary (should be (1, timesteps) already)
        attention_weights = attention_weights.flatten()

        plt.figure(figsize=(12, 6))
        sns.set_palette("viridis") # Using a sequential palette for attention

        # Plot the original time series
        plt.plot(X_test_orig[sample_index_attention, :, 0], label=f'Original Series (Label: {y_test_orig[sample_index_attention]})', color='blue', linewidth=1.5)
        
        # Overlay attention weights as a bar plot or fill between
        plt.bar(range(X_test_orig.shape[1]), attention_weights * np.max(X_test_orig[sample_index_attention, :, 0]), 
                alpha=0.3, color='orange', label='Attention Weights (scaled)')
        
        plt.title('Attention Weights for a Sample Prediction', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value / Scaled Attention Weight', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(plt)
    else:
        st.info("No test data available for attention visualization.")
else:
    st.info("Please train the Attention Model to visualize attention weights.")
```
