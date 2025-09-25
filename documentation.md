id: 68d54eae89fb95725ebc3d08_documentation
summary: Explainable AI Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Explainable AI for Sequence Data with Streamlit

## 1. Introduction to Explainable AI for Sequence Data and Synthetic Data Generation
Duration: 00:15:00

<aside class="positive">
In this first step, we'll set the stage for our exploration into **Explainable AI (XAI)**, specifically tailored for **sequence data**. This foundational understanding is crucial because it provides context for why these techniques are vital, especially when dealing with time-series or sequential information in sensitive domains like finance, healthcare, and industrial IoT. We will introduce the core concepts, outline the application's architecture, and guide you through generating the synthetic data we'll use throughout the lab.
</aside>

### 1.1. The Importance of Explainable AI for Sequence Data

Artificial Intelligence models, particularly deep learning networks, are often perceived as "black boxes" due to their complex internal structures. While highly effective, their lack of transparency can be a major hurdle in real-world adoption, particularly in fields where trust, accountability, and regulatory compliance are paramount. For models processing sequence data (like time series, natural language, or genomic sequences), understanding *why* a prediction was made is even more challenging due to temporal dependencies and the nuanced influence of specific time steps.

This QuLab application demonstrates practical **pre-modeling** and **in-modeling** XAI techniques for sequence data:

1.  **Pre-modeling Techniques (Data Augmentation)**: Applied *before* training, these methods enhance data robustness and implicitly improve interpretability by making models less sensitive to minor variations. Data augmentation helps models learn more generalized features.
2.  **In-modeling Techniques (Attention Mechanisms)**: Integrated *within* the model architecture, these mechanisms provide direct insights into the model's decision-making process by highlighting relevant parts of the input sequence.

The application conceptually relates the impact of these techniques to probabilistic relationships, similar to Bayes' theorem: $ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $. Here, $A$ could be the model's generalization ability, and $B$ could be the augmented data, illustrating how new data updates our understanding of the model's behavior. We'll practically measure this impact through observable changes in model performance.

### 1.2. Application Architecture Overview

The QuLab Streamlit application is structured into three main pages, accessible via a sidebar navigation. This modular design allows us to progress through the XAI concepts step-by-step. Data and model states are persistently shared across these pages using Streamlit's `st.session_state`.

Here's a conceptual flow of how the application works:

1.  **`app.py` (Main Orchestrator)**: Sets up the Streamlit page configuration, displays the main title, and manages the sidebar navigation. It dynamically loads the content for `page1.py`, `page2.py`, or `page3.py` based on the user's selection.
2.  **`application_pages/page1.py` (Data Generation)**: Focuses on introducing XAI concepts and generating synthetic time-series data based on user-defined parameters. It stores the generated data (`X`, `y`, `TIMESTEPS`) in `st.session_state`.
3.  **`application_pages/page2.py` (Data Augmentation & Baseline Model)**: Retrieves data from `st.session_state`, applies various data augmentation techniques, visualizes their effects, and then trains a baseline LSTM model on the *original* (non-augmented) training data. It stores the trained model and its performance metrics in `st.session_state`.
4.  **`application_pages/page3.py` (Attention Mechanisms & Model Comparison)**: Utilizes the original training data and augmentation parameters to train a new model on *augmented* data. It also introduces and trains an LSTM model enhanced with an attention mechanism. Finally, it provides visualizations to compare the performance of all models and to interpret the attention weights.

### 1.3. Setup and Library Imports

The application leverages standard Python libraries for numerical operations (`numpy`), data manipulation (`pandas`), interactive visualizations (`plotly`), machine learning utilities (`sklearn`), and deep learning (`tensorflow`). These are imported as needed within each `page.py` file.

```python
# From app.py:
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

# ... (intro markdown) ...

page = st.sidebar.selectbox(label="Navigation", options=["1. Introduction & Data Generation", "2. Data Augmentation & Baseline Model", "3. Attention Mechanisms & Model Comparison"])
if page == "1. Introduction & Data Generation":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "2. Data Augmentation & Baseline Model":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "3. Attention Mechanisms & Model Comparison":
    from application_pages.page3 import run_page3
    run_page3()
```

### 1.4. Synthetic Time-Series Data Generation

To provide a controlled environment for demonstrating XAI techniques, we will generate synthetic time-series data. Each sequence follows a sinusoidal pattern, with added noise and an optional linear trend. A binary classification label is assigned based on whether the final value of the sequence exceeds a predefined threshold.

The `generate_synthetic_time_series_data` function, defined in `application_pages/page1.py`, creates these sequences:

```python
# From application_pages/page1.py:
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def generate_synthetic_time_series_data(n_samples, timesteps, frequency, amplitude_noise_scale, pattern_noise_scale, trend_slope, threshold_for_label):
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
```

In the Streamlit application, you can adjust parameters such as the number of samples, timesteps, frequency, and various noise scales from the sidebar on the "1. Introduction & Data Generation" page.

<aside class="positive">
<b>Action:</b> Navigate to the "1. Introduction & Data Generation" page in the Streamlit application. Adjust the "Synthetic Data Generation Parameters" in the sidebar if you wish, and then click the "Generate Synthetic Data" button to create your dataset. This data will be stored in the session state for subsequent steps.
</aside>

### 1.5. Dataset Overview and Initial Visualization

After generating the synthetic data, the application provides an overview of its shape, descriptive statistics, and a visualization of a few sample sequences. This step is crucial for understanding the data's characteristics before applying any XAI techniques.

```python
# Snippet from application_pages/page1.py demonstrating data overview
if 'X' in st.session_state and 'y' in st.session_state:
    X = st.session_state['X']
    y = st.session_state['y']
    
    st.subheader("Dataset Shape")
    st.write(f"Features shape: {X.shape}") # e.g., (1000, 50, 1)
    st.write(f"Labels shape: {y.shape}")   # e.g., (1000,)

    # ... (descriptive statistics) ...

    st.subheader("Sample Synthetic Time Series Data")
    if X.shape[0] > 0 and X.shape[1] > 0:
        num_samples_to_plot = min(5, X.shape[0])
        fig = go.Figure()
        for i in range(num_samples_to_plot):
            fig.add_trace(go.Scatter(x=np.arange(X.shape[1]), y=X[i, :, 0], mode='lines',
                                     name=f'Sample {i}, Label: {y[i]}'))
        
        fig.update_layout(
            title='Sample Synthetic Time Series Data',
            xaxis_title='Time Step',
            yaxis_title='Value',
            font=dict(size=12),
            title_font_size=16,
            legend_font_size=12,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No synthetic data to plot. Please generate data first.")
```
The visualization, rendered using `plotly.graph_objects.Figure`, uses clear titles ($\\geq 16$ pt) and labels ($\\geq 12$ pt) and a color-blind-friendly palette to highlight individual series and their labels. This helps to visualize the sinusoidal patterns, the effect of noise, and the binary classification target.

## 2. Pre-modeling XAI: Data Augmentation and Baseline Model Training
Duration: 00:20:00

<aside class="positive">
In this step, we will dive into **pre-modeling XAI** by implementing and visualizing **data augmentation** techniques for our synthetic time-series data. Data augmentation is a powerful strategy to improve model robustness and generalization. We will also train a **baseline LSTM model** using the original data, which will serve as a crucial reference point for evaluating the impact of augmentation and, later, attention mechanisms.
</aside>

### 2.1. Pre-modeling XAI: Data Augmentation - Theory

Data augmentation is a technique to artificially expand the training dataset by creating modified versions of existing data points. For sequential data, this can involve adding noise, scaling amplitudes, shifting, or time warping. The goal is not just to increase data quantity but to introduce variability that the model might encounter in real-world scenarios, thereby making the model more robust and less prone to overfitting specific training examples.

From an XAI perspective, a more robust model is often implicitly more interpretable. If a model's prediction is stable across minor variations of an input, it suggests the model has learned generalizable features rather than brittle, context-specific patterns.

The application focuses on two common techniques:
*   **Adding Gaussian Noise**: Simulates minor sensor errors or natural fluctuations.
*   **Amplitude Scaling**: Varies the magnitude of the signal, mimicking different intensity levels of a phenomenon.

### 2.2. Pre-modeling XAI: Data Augmentation - Implementation

The application implements `add_gaussian_noise_augmentation` and `amplitude_scaling_augmentation` functions. These functions apply the specified transformations to batches of sequences.

```python
# Snippet from application_pages/page2.py:
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def add_gaussian_noise_augmentation(sequences, noise_level):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=sequences.shape)
    augmented_sequences = sequences + noise.astype(sequences.dtype)
    return augmented_sequences

def amplitude_scaling_augmentation(sequences, scale_factor_range):
    batch_size = sequences.shape[0]
    scaling_factor_shape = (batch_size,) + (1,) * (sequences.ndim - 1)
    scaling_factors = np.random.uniform(scale_factor_range[0], scale_factor_range[1], size=scaling_factor_shape).astype(sequences.dtype)
    augmented_sequences = sequences * scaling_factors
    return augmented_sequences

# ... (inside run_page2) ...
if 'X' in st.session_state:
    # Take a sample of original data for augmentation demonstration
    X_sample_indices = np.random.choice(st.session_state['X'].shape[0], min(st.session_state['X'].shape[0], 100), replace=False)
    X_sample_for_aug = st.session_state['X'][X_sample_indices]
    
    # Apply augmentations based on sidebar parameters
    X_augmented_noise = add_gaussian_noise_augmentation(X_sample_for_aug, NOISE_LEVEL)
    X_augmented_scale = amplitude_scaling_augmentation(X_sample_for_aug, SCALE_FACTOR_RANGE)

    st.session_state['X_sample_for_aug'] = X_sample_for_aug
    st.session_state['X_augmented_noise'] = X_augmented_noise
    st.session_state['X_augmented_scale'] = X_augmented_scale
else:
    st.warning("Please generate synthetic data on the 'Introduction & Data Generation' page first.")
```

<aside class="positive">
<b>Action:</b> Navigate to the "2. Data Augmentation & Baseline Model" page. Observe the "Data Augmentation Parameters" in the sidebar. These parameters (`Gaussian Noise Level`, `Min Amplitude Scale Factor`, `Max Amplitude Scale Factor`) control the strength of the augmentation. You'll see augmented data visualizations below, which automatically update as you adjust these parameters.
</aside>

### 2.3. Visualize Augmented Data

To intuitively understand the effect of data augmentation, the application visualizes samples of augmented data against their original counterparts. This helps confirm that the transformations are applied as intended and produce realistic variations.

```python
# Snippet from application_pages/page2.py (visualization)
if 'X_sample_for_aug' in st.session_state and st.session_state['X_sample_for_aug'].shape[0] > 0:
    sample_idx = st.sidebar.slider("Select Sample for Augmentation Visualization", 0, st.session_state['X_sample_for_aug'].shape[0] - 1, 0, key='aug_sample_idx')
    
    X_sample_val = st.session_state['X_sample_for_aug']
    X_augmented_noise_val = st.session_state['X_augmented_noise']
    X_augmented_scale_val = st.session_state['X_augmented_scale']

    st.subheader("Original vs. Noise Augmented Sample")
    # Plotly figure for noise augmented data
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_sample_val[sample_idx, :, 0], mode='lines', name='Original', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_augmented_noise_val[sample_idx, :, 0], mode='lines', name='Noise Augmented', line=dict(color='red', dash='dash')))
    fig1.update_layout(title='Original vs. Noise Augmented Sample', xaxis_title='Time Step', yaxis_title='Value', font=dict(size=12), title_font_size=16, legend_font_size=12, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Original vs. Scale Augmented Sample")
    # Plotly figure for scale augmented data
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_sample_val[sample_idx, :, 0], mode='lines', name='Original', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_augmented_scale_val[sample_idx, :, 0], mode='lines', name='Scale Augmented', line=dict(color='green', dash='dot')))
    fig2.update_layout(title='Original vs. Scale Augmented Sample', xaxis_title='Time Step', yaxis_title='Value', font=dict(size=12), title_font_size=16, legend_font_size=12, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Relationship: Original vs. Noise Augmented Values")
    # Scatter plot for noise augmentation effect
    df_scatter = pd.DataFrame({
        'Original Value': X_sample_val[:, :, 0].flatten(),
        'Noise Augmented Value': X_augmented_noise_val[:, :, 0].flatten()
    })
    fig3 = px.scatter(df_scatter, x='Original Value', y='Noise Augmented Value', opacity=0.3,
                      title='Relationship: Original vs. Noise Augmented Values',
                      labels={'Original Value': 'Original Value', 'Noise Augmented Value': 'Noise Augmented Value'},
                      color_discrete_sequence=px.colors.qualitative.Plotly
                    )
    fig3.update_layout(font=dict(size=12), title_font_size=16, hovermode="closest")
    st.plotly_chart(fig3, use_container_width=True)
```
The line plots compare an original sequence with its noise-augmented and amplitude-scaled versions, respectively, showing the subtle differences. The scatter plot further illustrates the distribution of noise by comparing original values against noise-augmented values across all time steps. These visualizations highlight the impact of the augmentation parameters.

### 2.4. Model Training (Pre-Augmentation)

Before exploring the benefits of augmentation, we need to establish a performance baseline. This involves training a standard Long Short-Term Memory (LSTM) model on the *original*, non-augmented synthetic data.

The data first undergoes preprocessing:
1.  **Splitting**: `X` and `y` are split into training and testing sets using `train_test_split`.
2.  **Scaling**: `StandardScaler` is applied to normalize the feature values, which is crucial for deep learning models.

The baseline LSTM model is built using `tensorflow.keras.Sequential` with an LSTM layer (64 units) and a Dense output layer with `sigmoid` activation for binary classification. It is compiled with the `adam` optimizer and `binary_crossentropy` loss.

```python
# Snippet from application_pages/page2.py:
def preprocess_and_split(X, y, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    original_shape = X.shape
    X_reshaped_for_scaler = X.reshape(-1, original_shape[-1])
    X_scaled_all_reshaped = scaler.fit_transform(X_reshaped_for_scaler)
    X_scaled_all = X_scaled_all_reshaped.reshape(original_shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_all, y, test_size=test_size, random_state=random_state
    )
    return X_scaled_all, X_train, X_test, y_train, y_test

def build_lstm_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(units=num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0 # Suppress detailed output in Streamlit
    )
    return history

# ... (inside run_page2, for training baseline) ...
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
        st.warning("Please generate synthetic data on the 'Introduction & Data Generation' page first.")
```

<aside class="positive">
<b>Action:</b> In the sidebar on the "2. Data Augmentation & Baseline Model" page, adjust "Model Training Parameters" (Epochs, Batch Size) if desired. Click "Train Baseline Model" to train the model on the original data. This will save its performance for later comparison.
</aside>

## 3. Attention Mechanisms & Model Comparison
Duration: 00:25:00

<aside class="positive">
In this final step, we will explore the impact of **data augmentation** on model performance and then dive into **in-modeling XAI** with **attention mechanisms**. We will train an LSTM model on augmented data, compare its performance to the baseline, and then build and visualize an attention-based LSTM model to understand how it focuses on different parts of the input sequence.
</aside>

### 3.1. Model Training (Post-Augmentation)

After understanding and visualizing data augmentation, we now train a new LSTM model using an augmented training dataset. This dataset is created by combining the original training data with its noise-augmented and amplitude-scaled versions. This step demonstrates the practical application and benefits of pre-modeling XAI.

```python
# Snippet from application_pages/page3.py (training augmented model)
# Augmentation functions are re-defined or imported from page2.py
# build_lstm_model, preprocess_and_split, train_model are also re-defined/imported

# ... (inside run_page3) ...
if st.sidebar.button("Train Augmented Model", key='train_augmented_button'):
    if 'X_train_orig' in st.session_state and 'y_train_orig' in st.session_state and 'X_test_orig' in st.session_state and 'y_test_orig' in st.session_state and 'TIMESTEPS' in st.session_state:
        with st.spinner("Generating augmented data and training model..."):
            X_train_orig = st.session_state['X_train_orig']
            y_train_orig = st.session_state['y_train_orig']
            X_test_orig = st.session_state['X_test_orig']
            y_test_orig = st.session_state['y_test_orig']

            # Apply augmentations and concatenate
            X_train_augmented_noise = add_gaussian_noise_augmentation(X_train_orig, NOISE_LEVEL)
            X_train_augmented_scale = amplitude_scaling_augmentation(X_train_orig, SCALE_FACTOR_RANGE)
            X_train_augmented = np.concatenate((X_train_orig, X_train_augmented_noise, X_train_augmented_scale), axis=0)
            y_train_augmented = np.concatenate((y_train_orig, y_train_orig, y_train_orig), axis=0)

            # Shuffle the combined dataset
            shuffle_indices = np.random.permutation(len(X_train_augmented))
            X_train_augmented = X_train_augmented[shuffle_indices]
            y_train_augmented = y_train_augmented[shuffle_indices]

            input_shape = (st.session_state['TIMESTEPS'], 1)
            augmented_model = build_lstm_model(input_shape, 1) # Same architecture as baseline
            history_augmented = train_model(augmented_model, X_train_augmented, y_train_augmented, X_test_orig, y_test_orig, epochs=EPOCHS, batch_size=BATCH_SIZE)
            loss_augmented, accuracy_augmented = augmented_model.evaluate(X_test_orig, y_test_orig, verbose=0)

            st.session_state['augmented_model'] = augmented_model
            st.session_state['accuracy_augmented'] = accuracy_augmented
            st.session_state['loss_augmented'] = loss_augmented

        st.success("Augmented model training complete!")
        st.write(f"Augmented Model Test Accuracy: {st.session_state['accuracy_augmented']:.4f}")
        st.write(f"Augmented Model Test Loss: {st.session_state['loss_augmented']:.4f}")
    else:
        st.warning("Please train the baseline model on the 'Data Augmentation & Baseline Model' page first.")
```

<aside class="positive">
<b>Action:</b> Navigate to the "3. Attention Mechanisms & Model Comparison" page. Click "Train Augmented Model" in the sidebar. This will train a new LSTM model on the expanded dataset.
</aside>

### 3.2. Compare Model Performance

To quantitatively assess the impact of data augmentation, the application compares the performance metrics (accuracy and loss) of the baseline model (trained on original data) and the augmented model (trained on augmented data).

```python
# Snippet from application_pages/page3.py (model comparison)
if 'accuracy_baseline' in st.session_state and 'accuracy_augmented' in st.session_state:
    performance_data = {
        'Model': ['Baseline', 'Augmented'],
        'Accuracy': [st.session_state['accuracy_baseline'], st.session_state['accuracy_augmented']],
        'Loss': [st.session_state['loss_baseline'], st.session_state['loss_augmented']]
    }
    df_performance = pd.DataFrame(performance_data)

    st.subheader("Model Accuracy Comparison")
    fig_acc = px.bar(df_performance, x='Model', y='Accuracy', title='Model Accuracy Comparison (Baseline vs. Augmented)',
                    color='Model', color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_acc.update_layout(xaxis_title='Model Type', yaxis_title='Test Accuracy', yaxis_range=[0.5, 1.0], font=dict(size=12), title_font_size=16, legend_font_size=12)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader("Model Loss Comparison")
    fig_loss = px.bar(df_performance, x='Model', y='Loss', title='Model Loss Comparison (Baseline vs. Augmented)',
                    color='Model', color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_loss.update_layout(xaxis_title='Model Type', yaxis_title='Test Loss', font=dict(size=12), title_font_size=16, legend_font_size=12)
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Please train both baseline and augmented models to compare performance (on this and the previous page).")
```
The bar plots generated by `plotly.express` clearly show how data augmentation (a pre-modeling XAI technique) impacts model performance, typically leading to higher accuracy and lower loss, demonstrating improved generalization.

### 3.3. In-modeling XAI: Attention Mechanism - Theory

Attention mechanisms are a core **in-modeling XAI** technique that enhances interpretability by allowing a neural network to focus selectively on specific parts of its input sequence. Instead of treating all time steps equally, attention assigns "weights" to each time step, indicating its relative importance for a given prediction.

The process typically involves:
1.  **Scoring**: A score $s_t$ is computed for each hidden state $h_t$ of the sequence, often by comparing it against a query vector or a learnable parameter. This score indicates how "relevant" $h_t$ is.
    $$ s_t = \text{score}(h_t, \text{query}) $$
2.  **Softmax Normalization**: These scores are then normalized using a softmax function across all time steps to produce attention weights $\alpha_t$. These weights sum to 1, ensuring a clear distribution of focus.
    $$ \alpha_t = \frac{\exp(s_t)}{\sum_{k=1}^T \exp(s_k)} $$
    Where $T$ is the total number of time steps.
3.  **Context Vector**: A context vector $c$ is computed as a weighted sum of the hidden states, using the attention weights. This vector summarizes the most relevant information from the input sequence.
    $$ c = \sum_{t=1}^T \alpha_t h_t $$
This context vector then feeds into the final prediction layer. By visualizing $\alpha_t$, we can directly observe which parts of the input sequence the model considered most important for its decision.

### 3.4. In-modeling XAI: Attention Mechanism - Implementation

To integrate attention, we define a custom Keras layer, `AttentionLayer`. This layer takes the output of the LSTM (which returns sequences of hidden states), calculates attention scores using a dense layer, applies softmax to get weights, and then produces a context vector.

```python
# Snippet from application_pages/page3.py:
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W_a = None

    def build(self, input_shape):
        self.W_a = tf.keras.layers.Dense(1, use_bias=True, name="attention_score_dense")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)
        scores = self.W_a(inputs) # scores shape: (batch_size, timesteps, 1)
        attention_weights_raw = tf.nn.softmax(scores, axis=1)
        attention_weights = tf.squeeze(attention_weights_raw, axis=-1) # attention_weights shape: (batch_size, timesteps)
        
        weighted_inputs = inputs * tf.expand_dims(attention_weights, axis=-1) # (batch_size, timesteps, features)
        context_vector = tf.reduce_sum(weighted_inputs, axis=1) # (batch_size, features)

        return context_vector, attention_weights

    # Method to retrieve attention weights directly for visualization
    def get_attention_weights(self, inputs):
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        scores = self.W_a(inputs_tensor)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=1)
        return attention_weights.numpy()

def build_lstm_attention_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    lstm_output = layers.LSTM(units=64, return_sequences=True, name="lstm_layer")(inputs) # LSTM returns full sequence
    attention_output, attention_weights = AttentionLayer(name="attention_layer")(lstm_output) # Attention gets LSTM sequence output
    outputs = layers.Dense(
        units=num_classes,
        activation="sigmoid" if num_classes == 1 else "softmax",
        name="output_layer"
    )(attention_output) # Dense layer gets context vector
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ... (inside run_page3, for training attention model) ...
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
        st.warning("Please train the baseline model on the 'Data Augmentation & Baseline Model' page first to ensure data is prepared.")
```

<aside class="positive">
<b>Action:</b> In the sidebar on the "3. Attention Mechanisms & Model Comparison" page, click "Train Attention Model". This will train the LSTM model equipped with the custom attention layer.
</aside>

### 3.5. Attention Mechanism Visualization

After training the attention model, the application provides an interactive visualization of the attention weights for a selected test sample. This allows you to directly observe which time steps contributed most to the model's prediction, offering valuable insights into its decision-making process.

```python
# Snippet from application_pages/page3.py (attention visualization)
if 'attention_model' in st.session_state and 'X_test_orig' in st.session_state:
    X_test_orig = st.session_state['X_test_orig']
    y_test_orig = st.session_state['y_test_orig']
    attention_model = st.session_state['attention_model']

    if X_test_orig.shape[0] > 0:
        sample_index_attention = st.slider("Select Sample for Attention Visualization", 0, X_test_orig.shape[0] - 1, 0, key='att_sample_idx')
        
        # Create an intermediate model to extract attention weights
        attention_layer_output = attention_model.get_layer("attention_layer").output
        attention_weights_output = attention_layer_output[1] 

        intermediate_model = keras.Model(
            inputs=attention_model.input,
            outputs=[attention_model.get_layer("lstm_layer").output, attention_weights_output]
        )
        
        lstm_hidden_states, attention_weights = intermediate_model.predict(X_test_orig[[sample_index_attention]])
        attention_weights = attention_weights.flatten()

        fig_att = go.Figure()
        fig_att.add_trace(go.Scatter(x=np.arange(X_test_orig.shape[1]), y=X_test_orig[sample_index_attention, :, 0], mode='lines',
                                     name=f'Original Series (Label: {y_test_orig[sample_index_attention]})', line=dict(color='blue')))
        fig_att.add_trace(go.Bar(x=np.arange(X_test_orig.shape[1]), y=attention_weights,
                                 name='Attention Weights', marker_color='orange', opacity=0.5))
        
        fig_att.update_layout(
            title='Attention Weights for a Sample Prediction',
            xaxis_title='Time Step',
            yaxis_title='Value / Attention Weight',
            font=dict(size=12),
            title_font_size=16,
            legend_font_size=12,
            hovermode="x unified"
        )
        st.plotly_chart(fig_att, use_container_width=True)
    else:
        st.info("No test data available for attention visualization. Please ensure synthetic data is generated.")
else:
    st.info("Please train the Attention Model to visualize attention weights. Click 'Train Attention Model' in the sidebar.")
```
The interactive plot combines the original time series with a bar chart representing the attention weights at each time step. A higher bar indicates that the model paid more "attention" to that particular time step when making its prediction. This direct visual feedback demystifies the model's internal workings and highlights the most influential parts of the sequence.

By exploring all three models (baseline, augmented, and attention-based), you can gain a comprehensive understanding of how different XAI techniques contribute to both performance and interpretability in sequence data analysis.
