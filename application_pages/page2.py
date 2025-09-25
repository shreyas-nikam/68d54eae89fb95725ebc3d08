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


def run_page2():
    st.title("2. Data Augmentation & Baseline Model")

    st.header("2.1. Pre-modeling XAI: Data Augmentation - Theory")
    st.markdown("""
    Data augmentation is a pre-modeling technique used to increase the diversity of the training data without actually collecting new samples. For sequence data, this involves applying various transformations such as adding noise, scaling, or time warping. The primary goal is to make the model more robust and generalize better to unseen data, which indirectly enhances its interpretability by making its predictions less sensitive to minor variations in input.

    The reference [1] highlights data augmentation as a method to add complexity and improve self-supervised learning, specifically mentioning "adding rotations or flips, and noise additions to the images". For time series, a "rotation" can be analogous to a phase shift or cyclic shift in the sequence, while a "flip" might be inverting the amplitude or reversing the sequence. In this demonstrator, we will focus on adding Gaussian noise and applying amplitude scaling, as these are common and effective for time-series data. These techniques help the model learn more generalizable features rather than memorizing specific data points.
    """)

    st.header("2.2. Pre-modeling XAI: Data Augmentation - Implementation")
    st.markdown("""
    We will implement two common data augmentation techniques suitable for time-series data: adding Gaussian noise and random amplitude scaling. These functions will be applied to the training data to create augmented versions.

    The `add_gaussian_noise_augmentation` function adds random noise to the time-series values, simulating natural data variability. The `amplitude_scaling_augmentation` function multiplies the entire sequence by a random factor within a specified range, altering the magnitude of the signal. These augmentations are applied to a subset of the original data for demonstration purposes, creating two different augmented versions.
    """)

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
    else:
        st.warning("Please generate synthetic data on the 'Introduction & Data Generation' page first.")

    st.header("2.3. Visualize Augmented Data")
    st.markdown("""
    Visualizing the augmented data against the original samples is essential to understand the transformations applied. This helps confirm that the augmentation techniques are working as intended and producing realistic variations of the original patterns.
    """)

    if 'X_sample_for_aug' in st.session_state and st.session_state['X_sample_for_aug'].shape[0] > 0:
        sample_idx = st.sidebar.slider("Select Sample for Augmentation Visualization", 0, st.session_state['X_sample_for_aug'].shape[0] - 1, 0, key='aug_sample_idx')

        X_sample_val = st.session_state['X_sample_for_aug']
        X_augmented_noise_val = st.session_state['X_augmented_noise']
        X_augmented_scale_val = st.session_state['X_augmented_scale']

        st.subheader("Original vs. Noise Augmented Sample")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_sample_val[sample_idx, :, 0], mode='lines', name='Original', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_augmented_noise_val[sample_idx, :, 0], mode='lines', name='Noise Augmented', line=dict(color='red', dash='dash')))
        fig1.update_layout(
            title='Original vs. Noise Augmented Sample',
            xaxis_title='Time Step',
            yaxis_title='Value',
            font=dict(size=12),
            title_font_size=16,
            legend_font_size=12,
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Original vs. Scale Augmented Sample")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_sample_val[sample_idx, :, 0], mode='lines', name='Original', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=np.arange(X_sample_val.shape[1]), y=X_augmented_scale_val[sample_idx, :, 0], mode='lines', name='Scale Augmented', line=dict(color='green', dash='dot')))
        fig2.update_layout(
            title='Original vs. Scale Augmented Sample',
            xaxis_title='Time Step',
            yaxis_title='Value',
            font=dict(size=12),
            title_font_size=16,
            legend_font_size=12,
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Relationship: Original vs. Noise Augmented Values")
        df_scatter = pd.DataFrame({
            'Original Value': X_sample_val[:, :, 0].flatten(),
            'Noise Augmented Value': X_augmented_noise_val[:, :, 0].flatten()
        })
        fig3 = px.scatter(df_scatter, x='Original Value', y='Noise Augmented Value', opacity=0.3,
                          title='Relationship: Original vs. Noise Augmented Values',
                          labels={'Original Value': 'Original Value', 'Noise Augmented Value': 'Noise Augmented Value'},
                          color_discrete_sequence=px.colors.qualitative.Plotly
                        )
        fig3.update_layout(
            font=dict(size=12),
            title_font_size=16,
            hovermode="closest"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No augmented data to visualize. Please ensure synthetic data is generated on the 'Introduction & Data Generation' page.")

    st.markdown("""
    Two line plots are generated: one comparing an original sample with its noise-augmented version, and another comparing an original sample with its amplitude-scaled version. These plots clearly show the subtle changes introduced by each augmentation technique. A scatter plot then illustrates the relationship between the original values and the noise-augmented values across all time steps in the sample, demonstrating the spread introduced by the noise. These visualizations confirm the expected impact of data augmentation.
    """)

    st.header("2.4. Model Training (Pre-Augmentation)")
    st.markdown("""
    We will train a simple Long Short-Term Memory (LSTM) model to classify our synthetic time-series data. This model will serve as a baseline to evaluate the impact of data augmentation and the interpretability provided by attention mechanisms. The LSTM architecture is well-suited for sequence prediction tasks due to its ability to capture long-term dependencies.

    The data is first preprocessed using `StandardScaler` and split into training and testing sets. A `tensorflow.keras.Sequential` model with an `LSTM` layer and a `Dense` output layer (with `sigmoid` activation for binary classification) is constructed. The `build_lstm_model` function encapsulates this. The model is then trained using `model.fit()` with `adam` optimizer and `binary_crossentropy` loss. After training, its performance on the test set (accuracy and loss) is evaluated and printed, establishing a baseline for comparison.
    """)

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
            st.warning("Please generate synthetic data on the 'Introduction & Data Generation' page first.")
    
    if 'accuracy_baseline' in st.session_state:
        st.markdown(f"**Current Baseline Model Test Accuracy:** `{st.session_state['accuracy_baseline']:.4f}`")
        st.markdown(f"**Current Baseline Model Test Loss:** `{st.session_state['loss_baseline']:.4f}`")
    else:
        st.info("Baseline model not yet trained. Click 'Train Baseline Model' in the sidebar.")
