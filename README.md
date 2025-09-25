# QuLab: Explainable AI for Sequence Data

## Project Title
**QuLab: Explainable AI for Sequence Data**

## Project Description
QuLab is an interactive Streamlit application designed as a lab project to explore **Explainable AI (XAI)** techniques specifically tailored for **sequence data** (e.g., time-series). Understanding the decision-making process of AI models on sequential information is crucial for building trust, ensuring fairness, and enabling effective decision-making across various domains like finance, healthcare, and IoT.

This application provides hands-on demonstrations of two primary categories of XAI:
1.  **Pre-modeling Techniques**: Applied *before* the model training phase, these methods often involve data manipulation to enhance the model's robustness and indirectly improve interpretability. We focus on **data augmentation** for time-series data, generating diverse training examples from existing ones to make models less sensitive to minor variations.
2.  **In-modeling Techniques**: Integrated *directly into the model's architecture* to provide transparency into its internal workings. We implement and visualize **attention mechanisms**, which allow sequence models to highlight the most relevant time steps in an input sequence when making a prediction, thereby revealing the model's "focus."

Through interactive controls and visualizations, users can generate synthetic time-series data, apply different augmentation strategies, train and compare various LSTM-based models, and visualize attention weights to gain deep insights into how these models interpret sequential patterns. The ultimate goal is to demystify the "black box" of sequence-based AI.

## Features
*   **Interactive Synthetic Time-Series Data Generation**: Generate custom synthetic time-series data with configurable parameters (number of samples, timesteps, frequency, noise levels, trend, and classification threshold).
*   **Comprehensive Data Visualization**:
    *   Plot sample synthetic time-series data to understand patterns and variability.
    *   Visualize the effects of data augmentation (Gaussian noise, amplitude scaling) on individual time series.
    *   Compare the performance (accuracy and loss) of baseline, augmented, and attention-based models.
    *   Visualize attention weights for specific predictions, highlighting influential time steps in the input sequence.
*   **Pre-modeling XAI: Data Augmentation**:
    *   Implement and apply Gaussian noise augmentation.
    *   Implement and apply amplitude scaling augmentation.
    *   Train an LSTM model on an augmented dataset and observe its impact on robustness and performance.
*   **Baseline Model Training**: Train a standard LSTM model on the original synthetic data to establish a performance benchmark.
*   **In-modeling XAI: Attention Mechanisms**:
    *   Integrate a custom Keras AttentionLayer into an LSTM model architecture.
    *   Train an LSTM model enhanced with an attention mechanism.
    *   Extract and visualize attention weights to understand the model's focus during prediction.
*   **Educational Content**: Detailed explanations of sequence-based XAI concepts, learning goals, theoretical background (including Bayes' theorem and attention mechanism principles), and implementation details.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
*   Python 3.8+
*   pip (Python package installer)
*   Git (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    (Replace `your-username/your-repository-name` with the actual path to your repository)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required libraries:**
    It's recommended to create a `requirements.txt` file first.
    
    **`requirements.txt` content:**
    ```
    streamlit>=1.0.0
    numpy>=1.20.0
    pandas>=1.2.0
    plotly>=5.0.0
    scikit-learn>=0.24.0
    tensorflow>=2.5.0
    ```
    
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```

2.  **Navigate the application:**
    The application will open in your web browser. Use the **sidebar** on the left to navigate between the different sections of the lab:
    *   **1. Introduction & Data Generation**: Start here to understand the project, define parameters for synthetic time-series data generation, and visualize initial samples.
    *   **2. Data Augmentation & Baseline Model**: Explore pre-modeling XAI techniques. Apply data augmentation, visualize its effects, and train a baseline LSTM model to establish performance metrics.
    *   **3. Attention Mechanisms & Model Comparison**: Dive into in-modeling XAI. Train a model with data augmentation, compare its performance against the baseline, and then train and visualize an attention-enhanced model.

3.  **Interact with parameters**:
    *   Adjust sliders and input fields in the sidebar to modify data generation, augmentation, and model training parameters.
    *   Click the "Generate Synthetic Data", "Train Baseline Model", "Train Augmented Model", and "Train Attention Model" buttons to execute steps and update visualizations/results.

## Project Structure

```
.
├── app.py
├── application_pages/
│   ├── __init__.py
│   ├── page1.py
│   ├── page2.py
│   └── page3.py
└── requirements.txt
```

*   `app.py`: The main entry point for the Streamlit application. It sets up the page configuration, main title, sidebar navigation, and routes to individual lab pages.
*   `application_pages/`: A directory containing the code for each distinct page of the Streamlit application.
    *   `page1.py`: Handles the introduction, theoretical background, synthetic time-series data generation, and initial data visualization.
    *   `page2.py`: Focuses on pre-modeling XAI (data augmentation), visualizes augmented data, and trains a baseline LSTM model.
    *   `page3.py`: Implements model training with augmented data, compares model performances, introduces in-modeling XAI (attention mechanisms), and visualizes attention weights.
*   `requirements.txt`: Lists all Python dependencies required to run the application.

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application interface.
*   **TensorFlow / Keras**: For building and training deep learning models, including LSTMs and custom attention layers.
*   **NumPy**: For numerical operations, especially with array manipulation for time-series data.
*   **Pandas**: For data manipulation and analysis, particularly for displaying descriptive statistics.
*   **Plotly**: For creating interactive and publication-quality data visualizations.
*   **Scikit-learn**: For data preprocessing (e.g., `StandardScaler`) and dataset splitting (`train_test_split`).

## Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
(Note: You should create a `LICENSE` file in your repository if you choose the MIT license).

## Contact
For any questions or feedback, please reach out:
*   **Your Name / QuantUniversity Team**
*   **Email**: [support@quantuniversity.com](mailto:support@quantuniversity.com)
*   **GitHub**: [https://github.com/your-username](https://github.com/your-username) (Replace with actual link)
*   **Project Link**: [https://github.com/your-username/your-repository-name](https://github.com/your-username/your-repository-name) (Replace with actual link)

