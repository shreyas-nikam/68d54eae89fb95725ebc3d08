# QuLab - Explainable AI: AI Bias Detection Tool

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab - Explainable AI: AI Bias Detection Tool** is a Streamlit-powered web application designed as a practical lab project to explore, understand, and mitigate bias in machine learning models. In an era where AI profoundly influences critical decisions, ensuring fairness and equity is paramount. This tool provides a hands-on experience to identify and address inherent biases that can arise from training data, demonstrating their impact and showcasing strategies for mitigation.

The application walks users through a complete pipeline, from generating synthetically biased data to training a baseline model, detecting various types of bias using established metrics, applying a mitigation technique (Reweighting), and finally, evaluating the impact of these interventions through interactive visualizations.

## Features

This application provides a comprehensive learning and analysis platform with the following key features:

*   **Synthetic Data Generation**:
    *   Generate a customizable synthetic dataset with numerical (age, income) and categorical (location, gender) features.
    *   Introduce controlled bias related to a sensitive attribute (e.g., gender) to simulate real-world fairness challenges.
*   **Data Validation and Preprocessing**:
    *   Perform basic data validation checks to ensure data quality.
    *   Apply essential preprocessing steps like Label Encoding for categorical features and StandardScaler for numerical features.
*   **Baseline Model Training**:
    *   Train a Logistic Regression model on the initial, potentially biased dataset.
    *   Evaluate its performance using standard metrics like Accuracy and AUC-ROC Score.
*   **Bias Detection Metrics**:
    *   Calculate and interpret fundamental fairness metrics:
        *   **Statistical Parity Difference (SPD)**: Measures the difference in favorable outcome rates between privileged and unprivileged groups.
        *   **Equal Opportunity Difference (EOD)**: Assesses fairness among truly qualified individuals across groups.
*   **Bias Mitigation Strategy**:
    *   Implement and apply the **Reweighting** technique to the training data to balance the representation of underrepresented groups.
*   **Mitigated Model Evaluation**:
    *   Retrain the Logistic Regression model on the reweighted data.
    *   Re-evaluate model performance and, critically, recalculate bias metrics to assess the effectiveness of mitigation.
*   **Interactive Visualizations**:
    *   **Bias Metrics Comparison**: Bar charts to visually compare SPD and EOD before and after bias mitigation.
    *   **Feature Importances**: Heatmap visualization of Logistic Regression coefficients to understand feature influence on predictions.
*   **Dynamic Interactivity**:
    *   Explore the impact of varying levels of initial data bias and reweighting strength using interactive sliders.
    *   Observe real-time updates to model performance, bias metrics, and visualizations as parameters are adjusted.
*   **Educational Content**: Rich explanations of concepts, mathematical formulas, business value, and technical implementation details are integrated throughout the application.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8+ installed. The application relies on several Python libraries which can be installed via `pip`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quslab-ai-bias-detection.git
    cd quslab-ai-bias-detection
    ```
    *(Replace `your-username/quslab-ai-bias-detection` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit>=1.0
    pandas>=1.0
    numpy>=1.20
    scikit-learn>=0.24
    plotly>=5.0
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if you created one):
    ```bash
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Navigate to the project's root directory** (where `app.py` is located) in your terminal.

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

### Navigating the Application

The application is structured into three main pages, accessible via the sidebar on the left:

1.  **Data Generation & Baseline Model**:
    *   Start here to generate the synthetic data and train your initial Logistic Regression model.
    *   The generated data and trained model are stored in the session for subsequent pages.
2.  **Bias Detection & Mitigation**:
    *   Proceed to this page to calculate initial bias metrics (SPD, EOD).
    *   Apply the reweighting mitigation technique and retrain the model on the adjusted data.
    *   Observe the new model's performance and bias metrics.
3.  **Visualizations & Interactivity**:
    *   Visualize the comparison of bias metrics before and after mitigation using bar charts.
    *   View feature importances as a heatmap.
    *   Engage with interactive sliders to dynamically adjust initial data bias and reweighting strength, observing real-time changes in all metrics and visualizations.

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
└── README.md
```

*   `app.py`: The main entry point of the Streamlit application. It sets up the page configuration, displays the project overview, and handles navigation between different sections using `st.sidebar.selectbox`.
*   `application_pages/`: A directory containing the logic for each distinct page of the application.
    *   `page1.py`: Handles synthetic data generation, validation, preprocessing, and training of the initial baseline model.
    *   `page2.py`: Implements bias detection metrics (SPD, EOD) and the reweighting bias mitigation technique, followed by retraining and re-evaluation of the model.
    *   `page3.py`: Contains functions for interactive analysis, including dynamic data generation, model training, bias calculation, and comprehensive visualizations (bias metrics comparison, feature importances).
*   `requirements.txt`: Lists all Python dependencies required to run the application.
*   `README.md`: This file, providing an overview and instructions for the project.

## Technology Stack

The application leverages the following key libraries and frameworks:

*   **Streamlit**: For rapidly building and deploying interactive web applications with Python.
*   **pandas**: For efficient data manipulation and analysis.
*   **NumPy**: For numerical operations, especially in data generation and array manipulation.
*   **scikit-learn**: A robust machine learning library used for:
    *   `LogisticRegression`: The primary machine learning model.
    *   `train_test_split`: For splitting data into training and testing sets.
    *   `accuracy_score`, `roc_auc_score`: For model performance evaluation.
    *   `LabelEncoder`, `StandardScaler`: For data preprocessing.
*   **Plotly (plotly.express, plotly.graph_objects)**: For creating interactive and publication-quality data visualizations.

**References (for further exploration of AI Fairness):**
*   **Aequitas**: An open-source toolkit for bias and fairness auditing.
*   **Fairlearn**: A Python package for assessing and improving fairness of AI systems.
*   **IBM AI Fairness 360 (AIF360)**: An extensible open-source toolkit that helps detect and mitigate bias in machine learning models.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Clone** your forked repository: `git clone https://github.com/your-username/quslab-ai-bias-detection.git`
3.  Create a new **branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`
4.  Make your changes and **commit** them: `git commit -m "feat: Add new feature X"`
5.  **Push** your changes to your fork: `git push origin feature/your-feature-name`
6.  Open a **Pull Request** to the `main` branch of the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(You'll need to create a `LICENSE` file in your repository with the MIT license text.)*

## Contact

For any questions, suggestions, or feedback, please reach out to:

*   **Project Maintainer**: [Your Name/QuantUniversity Team]
*   **Email**: [your.email@example.com / info@quantuniversity.com]
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **GitHub**: [https://github.com/your-username](https://github.com/your-username)
