id: 68d54eae89fb95725ebc3d08_documentation
summary: Explainable AI Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab - Explainable AI: AI Bias Detection Tool

## 1. Introduction to AI Bias and QuLab Overview
Duration: 0:10:00

Artificial Intelligence (AI) models are increasingly integrated into critical decision-making processes across various sectors, from finance and healthcare to recruitment and criminal justice. While AI promises efficiency and objectivity, it can inadvertently perpetuate or even amplify existing societal biases present in the data it's trained on. This can lead to unfair or discriminatory outcomes for certain demographic groups.

This Streamlit application, **QuLab - Explainable AI: AI Bias Detection Tool**, is designed to help developers understand, identify, and mitigate different types of bias in machine learning models.

<aside class="positive">
<b>Why is this important?</b> Building fair and ethical AI systems is not just an ethical imperative but also a business necessity. Biased AI can lead to reputational damage, regulatory penalties, and suboptimal business outcomes by unfairly impacting certain user groups.
</aside>

### Learning Goals:

*   **Understand AI Bias**: Grasp the fundamental concepts of AI bias, its origins, and its potential societal consequences.
*   **Identify Bias Detection Techniques**: Learn how to employ various metrics to quantify and pinpoint bias in model predictions.
*   **Explore Bias Mitigation Strategies**: Discover and apply techniques to reduce or remove identified biases from models and datasets.
*   **Interpret Key Insights**: Analyze the impact of bias and mitigation strategies through visualizations and comparative metrics.

### Business Value:

This tool provides a practical framework for addressing a critical ethical and business challenge in AI development. By effectively detecting and mitigating bias, organizations can:

*   **Enhance fairness and equity:** Ensure AI systems treat all individuals justly, regardless of sensitive attributes.
*   **Improve model reliability and trustworthiness:** Build AI solutions that are robust and dependable, fostering greater user confidence.
*   **Reduce legal and reputational risks:** Comply with anti-discrimination regulations and avoid public backlash from biased AI.
*   **Optimize business outcomes:** Develop AI that performs well across diverse user groups, leading to broader market acceptance and better results.

### What We Will Be Covering / Learning:

In this application, we will walk through the entire pipeline of building and evaluating an AI model with a focus on bias. We will:

1.  **Generate Synthetic Data:** Create a dataset with inherent bias to simulate real-world scenarios.
2.  **Validate and Preprocess Data:** Ensure data quality and prepare it for model training.
3.  **Train a Baseline Model:** Develop a logistic regression model and evaluate its initial performance and bias.
4.  **Detect Bias:** Utilize key metrics like Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) to quantify bias.
5.  **Mitigate Bias:** Implement the Reweighting technique to adjust the dataset and reduce observed bias.
6.  **Evaluate Mitigated Model:** Assess the performance and bias of the model after applying mitigation.
7.  **Visualize Results:** Create compelling visualizations to understand bias metrics and feature importances.
8.  **Enable Interactivity:** Allow users to explore the impact of bias and mitigation factors dynamically.

We will explain the underlying mathematical formulas and their business relevance throughout this process. By the end, you will have a solid understanding of how to proactively address bias in your AI applications.

### Application Architecture and Flow

The QuLab application is built using Streamlit and organized into three main pages, accessible via the sidebar navigation. This modular design allows for a clear, step-by-step progression through the AI bias detection and mitigation pipeline.

```mermaid
graph TD
    A[app.py - Main Application] --> B{Sidebar Navigation};
    B -- "Data Generation & Baseline Model" --> C[Page 1: application_pages/page1.py];
    B -- "Bias Detection & Mitigation" --> D[Page 2: application_pages/page2.py];
    B -- "Visualizations & Interactivity" --> E[Page 3: application_pages/page3.py];

    C -- "Generates Synthetic Data" --> F[st.session_state (synthetic_data)];
    C -- "Preprocesses Data" --> G[st.session_state (X, y, X_train, y_train, X_test, y_test, etc.)];
    C -- "Trains Baseline Model" --> H[st.session_state (model, accuracy, auc_roc)];

    D -- "Loads Data/Model from Session" --> F;
    D -- "Calculates Bias Metrics (SPD, EOD)" --> I[st.session_state (spd, eod)];
    D -- "Applies Reweighting Mitigation" --> J[st.session_state (reweighted_data)];
    D -- "Retrains & Re-evaluates Model" --> K[st.session_state (model_reweighted, spd_reweighted, eod_reweighted, etc.)];

    E -- "Loads All Metrics/Models from Session" --> I;
    E -- "Loads All Metrics/Models from Session" --> K;
    E -- "Visualizes Bias & Feature Importances" --> L[Plotly Charts];
    E -- "Enables Interactive Analysis" --> C;
    E -- "Enables Interactive Analysis" --> D;
```

**Workflow:**

1.  **Page 1 (`page1.py`)**: You begin by generating a synthetic dataset where bias is intentionally introduced. This page also handles data preprocessing (encoding categorical features, scaling numerical features) and trains an initial "baseline" Logistic Regression model. The results and processed data are stored in Streamlit's `session_state` for subsequent steps.
2.  **Page 2 (`page2.py`)**: This page focuses on quantifying the bias in the baseline model using metrics like Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD). It then applies a bias mitigation technique called "Reweighting" to the dataset and retrains the model on this adjusted data, recalculating the bias metrics to assess the impact of mitigation. All new data and models are also stored in `session_state`.
3.  **Page 3 (`page3.py`)**: The final page provides visual comparisons of the bias metrics before and after mitigation, displays feature importances, and crucially, offers an interactive component. This interactive section allows you to dynamically adjust the initial bias and the strength of the reweighting technique, rerunning the entire pipeline on the fly to observe the immediate effects on fairness and performance.

This design ensures that each step builds upon the previous one, providing a cohesive and comprehensive learning experience for understanding and addressing AI bias.

## 2. Setup and Running the Application
Duration: 0:05:00

To get started with the QuLab application, you'll need to set up your Python environment and install the necessary dependencies.

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Directory Structure

The application consists of three main Python files:

```
.
├── app.py
└── application_pages/
    ├── __init__.py
    ├── page1.py
    ├── page2.py
    └── page3.py
```

*   `app.py`: The main entry point for the Streamlit application. It handles the overall structure, introduction, and navigation.
*   `application_pages/page1.py`: Contains the logic for synthetic data generation, validation, preprocessing, and baseline model training.
*   `application_pages/page2.py`: Contains the functions for bias detection (SPD, EOD) and bias mitigation (Reweighting), along with retraining the model.
*   `application_pages/page3.py`: Contains the logic for visualizations, interactive analysis, and the conclusion.

### Installation

1.  **Clone the repository (if applicable) or create the files locally.**
2.  **Install dependencies:** Navigate to the root directory of your project (where `app.py` is located) in your terminal or command prompt and run:

    ```bash
    pip install streamlit pandas numpy scikit-learn plotly
    ```

### Running the Application

Once the dependencies are installed, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your default web browser. You can then navigate through the different pages using the sidebar.

<aside class="positive">
<b>Tip:</b> If the application doesn't open automatically, Streamlit will provide a URL in your terminal (usually `http://localhost:8501`) that you can copy and paste into your browser.
</aside>

## 3. Data Generation: Simulating Bias
Duration: 0:10:00

Our journey begins by creating a synthetic dataset designed to simulate real-world scenarios where inherent biases exist. This controlled environment allows us to clearly observe the impact of bias and test mitigation strategies.

### Function: `generate_synthetic_data`

The `generate_synthetic_data` function is responsible for creating a Pandas DataFrame with features like `age`, `income`, `location`, `gender`, and a binary target `loan_approval`. A key aspect of this function is the intentional introduction of bias related to the `gender` feature.

```python
def generate_synthetic_data(num_samples, bias_factor, seed):
    """
    Generates synthetic data with a specified bias, including numeric (age, income) and categorical (location, gender) features,
    and a biased binary target variable (loan approval). The bias is introduced such that one gender group has a higher
    baseline probability of loan approval.

    Arguments:
    num_samples (int): The number of samples (rows) to generate in the dataset.
    bias_factor (float): A value controlling the strength of the bias related to the 'gender' feature,
                         specifically increasing the loan approval probability for the privileged group.
    seed (int): A random seed for NumPy to ensure reproducibility of the generated data.

    Output:
    pd.DataFrame: A Pandas DataFrame containing the synthetic data with features 'age', 'income', 'location', 'gender',
                  and the target 'loan_approval'.
    """
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if not isinstance(bias_factor, float):
        raise TypeError("bias_factor must be a float.")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if num_samples == 0:
        return pd.DataFrame(columns=['age', 'income', 'location', 'gender', 'loan_approval'])

    np.random.seed(seed)

    age = np.random.randint(low=20, high=66, size=num_samples)
    income = np.random.uniform(low=30000, high=150000, size=num_samples)

    locations = ['Urban', 'Suburban', 'Rural']
    location = np.random.choice(locations, size=num_samples)

    genders = ['Male', 'Female']
    gender = np.random.choice(genders, size=num_samples)

    base_approval_prob = 0.55

    loan_approval_probs = np.full(num_samples, base_approval_prob)

    male_indices = (gender == 'Male')
    loan_approval_probs[male_indices] = loan_approval_probs[male_indices] + bias_factor

    loan_approval_probs = np.clip(loan_approval_probs, 0.0, 1.0)

    loan_approval = (np.random.rand(num_samples) < loan_approval_probs).astype(float)

    data = {
        'age': age,
        'income': income,
        'location': location,
        'gender': gender,
        'loan_approval': loan_approval
    }
    df = pd.DataFrame(data)

    return df
```

**Key aspects of bias introduction:**

*   A `base_approval_prob` is set for everyone.
*   For `male_indices` (gender == 'Male'), the `loan_approval_probs` are increased by the `bias_factor`. This means males will have a higher probability of `loan_approval` by default, simulating a historical or systemic bias in the data.
*   The `loan_approval` target variable is then generated based on these biased probabilities.

### Streamlit UI in `page1.py`

On the "Data Generation & Baseline Model" page, you'll find sliders to control the `Number of Samples` and the `Bias Factor`.

*   **Number of Samples**: Adjusts the total rows in the generated dataset.
*   **Bias Factor**: Directly influences the magnitude of bias towards males in loan approvals. A higher factor means a stronger bias.

After adjusting these, click "Generate Data". The application will display the first few rows of the generated data, allowing you to observe the raw, biased dataset.

<aside class="negative">
Remember, the `bias_factor` in this synthetic data generation is intentionally used to simulate unfairness for demonstration purposes. In real-world applications, bias can be subtle and unintended.
</aside>

## 4. Data Validation and Preprocessing
Duration: 0:08:00

Before training any machine learning model, it's crucial to validate and preprocess the data. This ensures data quality, consistency, and suitability for the chosen algorithms.

### Function: `validate_data`

The `validate_data` function performs essential checks on the generated DataFrame:

```python
def validate_data(df, expected_columns):
    """
    Validates the input DataFrame by checking for the presence of expected column names,
    verifying data types for specific columns (e.g., 'age'), and ensuring there are
    no missing values in critical fields. It also prints summary statistics for numeric
    columns if validation passes.

    Arguments:
        df (pd.DataFrame): The DataFrame to be validated.
        expected_columns (list): A list of strings representing the column names
                                 that are expected to be present in the DataFrame.
    Output:
        bool: True if all validation checks pass, False otherwise. Error messages
              are printed to the console if validation fails.
    """

    actual_columns = set(df.columns)
    missing_columns = set(expected_columns) - actual_columns

    if missing_columns:
        st.error(f"Validation Error: Missing expected columns: {missing_columns}")
        return False

    # Example: Check data types for critical columns
    if not pd.api.types.is_numeric_dtype(df['age']):
        st.error("Validation Error: 'age' column is not numeric.")
        return False

    if not pd.api.types.is_numeric_dtype(df['income']):
        st.error("Validation Error: 'income' column is not numeric.")
        return False
    
    # Check for missing values in critical columns
    if df[['age', 'income', 'location', 'gender', 'loan_approval']].isnull().any().any():
        st.error("Validation Error: Missing values found in critical columns.")
        return False

    st.success("Data validation passed successfully!")
    st.subheader("Summary Statistics for Numeric Features:")
    st.write(df[['age', 'income']].describe())
    st.subheader("Value Counts for Categorical Features:")
    st.write(df['location'].value_counts())
    st.write(df['gender'].value_counts())
    st.write(df['loan_approval'].value_counts())

    return True
```

This function verifies column presence, data types, and checks for missing values, providing early feedback on data integrity. It also outputs summary statistics, which are useful for understanding the distribution of your data.

### Function: `preprocess_data`

The `preprocess_data` function prepares the data for machine learning by:

1.  **Label Encoding:** Converts categorical features (`location`, `gender`) into numerical representations. This is necessary for algorithms like Logistic Regression that expect numerical inputs.
2.  **Standard Scaling:** Normalizes numerical features (`age`, `income`) to have a mean of 0 and a standard deviation of 1. This prevents features with larger scales from dominating the learning process.

```python
def preprocess_data(df):
    """
    Preprocesses the raw synthetic data by encoding categorical features (location, gender)
    and scaling numerical features (age, income) using StandardScaler. The target variable
    'loan_approval' is separated and kept as is.

    Arguments:
    df (pd.DataFrame): The raw DataFrame to be preprocessed.

    Output:
    tuple: A tuple containing:
           - pd.DataFrame: The preprocessed features (X).
           - pd.Series: The target variable (y).
           - sklearn.preprocessing.LabelEncoder: The LabelEncoder fitted on 'gender'.
    """

    df_processed = df.copy()

    # Encode categorical features
    le_location = LabelEncoder()
    df_processed['location'] = le_location.fit_transform(df_processed['location'])
    
    le_gender = LabelEncoder()
    df_processed['gender'] = le_gender.fit_transform(df_processed['gender'])
    
    st.subheader("Label Encoding Map for 'Gender':")
    st.write(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
    st.subheader("Label Encoding Map for 'Location':")
    st.write(dict(zip(le_location.classes_, le_location.transform(le_location.classes_))))

    # Scale numerical features
    numerical_cols = ['age', 'income']
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

    # Separate features (X) and target (y)
    X = df_processed.drop('loan_approval', axis=1)
    y = df_processed['loan_approval']

    st.success("Data preprocessed successfully (Categorical features encoded, Numerical features scaled).")
    st.subheader("First 5 rows of preprocessed features (X):")
    st.write(X.head())
    st.subheader("First 5 rows of target (y):")
    st.write(y.head())

    return X, y, le_gender
```

Notice the output of the `LabelEncoder` for 'Gender'. If 'Male' maps to 1 and 'Female' to 0, it means that for our bias detection later, 'Male' (1) will be the privileged group and 'Female' (0) the unprivileged group.

### Streamlit UI in `page1.py`

After generating the data, clicking "Validate & Preprocess Data" triggers these functions. You'll see:

*   Validation messages (success or error).
*   Summary statistics for numerical features.
*   Value counts for categorical features.
*   The mapping for label encoding, particularly for `gender`.
*   The first few rows of the preprocessed feature DataFrame (`X`) and target Series (`y`).

<aside class="positive">
<b>Best Practice:</b> Data validation and preprocessing are critical steps. They ensure your model receives clean, properly formatted data, which is foundational for both performance and reliable bias detection.
</aside>

## 5. Baseline Model Training and Evaluation
Duration: 0:07:00

With the data preprocessed, we can now train our initial machine learning model, which will serve as a baseline for understanding performance and identifying existing biases.

### Function: `train_baseline_model`

The `train_baseline_model` function performs the following:

1.  **Data Splitting:** Divides the preprocessed data into training and testing sets using `train_test_split`. This ensures the model is evaluated on unseen data.
2.  **Model Instantiation:** Creates an instance of `LogisticRegression`, a common and interpretable classification algorithm.
3.  **Model Training:** Fits the Logistic Regression model on the training data (`X_train`, `y_train`).
4.  **Prediction:** Generates predictions on the test set (`X_test`).
5.  **Evaluation:** Calculates two key performance metrics:
    *   **Accuracy Score**: The proportion of correctly classified instances.
    *   **AUC-ROC Score**: Measures the model's ability to distinguish between classes, particularly useful for imbalanced datasets.

```python
def train_baseline_model(X, y):
    """
    Trains a Logistic Regression model on the provided data, splits it into
    training and testing sets, and evaluates the model's performance using
    accuracy and AUC-ROC score.

    Arguments:
    X (pd.DataFrame): The preprocessed feature DataFrame.
    y (pd.Series): The target variable Series.

    Output:
    tuple: A tuple containing:
           - sklearn.linear_model.LogisticRegression: The trained model.
           - float: The accuracy score on the test set.
           - float: The AUC-ROC score on the test set.
           - pd.DataFrame: X_train
           - pd.DataFrame: X_test
           - pd.Series: y_train
           - pd.Series: y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"Training data size: {len(X_train)} samples")
    st.write(f"Testing data size: {len(X_test)} samples")

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    st.success("Baseline Logistic Regression model trained and evaluated.")
    st.write(f"Baseline Model Accuracy: {accuracy:.4f}")
    st.write(f"Baseline AUC-ROC Score: {auc_roc:.4f}")

    return model, accuracy, auc_roc, X_train, X_test, y_train, y_test
```

### Streamlit UI in `page1.py`

Clicking "Train & Evaluate Baseline Model" on the "Data Generation & Baseline Model" page executes this function. You will see:

*   The sizes of the training and testing datasets.
*   The calculated `Baseline Model Accuracy`.
*   The calculated `Baseline AUC-ROC Score`.

These metrics provide an initial understanding of how well our model performs *before* any explicit bias detection or mitigation efforts. This step sets the stage for comparing the impact of our fairness interventions.

## 6. Bias Detection: Statistical Parity Difference (SPD)
Duration: 0:08:00

Once a baseline model is established, the next crucial step is to quantify any existing biases. Statistical Parity Difference (SPD) is a fundamental metric for detecting a specific type of bias known as **disparate impact**.

<aside class="positive">
<b>Business Value:</b> SPD is a crucial metric for identifying disparate impact, a form of discrimination where a policy or practice results in a disproportionate negative effect on a protected group. In business, understanding SPD helps organizations ensure their AI-driven decisions (e.g., loan approvals, hiring recommendations) do not inadvertently disadvantage certain demographics, thereby maintaining ethical standards, complying with regulations (like non-discrimination laws), and preserving brand reputation.
</aside>

### Technical Implementation

Statistical Parity Difference (SPD) measures the difference in the rate of a favorable outcome (e.g., loan approval) between a privileged group and an unprivileged group. It is calculated as:

$$ SPD = P(\text{outcome}=1 | \text{group}=A) - P(\text{outcome}=1 | \text{group}=B) $$

Where:
*   $P(\text{outcome}=1 | \text{group}=A)$ is the probability of the favorable outcome (e.g., loan approved) for the **privileged group (A)**.
*   $P(\text{outcome}=1 | \text{group}=B)$ is the probability of the favorable outcome for the **unprivileged group (B)**.

A SPD value of **0** indicates perfect statistical parity. A positive SPD means the privileged group has a higher rate of favorable outcomes, while a negative SPD means the unprivileged group has a higher rate. In our synthetic data, we have set `gender` as the sensitive attribute, and `gender=1` (Male, after encoding) is considered the **privileged group**, while `gender=0` (Female, after encoding) is the **unprivileged group** for the purpose of demonstrating bias.

### Function: `statistical_parity_difference`

```python
def statistical_parity_difference(df, group_col, outcome_col, privileged_group):
    """Calculates the Statistical Parity Difference (SPD), a bias metric.
    SPD measures the difference in the proportion of favorable outcomes between a privileged group and an unprivileged group.
    A value of 0 indicates no statistical parity difference.

    Arguments:
        df (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column defining groups.
        outcome_col (str): The name of the column representing the binary outcome (1 for favorable).
        privileged_group (any): The specific value in `group_col` that identifies the privileged group.

    Output:
        float: The calculated Statistical Parity Difference. Returns NaN if a group is absent.
    """

    privileged_df = df[df[group_col] == privileged_group]
    unprivileged_df = df[df[group_col] != privileged_group]

    prob_privileged = privileged_df[outcome_col].mean()
    prob_unprivileged = unprivileged_df[outcome_col].mean()

    spd = prob_privileged - prob_unprivileged

    return spd
```

### Streamlit UI in `page2.py`

On the "Bias Detection & Mitigation" page, the application automatically calculates and displays the SPD using the `synthetic_data` (before any reweighting).

```python
spd = statistical_parity_difference(synthetic_data, 'gender', 'loan_approval', 1)
st.write(f"Statistical Parity Difference: {spd:.4f}")
```

The printed output displays the calculated Statistical Parity Difference (SPD). A positive value indicates that the privileged group (Male) has a higher probability of receiving a loan approval than the unprivileged group (Female). This directly reflects the bias we intentionally introduced during data generation. For example, an SPD of 0.2 means males are 20% more likely to get a loan approved compared to females, demonstrating a clear disparate impact.

## 7. Bias Detection: Equal Opportunity Difference (EOD)
Duration: 0:08:00

Beyond just looking at overall outcome rates (SPD), it's important to assess if the model is fair for *qualified* individuals across different groups. This is where Equal Opportunity Difference (EOD) comes in.

<aside class="positive">
<b>Business Value:</b> Equal Opportunity Difference (EOD) focuses on a specific aspect of fairness: ensuring that individuals who are truly qualified (i.e., should have a positive outcome) have an equal chance of receiving that positive outcome, regardless of their sensitive group membership. In a business context, this translates to ensuring that equally creditworthy individuals, equally qualified job applicants, or equally healthy patients receive the same positive treatment from an AI system, preventing false negatives for specific groups. Addressing EOD is vital for ethical decision-making and preventing a form of unfairness that can lead to significant real-world harm.
</aside>

### Technical Implementation

Equal Opportunity Difference (EOD) measures the difference in the true positive rate (or recall) between a privileged group and an unprivileged group. It specifically looks at the rate of favorable outcomes *among those who actually deserve the favorable outcome*. It is calculated as:

$$ EOD = P(\text{outcome}=1 | \text{group}=A, \text{actual}=1) - P(\text{outcome}=1 | \text{group}=B, \text{actual}=1) $$

Where:
*   $P(\text{outcome}=1 | \text{group}=A, \text{actual}=1)$ is the probability of the favorable outcome for the **privileged group (A)**, given that their true (actual) outcome is 1 (favorable).
*   $P(\text{outcome}=1 | \text{group}=B, \text{actual}=1)$ is the probability of the favorable outcome for the **unprivileged group (B)**, given that their true (actual) outcome is 1 (favorable).

An EOD value of **0** indicates equal opportunity. A positive EOD means the privileged group has a higher rate of correct positive predictions among those truly positive, while a negative EOD means the unprivileged group performs better in this regard. Again, in our context, `gender=1` (Male) is the privileged group, and `gender=0` (Female) is the unprivileged group.

### Function: `equal_opportunity_difference`

```python
def equal_opportunity_difference(df, group_col, outcome_col, privileged_group):
    """
    Calculates the Equal Opportunity Difference (EOD).
    Measures the difference in P(outcome=1 | actual=1) between privileged and unprivileged groups.

    Arguments:
    df (pd.DataFrame): DataFrame with group and outcome columns.
    group_col (str): Name of the group column.
    outcome_col (str): Name of the binary outcome column (1 for favorable).
    privileged_group (any): Value identifying the privileged group.

    Returns:
    float: The calculated Equal Opportunity Difference.
    """
    favorable_outcome_df = df[df[outcome_col] == 1]

    privileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] == privileged_group]
    rate_privileged = privileged_favorable_df[outcome_col].mean()

    unprivileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] != privileged_group]
    rate_unprivileged = unprivileged_favorable_df[outcome_col].mean()

    eod = rate_privileged - rate_unprivileged

    return eod
```

### Streamlit UI in `page2.py`

On the "Bias Detection & Mitigation" page, the EOD is also automatically calculated and displayed:

```python
eod = equal_opportunity_difference(synthetic_data, 'gender', 'loan_approval', 1)
st.write(f"Equal Opportunity Difference: {eod:.4f}")
```

The printed output displays the calculated EOD. A positive EOD indicates that among those who *actually* deserve a loan (i.e., their `loan_approval` is 1), the privileged group (Males) are being correctly identified at a higher rate than the unprivileged group (Females). This metric is crucial because it highlights if the model is systematically failing to grant favorable outcomes to qualified individuals from the unprivileged group, which is a significant fairness concern.

## 8. Bias Mitigation: Reweighting
Duration: 0:10:00

After detecting biases, the next step is to mitigate them. Reweighting is an in-processing bias mitigation technique that adjusts the weights of individual training samples to reduce bias in the learned model.

<aside class="positive">
<b>Business Value:</b> Reweighting is a practical and interpretable bias mitigation technique that directly addresses dataset imbalances. In many real-world scenarios, historical data may reflect existing societal biases, leading to underrepresentation or under-selection of certain groups. By reweighting, businesses can proactively adjust their training data to promote fairness, leading to models that make more equitable decisions. This not only enhances ethical compliance but can also improve model performance for historically underserved groups, broadening market reach and improving user satisfaction.
</aside>

### Technical Implementation

The Reweighting technique works by assigning different weights to individual data points, typically increasing the influence of underrepresented or disadvantaged groups during model training. The goal is to create a more balanced dataset in terms of group representation and outcome distribution, without physically altering the feature values.

Our `reweight` function implements a specific form of reweighting, by **duplicating rows** of the underrepresented group. Here's the concept:

1.  **Identify Underrepresented Group**: The function first compares the counts of the privileged and non-privileged groups to determine which one is underrepresented based on initial data distribution.
2.  **Calculate Duplication Target**: It then calculates how many rows of the underrepresented group should be duplicated based on a `weight` factor and the total dataset size. The number of rows to add is constrained by `min(reweighted_size, group_B_size)`, implying it aims to increase the presence of the underrepresented group without making it disproportionately dominant over the *original* size of the larger group.
3.  **Duplicate Rows**: If `rows_to_duplicate` is greater than zero, it randomly samples rows from the underrepresented group *with replacement* and concatenates them back to the original DataFrame.

This process effectively increases the presence of the underrepresented group in the training data, allowing the model to learn more from these instances and potentially reduce bias in its predictions. The `weight` parameter controls the extent of this reweighting, with higher values leading to more duplication and a stronger push towards balancing the groups.

Mathematically, this can be seen as altering the effective sample size for different subgroups, thereby influencing the empirical probabilities that the model learns:

$$ P_{reweighted}(Y=y, A=a) = \frac{\sum_{i \in (Y=y, A=a)} w_i}{\sum_{i} w_i} $$

Where $w_i$ are the weights assigned to each sample. In our duplication method, $w_i$ for duplicated samples is effectively $>1$, while for others it is $1$.

### Function: `reweight`

```python
def reweight(df, group_col, outcome_col, privileged_group, weight):
    """Applies a reweighting technique to the input DataFrame to mitigate bias by duplicating samples from the underrepresented group.

    Arguments:
    df (pd.DataFrame): The original DataFrame to be reweighted.
    group_col (str): The name of the column representing the sensitive group.
    outcome_col (str): The name of the column representing the outcome variable.
    privileged_group (any): The value in `group_col` that identifies the privileged group.
    weight (float): A factor used to determine the number of rows to duplicate from the underrepresented group.

    Output:
    pd.DataFrame: A new DataFrame with samples reweighted to mitigate bias, potentially having more rows than the original DataFrame.
    """

    if group_col not in df.columns:
        raise KeyError(f"Column \'{group_col}\' not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"Column \'{outcome_col}\' not found in DataFrame.")

    if df.empty:
        return df.copy()

    if weight == 0.0:
        return df.copy()

    unique_groups = df[group_col].unique()

    if len(unique_groups) < 2:
        return df.copy()

    non_privileged_group_vals = [val for val in unique_groups if val != privileged_group]

    if not non_privileged_group_vals or len(non_privileged_group_vals) > 1:
        return df.copy()

    group_A_val = privileged_group
    group_B_val = non_privileged_group_vals[0]

    group_A_initial_count = df[df[group_col] == group_A_val].shape[0]
    group_B_initial_count = df[df[group_col] == group_B_val].shape[0]

    underrepresented_group_val = None
    underrepresented_group_initial_count = 0

    if group_A_initial_count < group_B_initial_count:
        underrepresented_group_val = group_A_val
        underrepresented_group_initial_count = group_A_initial_count
    else:
        underrepresented_group_val = group_B_val
        underrepresented_group_initial_count = group_B_initial_count

    if underrepresented_group_initial_count == 0:
        return df.copy()

    df_len = len(df)

    reweighted_target_add_size = int(weight * df_len)

    rows_to_duplicate = min(reweighted_target_add_size, underrepresented_group_initial_count)

    if rows_to_duplicate > 0:
        underrepresented_group_df = df[df[group_col] == underrepresented_group_val]
        duplicated_rows = underrepresented_group_df.sample(n=rows_to_duplicate, replace=True, random_state=42)
        reweighted_df = pd.concat([df, duplicated_rows], ignore_index=True)
        return reweighted_df
    else:
        return df.copy()
```

### Streamlit UI in `page2.py`

On the "Bias Detection & Mitigation" page, the `reweight` function is applied:

```python
reweighted_data = reweight(synthetic_data, 'gender', 'loan_approval', 1, 0.2)
st.write(f"Original data size: {len(synthetic_data)}")
st.write(f"Reweighted data size: {len(reweighted_data)}")
```

The printed output shows the size of the `original data` and the `reweighted data`. You will observe that the `reweighted_data` DataFrame has a larger number of rows compared to the original. This increase in size is due to the duplication of samples from the underrepresented group (Females in this case), thereby increasing their representation in the dataset. This modified dataset will then be used to retrain our model, aiming to reduce the observed bias.

## 9. Model Training and Evaluation after Reweighting
Duration: 0:08:00

After applying a bias mitigation technique like reweighting, it's essential to retrain the model on the adjusted data and re-evaluate its performance and fairness. This step directly assesses whether the mitigation strategy was effective in reducing bias without significantly sacrificing predictive accuracy.

<aside class="positive">
<b>Business Value:</b> From a business perspective, this ensures that efforts to improve fairness are validated and that the deployed AI system remains both equitable and performant, maintaining trust and regulatory compliance.
</aside>

### Technical Implementation

This section performs the following steps:

1.  **Data Splitting (Reweighted Data)**: The `reweighted_data` is split into new training and testing sets (`X_train_reweighted`, `X_test_reweighted`, `y_train_reweighted`, `y_test_reweighted`). It's crucial to split the *reweighted* data to ensure the model learns from the adjusted distribution.
2.  **Model Retraining**: A new `LogisticRegression` model (`model_reweighted`) is instantiated and trained using the `X_train_reweighted` and `y_train_reweighted`. The model now learns from the data where the underrepresented group has increased presence.
3.  **Model Evaluation**: The retrained model's performance is evaluated on `X_test_reweighted` using `Accuracy Score` and `AUC-ROC Score`.
4.  **Bias Metrics Re-evaluation**: The `statistical_parity_difference` and `equal_opportunity_difference` functions are called again on the `reweighted_data` (or predictions made on its test split), to see how the bias metrics have changed after mitigation.

By comparing the accuracy, AUC-ROC, SPD, and EOD values before and after reweighting, we can assess the trade-offs. Ideally, bias metrics should move closer to zero (indicating fairness), while accuracy should remain high. A significant drop in accuracy after mitigation might indicate an overcorrection or a need for a different mitigation strategy.

### Streamlit UI in `page2.py`

The following code is executed on the "Bias Detection & Mitigation" page:

```python
X_reweighted = reweighted_data.drop('loan_approval', axis=1)
y_reweighted = reweighted_data['loan_approval']
X_train_reweighted, X_test_reweighted, y_train_reweighted, y_test_reweighted = train_test_split(X_reweighted, y_reweighted, test_size=0.2, random_state=42)
model_reweighted = LogisticRegression(random_state=42)
model_reweighted.fit(X_train_reweighted, y_train_reweighted)
y_pred_reweighted = model_reweighted.predict(X_test_reweighted)
accuracy_reweighted = accuracy_score(y_test_reweighted, y_pred_reweighted)
auc_roc_reweighted = roc_auc_score(y_test_reweighted, model_reweighted.predict_proba(X_test_reweighted)[:, 1])
spd_reweighted = statistical_parity_difference(reweighted_data, 'gender', 'loan_approval', 1)
eod_reweighted = equal_opportunity_difference(reweighted_data, 'gender', 'loan_approval', 1)

st.write(f"Reweighted Model Accuracy: {accuracy_reweighted:.4f}")
st.write(f"Reweighted AUC-ROC Score: {auc_roc_reweighted:.4f}")
st.write(f"Reweighted Statistical Parity Difference: {spd_reweighted:.4f}")
st.write(f"Reweighted Equal Opportunity Difference: {eod_reweighted:.4f}")
```

The code section performs the critical step of retraining our model on the reweighted data and then re-evaluating its performance and fairness metrics.

By comparing these `Reweighted Model Accuracy`, `Reweighted AUC-ROC Score`, `Reweighted Statistical Parity Difference`, and `Reweighted Equal Opportunity Difference` values with their original counterparts from the baseline model, we can analyze the trade-off. Ideally, we would see the bias metrics (SPD and EOD) move closer to zero (indicating reduced bias), while the accuracy and AUC-ROC scores remain comparable or improve. This comparison helps us understand the effectiveness of reweighting in promoting fairness without unduly compromising predictive power.

## 10. Visualization: Bias Metrics Comparison
Duration: 0:07:00

Numerical metrics can be abstract. Visualizing bias metrics before and after mitigation is essential for clear communication and impact assessment. A compelling chart immediately highlights the reduction in unfairness achieved by mitigation strategies.

<aside class="positive">
<b>Business Value:</b> This visual evidence supports ethical decision-making, stakeholder communication, and demonstrates accountability in building fair AI systems. It allows practitioners and non-technical stakeholders alike to quickly grasp the effectiveness of fairness interventions.
</aside>

### Technical Implementation

This section generates a bar chart using `plotly.graph_objects` to visually compare the Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) before and after applying the reweighting mitigation technique.

*   **Data Preparation**: We gather the calculated `spd`, `eod` (original values) and `spd_reweighted`, `eod_reweighted` (values after mitigation).
*   **Bar Chart Creation**: Two sets of bars are plotted for each metric (SPD and EOD): one for 'Original' and one for 'Reweighted'.
*   **Labels and Title**: The chart is appropriately titled 'Bias Metrics Comparison', with 'Difference' on the y-axis and the specific 'Metrics' on the x-axis. A horizontal line at $y=0$ is added to represent the ideal fairness state.

### Streamlit UI in `page3.py`

The following code generates the plot on the "Visualizations & Interactivity" page:

```python
metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
original_values = [spd, eod]
reweighted_values = [spd_reweighted, eod_reweighted]

fig_bias = go.Figure()
fig_bias.add_trace(go.Bar(name='Original', x=metrics_names, y=original_values))
fig_bias.add_trace(go.Bar(name='Reweighted', x=metrics_names, y=reweighted_values))
fig_bias.update_layout(barmode='group', title_text='Bias Metrics Comparison', yaxis_title='Difference')
fig_bias.add_hline(y=0, line_dash="dash", line_color="grey") # Line for ideal fairness (0 difference)
st.plotly_chart(fig_bias)
```

The chart visually demonstrates the impact of reweighting. By observing the height of the bars, we can easily see if the mitigation strategy has successfully moved the bias metrics closer to zero. A significant reduction in the magnitude of the 'Reweighted' bars compared to the 'Original' bars indicates that the reweighting technique has been effective in reducing the unfairness in the model's predictions with respect to the `gender` attribute.

## 11. Visualization: Feature Importances
Duration: 0:07:00

Understanding which features most influence a model's decisions is paramount for transparency, interpretability, and debugging. In the context of AI bias, visualizing feature importances can reveal if sensitive attributes (or proxies for them) are disproportionately driving biased outcomes.

<aside class="positive">
<b>Business Value:</b> This insight allows data scientists to identify the root causes of bias, guide feature engineering efforts, and build more ethical and explainable AI systems. For business stakeholders, it provides confidence in knowing *why* a model makes certain predictions.
</aside>

### Technical Implementation

This section generates a heatmap to visualize the coefficients of our Logistic Regression model, which serve as indicators of feature importance.

*   **Feature Importances Extraction**: For a linear model like Logistic Regression, the absolute values of the coefficients (`model.coef_[0]`) directly reflect the strength and direction of each feature's influence on the outcome. A larger absolute coefficient implies a greater impact.
*   **DataFrame Creation**: A Pandas DataFrame (`feature_importances_df`) is created to store the feature names and their corresponding importance scores.
*   **Sorting**: The features are sorted by their importance in descending order, making it easy to identify the most influential factors.
*   **Heatmap Visualization**: `plotly.graph_objects` is used to create the visualization, displaying the `Importance` values with `Feature` names as labels, enhanced with numerical annotations and a colorscale.

### Streamlit UI in `page3.py`

The following code generates the heatmap on the "Visualizations & Interactivity" page:

```python
feature_importances = abs(model.coef_[0])

feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

fig_imp = go.Figure(data=go.Heatmap(
    z=feature_importances_df['Importance'].values.reshape(1, -1),
    x=feature_importances_df['Feature'],
    y=['Importance'],
    colorscale='viridis',
    colorbar_title='Importance'
))

fig_imp.update_layout(
    title_text='Feature Importances',
    xaxis_title='Feature',
    yaxis_title='',
    yaxis_automargin=True
)

annotations = []
for i, feature in enumerate(feature_importances_df['Feature']):
    annotations.append({
        "x": feature,
        "y": 'Importance',
        "xref": "x",
        "yref": "y",
        "text": f"{feature_importances_df.iloc[i]['Importance']:.3f}",
        "showarrow": False,
        "font": {"color": "black" if feature_importances_df.iloc[i]['Importance'] < (feature_importances_df['Importance'].max() / 2) else "white"}
    })
fig_imp.update_layout(annotations=annotations)

st.plotly_chart(fig_imp)
```

By examining this heatmap, we can gain insights into which features the model primarily relies on to make its `loan_approval` predictions. If the `gender` feature (or any other feature that might serve as a proxy for it) shows a high level of importance, it reinforces our understanding of where the model's bias might be originating. This visualization is crucial for understanding the model's decision-making process and for pinpointing areas that might require further attention in bias mitigation efforts.

## 12. Interactive Bias Analysis
Duration: 0:12:00

Interactivity is an invaluable feature for an AI bias detection tool, enabling deeper, intuitive understanding of the complex interplay between initial data biases, mitigation strategies, and resulting model fairness and performance.

<aside class="positive">
<b>Business Value:</b> This transforms a static analysis into a dynamic learning environment, enabling users to gain deeper, intuitive understanding of these trade-offs. This directly supports research, development, and ethical AI deployment by allowing for rapid experimentation and visual assessment of different scenarios.
</aside>

### Technical Implementation

The interactive analysis is driven by helper functions (e.g., `_generate_synthetic_data_interactive`, `_preprocess_data_interactive`, `_apply_reweighting_interactive`, `_train_model_interactive`, `_calculate_bias_metrics_interactive`, `_plot_bias_metrics_interactive`, `_plot_feature_importances_interactive`) which mirror the core logic of the previous steps but are designed to be re-run dynamically based on user input.

The `interactive_analysis` function orchestrates this end-to-end pipeline:

```python
def interactive_analysis(bias_factor_val, reweighting_factor_val):
    """Performs an end-to-end AI bias analysis pipeline interactively."""

    st.info(f"Starting Interactive AI Bias Analysis with:")
    st.info(f"  Bias Factor: {bias_factor_val}")
    st.info(f"  Reweighting Factor: {reweighting_factor_val}")

    num_samples = 2000
    df = _generate_synthetic_data_interactive(num_samples, bias_factor_val)
    st.write(f"\nGenerated {len(df)} synthetic data samples.")

    X, y, sensitive_feature_col, feature_names = _preprocess_data_interactive(df)
    st.write("Data preprocessed (target encoded, numerical features scaled).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    st.write(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")

    st.write("\nTraining original Logistic Regression model...")
    original_model = _train_model_interactive(X_train, y_train)
    y_pred_original = original_model.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    st.write(f"Original Model Accuracy: {accuracy_original:.4f}")

    original_metrics = _calculate_bias_metrics_interactive(original_model, X_test, y_test, sensitive_feature_col)
    st.write(f"Original Bias Metrics:")
    st.write(f"  Statistical Parity Difference (SPD): {original_metrics['SPD']:.4f}")
    st.write(f"  Equal Opportunity Difference (EOD): {original_metrics['EOD']:.4f}")

    st.write(f"\nApplying reweighting with factor {reweighting_factor_val} and training mitigated model...")
    sample_weights_train = _apply_reweighting_interactive(X_train, y_train, sensitive_feature_col, reweighting_factor_val)
    reweighted_model = _train_model_interactive(X_train, y_train, sample_weight=sample_weights_train)
    y_pred_reweighted = reweighted_model.predict(X_test)
    accuracy_reweighted = accuracy_score(y_test, y_pred_reweighted)
    st.write(f"Reweighted Model Accuracy: {accuracy_reweighted:.4f}")

    reweighted_metrics = _calculate_bias_metrics_interactive(reweighted_model, X_test, y_test, sensitive_feature_col)
    st.write(f"Reweighted Bias Metrics:")
    st.write(f"  Statistical Parity Difference (SPD): {reweighted_metrics['SPD']:.4f}")
    st.write(f"  Equal Opportunity Difference (EOD): {reweighted_metrics['EOD']:.4f}")

    _plot_bias_metrics_interactive(original_metrics, reweighted_metrics)
    st.write("\nBias metrics comparison plot displayed.")

    _plot_feature_importances_interactive(reweighted_model, feature_names)
    st.write("Feature importances plot displayed.")

    st.success("\nInteractive AI Bias Analysis pipeline completed.")
```

### Streamlit UI in `page3.py`

On the "Visualizations & Interactivity" page, you will find two sliders:

*   **Bias Factor (Controls initial data bias)**: Adjusts the initial strength of bias introduced in the synthetic dataset.
*   **Reweighting Factor (Adjusts mitigation strength)**: Determines the extent to which samples from the underrepresented group are duplicated, influencing the strength of bias mitigation.

```python
bias_factor_slider = st.slider(
    "Bias Factor (Controls initial data bias):",
    min_value=0.0,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="Controls the initial strength of bias introduced in the synthetic dataset, where higher values mean greater disparity in loan approval probabilities for the privileged group."
)

reweighting_factor_slider = st.slider(
    "Reweighting Factor (Adjusts mitigation strength):",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Determines the extent to which samples from the underrepresented group are duplicated, influencing the strength of bias mitigation. A factor of 0 means no reweighting, 1 means full reweighting to balance the groups."
)

interactive_analysis(bias_factor_slider, reweighting_factor_slider)
```

By adjusting these sliders, the entire analysis pipeline (data generation, preprocessing, model training, bias detection, reweighting, and re-evaluation) is re-run, and the following outputs are updated in real-time:

*   **Original Model Accuracy and Bias Metrics**: These show the baseline performance and fairness given the current `bias_factor`.
*   **Reweighted Model Accuracy and Bias Metrics**: These show the impact of the `reweighting_factor` on both performance and fairness.
*   **Bias Metrics Comparison Plot**: This bar chart visually updates to show the changes in SPD and EOD before and after mitigation, helping you see the effectiveness of the reweighting.
*   **Feature Importances Plot**: This heatmap shows the updated feature importances for the reweighted model.

This hands-on exploration deepens the understanding of bias dynamics and mitigation strategies in AI, providing immediate feedback on the trade-offs between model accuracy and fairness.

## 13. Conclusion
Duration: 0:05:00

Throughout this Streamlit application, we have embarked on a comprehensive journey to understand, detect, and mitigate AI bias. We began by acknowledging the critical business and ethical implications of bias in AI systems, particularly in sensitive decision-making contexts like loan approvals.

### Key Learning Outcomes:

*   **Synthetic Data Generation**: We learned how to create a controlled synthetic dataset, allowing us to inject and observe specific biases related to sensitive attributes like `gender`.
*   **Data Validation and Preprocessing**: We reinforced the importance of robust data quality checks and preparation (encoding, scaling) for reliable model training.
*   **Model Training Baseline**: We established a baseline Logistic Regression model and evaluated its initial performance, setting the stage for bias analysis.
*   **Bias Detection Metrics**: We applied and interpreted two fundamental fairness metrics:
    *   **Statistical Parity Difference (SPD)**: Measuring disparate impact (difference in favorable outcomes across groups).
    *   **Equal Opportunity Difference (EOD)**: Assessing equal chances for truly qualified individuals across groups.
*   **Bias Mitigation with Reweighting**: We implemented and evaluated a proactive bias mitigation technique, reweighting, which adjusted the influence of underrepresented groups in the training data.
*   **Visual Analysis**: We utilized visualizations (bar charts for bias comparison, heatmaps for feature importances) to clearly communicate the presence of bias and the impact of mitigation strategies.
*   **Interactive Exploration**: The interactive elements allowed for dynamic experimentation, demonstrating how varying bias levels and mitigation efforts affect both model performance and fairness in real-time.

### Importance of AI Bias Detection and Mitigation:

This exercise highlights that building fair AI is not just an ethical imperative but also a business necessity. Unfair AI can lead to:

*   **Reputational Damage**: Loss of public trust and negative brand perception.
*   **Regulatory Penalties**: Non-compliance with anti-discrimination laws and increasing AI regulations.
*   **Suboptimal Business Outcomes**: Missing out on qualified candidates or customers due to biased decisions.

By systematically detecting and addressing bias, we can develop AI systems that are more equitable, robust, and trustworthy, ultimately delivering greater value to society and organizations alike.

### Further Exploration:

This application provides a foundational understanding. For deeper insights, consider exploring:

*   **Other Bias Mitigation Techniques**: Investigate in-processing (e.g., adversarial debiasing) and post-processing (e.g., equalized odds post-processing) mitigation strategies.
*   **Additional Fairness Metrics**: Explore metrics like Disparate Impact, Predictive Parity, or Treatment Equality.
*   **Different Model Architectures**: Apply bias detection and mitigation to more complex models (e.g., Neural Networks).
*   **Real-world Datasets**: Apply these techniques to real-world datasets, being mindful of data privacy and ethical considerations.
