
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will provide an interactive **AI Bias Detection Tool**, designed to help users understand, identify, and mitigate different types of bias in machine learning models. It builds upon a structured Jupyter notebook, converting its sequential analysis into an engaging and dynamic web application.

### Learning Goals

The application aims to enable users to:

*   **Understand AI Bias**: Grasp the fundamental concepts of AI bias, its origins, and its potential societal consequences.
*   **Identify Bias Detection Techniques**: Learn how to employ various metrics to quantify and pinpoint bias in model predictions.
*   **Explore Bias Mitigation Strategies**: Discover and apply techniques to reduce or remove identified biases from models and datasets.
*   **Interpret Key Insights**: Analyze the impact of bias and mitigation strategies through visualizations and comparative metrics.
*   **Understand the key insights** contained in the generated data and analysis.

## 2. User Interface Requirements

### Layout and Navigation Structure

The application will feature a single-page layout, structured logically to follow the AI bias analysis pipeline.

*   **Main Content Area**: This will display the narrative explanations, code snippets, calculated metrics, and visualizations.
*   **Sidebar (Optional for Global Controls)**: While the notebook's interactivity is central, a sidebar could be used for general application settings or data loading options, if an optional external dataset upload feature were to be added (though current scope is synthetic data). For the given scope, all primary controls (`bias_factor`, `reweighting_factor`) will be prominently placed in the main content area, integrated with the interactive analysis section.

### Input Widgets and Controls

The application will include interactive controls to allow users to modify key parameters and observe the real-time impact on bias detection and mitigation.

*   **Bias Factor Slider**: A slider to control the initial strength of bias introduced during synthetic data generation.
    *   **Type**: `st.slider` (FloatSlider)
    *   **Range**: 0.0 to 0.5 (as per notebook's interactive section)
    *   **Step**: 0.05
    *   **Default Value**: 0.3
*   **Reweighting Factor Slider**: A slider to control the strength of the reweighting mitigation technique applied to the dataset.
    *   **Type**: `st.slider` (FloatSlider)
    *   **Range**: 0.0 to 1.0 (as per notebook's interactive section)
    *   **Step**: 0.05
    *   **Default Value**: 0.2
*   **Number of Samples (Fixed/Configurable)**: For the main pipeline demonstration, this can be fixed initially, but for interactive exploration, it might be a text input or slider to control data size, currently it is hardcoded to 2000 in the interactive part.
    *   **Type**: `st.number_input` or `st.slider` for `num_samples` (e.g., 500 to 5000, step 100).

### Visualization Components

The application will present various visualizations to illustrate data characteristics, model performance, and bias metrics.

*   **Data Head Display**: Table showing the first few rows of the generated synthetic data.
    *   **Type**: `st.dataframe`
*   **Summary Statistics Table**: Table displaying descriptive statistics for numerical columns.
    *   **Type**: `st.dataframe`
*   **Bias Metrics Comparison Bar Chart**: A bar chart comparing Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) before and after reweighting.
    *   **Type**: `st.pyplot` (generated using `matplotlib` and `seaborn`)
    *   **Style**: Color-blind-friendly palette, clear titles, labeled axes, and legends.
*   **Feature Importances Heatmap**: A heatmap showing the coefficient values (importances) of features in the Logistic Regression model.
    *   **Type**: `st.pyplot` (generated using `matplotlib` and `seaborn`)
    *   **Style**: Color-blind-friendly palette, clear titles, labeled axes, and legends.

### Interactive Elements and Feedback Mechanisms

*   **Dynamic Updates**: All downstream analysis, metrics, and visualizations will dynamically update in real-time as the `bias_factor` and `reweighting_factor` sliders are adjusted.
*   **Status and Metric Outputs**: Informative textual feedback using `st.write`, `st.info`, `st.success`, `st.error` to communicate:
    *   Current values of bias and reweighting factors.
    *   Status of data generation, preprocessing, and model training.
    *   Calculated model accuracy and AUC-ROC scores (original and reweighted).
    *   Calculated SPD and EOD values (original and reweighted).
    *   Confirmation messages for validation success or failure.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

*   **Sliders**: Both "Bias Factor" and "Reweighting Factor" sliders will include inline help text or tooltips (`help` argument in `st.slider`) to clearly describe their purpose and impact.
    *   **Bias Factor Help Text**: "Controls the initial strength of bias introduced in the synthetic dataset, where higher values mean greater disparity in loan approval probabilities for the privileged group."
    *   **Reweighting Factor Help Text**: "Determines the extent to which samples from the underrepresented group are duplicated, influencing the strength of bias mitigation. A factor of 0 means no reweighting, 1 means full reweighting to balance the groups."
*   **Section Headers**: All main sections (e.g., "Synthetic Data Generation", "Bias Detection") will have clear `st.header` or `st.subheader` titles.
*   **Explanatory Text**: Markdown cells from the Jupyter notebook will be converted to `st.markdown` to provide comprehensive inline explanations for each step, metric, and visualization.
*   **Mathematical Formulas**: All mathematical formulas will be rendered using Streamlit's LaTeX support (`st.latex`) following the specified LaTeX formatting rules.

### Save the States of the Fields Properly

Streamlit's default behavior for widgets automatically handles the state of input fields (sliders, text inputs) across reruns. The `st.slider` widgets used for `bias_factor` and `reweighting_factor` will inherently maintain their values, ensuring that changes are not lost upon interaction or code rerun. This provides a seamless interactive experience where users can continually tweak parameters without losing their previous settings.

## 4. Notebook Content and Code Requirements

This section details how the Jupyter notebook content, including markdown and code, will be integrated into the Streamlit application. All markdown content will be rendered using `st.markdown`, and mathematical expressions using `st.latex`. Code stubs will be displayed and executed as part of the application flow.

---

### Application Title and Overview

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AI Bias Detection Tool: Understanding and Mitigating Unfairness in Machine Learning Models")

st.markdown("""
Artificial Intelligence (AI) models are increasingly integrated into critical decision-making processes across various sectors, from finance and healthcare to recruitment and criminal justice. While AI promises efficiency and objectivity, it can inadvertently perpetuate or even amplify existing societal biases present in the data it's trained on. This can lead to unfair or discriminatory outcomes for certain demographic groups.

This Streamlit application introduces an **AI Bias Detection Tool** designed to help understand, identify, and mitigate different types of bias in machine learning models.
""")

st.header("Learning Goals:")
st.markdown("""
*   **Understand AI Bias**: Grasp the fundamental concepts of AI bias, its origins, and its potential societal consequences.
*   **Identify Bias Detection Techniques**: Learn how to employ various metrics to quantify and pinpoint bias in model predictions.
*   **Explore Bias Mitigation Strategies**: Discover and apply techniques to reduce or remove identified biases from models and datasets.
*   **Interpret Key Insights**: Analyze the impact of bias and mitigation strategies through visualizations and comparative metrics.
""")

st.subheader("Business Value:")
st.markdown("""
This tool provides a practical framework for addressing a critical ethical and business challenge in AI development. By effectively detecting and mitigating bias, organizations can:

*   **Enhance fairness and equity:** Ensure AI systems treat all individuals justly, regardless of sensitive attributes.
*   **Improve model reliability and trustworthiness:** Build AI solutions that are robust and dependable, fostering greater user confidence.
*   **Reduce legal and reputational risks:** Comply with anti-discrimination regulations and avoid public backlash from biased AI.
*   **Optimize business outcomes:** Develop AI that performs well across diverse user groups, leading to broader market acceptance and better results.
""")

st.subheader("What We Will Be Covering / Learning:")
st.markdown("""
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
""")

st.header("References")
st.markdown("""
*   **A Fairer World**: For more in-depth information on AI fairness, explore resources from organizations dedicated to ethical AI.
*   **scikit-learn**: For machine learning algorithms and utilities. (`sklearn`)
*   **pandas**: For data manipulation and analysis. (`pandas`)
*   **numpy**: For numerical operations. (`numpy`)
*   **matplotlib & seaborn**: For data visualization. (`matplotlib.pyplot`, `seaborn`)
*   **Aequitas**: An open-source toolkit for bias and fairness auditing.
*   **Fairlearn**: A Python package for assessing and improving fairness of AI systems.
*   **IBM AI Fairness 360 (AIF360)**: An extensible open-source toolkit that helps detect and mitigate bias in machine learning models.
""")
```

### 3. Synthetic Data Generation

```python
st.header("3. Synthetic Data Generation")

st.markdown("""
To effectively demonstrate AI bias and its mitigation, we need a dataset that is both controllable and interpretable. Real-world datasets can be complex and difficult to attribute bias to specific factors. Therefore, we will generate synthetic data.

**Business Value**: By creating synthetic data, we can explicitly inject bias related to a sensitive attribute (e.g., gender) into a decision-making process (e.g., loan approval). This allows us to clearly observe how a model trained on such data will exhibit biased behavior and provides a controlled environment to test bias detection and mitigation techniques. This step is crucial for research, development, and demonstrating the impact of unfair data practices without relying on sensitive real-world information.

**Technical Implementation**: The `generate_synthetic_data` function will create a DataFrame with a specified number of samples. It includes:

*   **Numeric Features**: `age` (integer, e.g., 18-65) and `income` (float, e.g., normal distribution around a mean).
*   **Categorical Features**: `location` (e.g., 'Urban', 'Suburban', 'Rural') and `gender` (e.g., 'Male', 'Female').
*   **Target Variable**: `loan_approval` (binary: 0 or 1). We introduce bias such that the probability of loan approval is higher for one gender group (e.g., 'Male') by a `bias_factor`. This simulates a scenario where historical data might show preferential treatment.

The formula for loan approval probability ($P(loan_{approval})$) is designed to be influenced by age, income, and critically, gender:

$$ P(loan_{approval} | age, income, gender) = BaseProb + BiasFactor \\times I(gender=Male) + f(age) + g(income) $$

Where:
- $BaseProb$ is a baseline approval probability.
- $I(gender=Male)$ is an indicator function, equal to 1 if the gender is Male, and 0 otherwise. This is where the `bias_factor` explicitly influences the outcome based on gender.
- $f(age)$ and $g(income)$ are functions representing the influence of age and income on loan approval, generally increasing with higher age and income within reasonable ranges.

By fixing a `seed`, we ensure that the generated data is reproducible, allowing for consistent experimentation and demonstration of results.
""")

# Code Stub for generate_synthetic_data (using the non-interactive version first)
st.subheader("`generate_synthetic_data` function definition")
st.code("""
def generate_synthetic_data(num_samples, bias_factor, seed):
    \"\"\"
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
    \"\"\"
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
""", language="python")

# Actual execution and display
num_samples_fixed = 1000 # Using a fixed number for the initial walkthrough
bias_factor_fixed = 0.3
seed_fixed = 42

synthetic_data = generate_synthetic_data(num_samples=num_samples_fixed, bias_factor=bias_factor_fixed, seed=seed_fixed)
st.markdown(f"Generated synthetic data with `num_samples={num_samples_fixed}`, `bias_factor={bias_factor_fixed}`, `seed={seed_fixed}`.")
st.subheader("First 5 rows of Synthetic Data:")
st.dataframe(synthetic_data.head())

st.markdown("""
The code above generates a synthetic dataset. We called the `generate_synthetic_data` function with:

*   `num_samples = 1000`: Creating 1000 rows of data.
*   `bias_factor = 0.3`: Introducing a significant bias where 'Male' individuals have a 0.3 higher baseline probability of loan approval compared to 'Female' individuals.
*   `seed = 42`: Ensuring reproducibility, so running this code multiple times will yield the same dataset.

The printed output shows the first 5 rows of the `synthetic_data` DataFrame. You can observe the generated features (`age`, `income`, `location`, `gender`) and the biased target variable (`loan_approval`). Notice how `loan_approval` is a binary (0 or 1) outcome.
""")
```

### 4. Data Validation

```python
st.header("4. Data Validation")

st.markdown("""
**Business Value**: Data validation is a critical step in any data-driven project. In the context of AI, invalid or inconsistent data can lead to models that perform poorly, make unfair decisions, or even crash. Ensuring data quality from the outset prevents costly errors down the line and builds trust in the AI system's outputs. For an AI Bias Detection Tool, validating that the expected sensitive attributes and outcome variables are correctly structured is paramount to accurately identifying and mitigating bias.

**Technical Implementation**: The `validate_data` function performs several checks:

1.  **Column Name Check**: It verifies that all `expected_columns` are present in the DataFrame. Missing columns could indicate issues in data generation or loading.
2.  **Data Type Check**: It checks if specific columns (e.g., `age`) have the correct data types. Incorrect types can cause errors in subsequent preprocessing or model training steps.
3.  **Missing Values Check**: It identifies if there are any missing values (`NaN`) in critical columns. Missing data often requires imputation or removal, and its presence can skew results or lead to model failures.
4.  **Summary Statistics**: If validation passes, it prints descriptive statistics for numeric columns (`age`, `income`). This provides a quick overview of the data's distribution and helps in initial data understanding.

If any validation check fails, the function prints a descriptive error message and returns `False`; otherwise, it returns `True`.
""")

# Code Stub for validate_data
st.subheader("`validate_data` function definition")
st.code("""
def validate_data(df, expected_columns):
    \"\"\"
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
    \"\"\"

    actual_columns = set(df.columns)
    missing_columns = set(expected_columns) - actual_columns

    if missing_columns:
        st.error(f"Validation Error: Missing expected columns: {', '.join(sorted(list(missing_columns)))}")
        return False

    critical_columns_with_nan = []
    for col in expected_columns:
        if df[col].isnull().any():
            critical_columns_with_nan.append(col)

    if critical_columns_with_nan:
        st.error(f"Validation Error: Missing values found in critical columns: {', '.join(sorted(critical_columns_with_nan))}")
        return False

    if 'age' in df.columns:
        if not pd.api.types.is_integer_dtype(df['age']):
            st.error("Type Error: Incorrect data type for 'age' column. Expected integer type.")
            return False

    st.success("Validation successful.")
    st.subheader("Summary statistics for numeric columns:")
    st.dataframe(df.describe(include=np.number))

    return True
""", language="python")

# Actual execution and display
expected_columns = ['age', 'income', 'location', 'gender', 'loan_approval']
is_valid = validate_data(synthetic_data.copy(), expected_columns)

if is_valid:
    st.success("Data validation successful!")
else:
    st.error("Data validation failed.")

st.markdown("""
The code executes the `validate_data` function on our `synthetic_data` DataFrame. It checks for the presence of the `expected_columns` (`age`, `income`, `location`, `gender`, `loan_approval`), verifies that `age` has an integer data type, and ensures there are no missing values in these critical fields.

Upon successful validation, the output first confirms "Validation successful!" and then prints summary statistics for the numeric columns (`age`, `income`). These statistics include count, mean, standard deviation, min, max, and quartile values, providing a quick overview of the central tendency and spread of our numerical data.

Finally, it prints "Data validation successful!" to confirm the data quality before proceeding to further steps.
""")
```

### 5. Data Preprocessing

```python
st.header("5. Data Preprocessing")

st.markdown("""
**Business Value**: Data preprocessing is a crucial step to prepare raw data for machine learning models. Models often require numerical input and can perform poorly with unscaled features or categorical data in string format. Proper preprocessing ensures that the model can learn effectively, leading to better performance and more reliable bias detection and mitigation.

**Technical Implementation**: This section performs two key preprocessing steps:

1.  **Encoding Categorical Variables**: Machine learning algorithms typically work with numerical input. Our `gender` and `location` columns are categorical strings. We use `LabelEncoder` to convert these into numerical representations. For example, 'Male' might become 0 and 'Female' 1, or 'Urban' 0, 'Suburban' 1, and 'Rural' 2.
    *   `LabelEncoder` assigns a unique integer to each category. This is suitable for ordinal features or when the number of categories is small and no inherent order is implied (though for gender, one-hot encoding might be preferred in some contexts, `LabelEncoder` is simpler for this demonstration).

2.  **Scaling Numerical Features**: Features like `age` and `income` can have vastly different scales. `StandardScaler` transforms these features such that they have a mean of 0 and a standard deviation of 1. This prevents features with larger numerical ranges (like income) from disproportionately influencing the model compared to features with smaller ranges (like age).
    *   The formula for standardization is: $$ z = \frac{x - \mu}{\sigma} $$
    Where:
        *   $z$ is the scaled value.
        *   $x$ is the original feature value.
        *   $\mu$ is the mean of the feature.
        *   $\sigma$ is the standard deviation of the feature.

These steps ensure that our data is in a suitable format for the Logistic Regression model, allowing it to converge more efficiently and perform optimally.
""")

# Code Stub for preprocessing
st.subheader("Preprocessing Code")
st.code("""
# Encode categorical features
label_encoder = LabelEncoder()
synthetic_data['gender'] = label_encoder.fit_transform(synthetic_data['gender'])
synthetic_data['location'] = label_encoder.fit_transform(synthetic_data['location'])

# Scale numerical features
numerical_features = ['age', 'income']
scaler = StandardScaler()
synthetic_data[numerical_features] = scaler.fit_transform(synthetic_data[numerical_features])
""", language="python")

# Actual execution
label_encoder = LabelEncoder()
synthetic_data['gender'] = label_encoder.fit_transform(synthetic_data['gender'])
synthetic_data['location'] = label_encoder.fit_transform(synthetic_data['location'])

numerical_features = ['age', 'income']
scaler = StandardScaler()
synthetic_data[numerical_features] = scaler.fit_transform(synthetic_data[numerical_features])

st.subheader("First 5 rows of Preprocessed Data:")
st.dataframe(synthetic_data.head())

st.markdown("""
The code performs the necessary preprocessing steps on the `synthetic_data` DataFrame.

First, `LabelEncoder` is applied to the `gender` and `location` columns. This converts the categorical string values (e.g., 'Male', 'Female', 'Urban', 'Suburban', 'Rural') into numerical labels. For example, 'Female' and 'Male' might be converted to 0 and 1 respectively, and similarly for 'location'. This is crucial because machine learning models require numerical input.

Second, `StandardScaler` is applied to the `age` and `income` columns. These numerical features are scaled to have a mean of 0 and a standard deviation of 1. This standardization ensures that features with larger numerical ranges do not unduly influence the model's learning process.

The printed output displays the first five rows of the preprocessed `synthetic_data` DataFrame. You can observe that the `gender` and `location` columns now contain integer values, and the `age` and `income` columns contain scaled numerical values, typically centered around zero with small standard deviations. This transformation prepares the data for effective model training.
""")
```

### 6. Data Splitting

```python
st.header("6. Data Splitting")

st.markdown("""
**Business Value**: The primary business value of splitting data is to rigorously evaluate the generalization capability of a machine learning model. A model that performs well only on the data it was trained on (known as overfitting) is not reliable for making predictions on new, unseen data. By training on one portion of the data and testing on another, we can assess how well the model will perform in real-world scenarios, thereby building more robust and trustworthy AI systems.

**Technical Implementation**: We use `sklearn.model_selection.train_test_split` to divide our preprocessed data into two distinct sets:

*   **Training Set (80%)**: This portion of the data (80% in this case) is used to train the machine learning model. The model learns patterns and relationships from these examples.
*   **Testing Set (20%)**: This unseen portion of the data (20%) is reserved exclusively for evaluating the trained model's performance. By testing on data the model has never encountered, we get an unbiased estimate of its predictive power.

The parameters used are:
*   `X`: The features (independent variables) of our dataset, excluding the target variable.
*   `y`: The target variable (dependent variable), which is `loan_approval` in our case.
*   `test_size=0.2`: Specifies that 20% of the data should be allocated to the testing set, and consequently, 80% to the training set.
*   `random_state=42`: This parameter ensures reproducibility. If you run the split multiple times with the same `random_state`, you will always get the same training and testing sets. This is crucial for consistent experimentation.

After splitting, we print the sizes of the training and testing sets to confirm the split was performed correctly.
""")

# Code Stub for data splitting
st.subheader("Data Splitting Code")
st.code("""
X = synthetic_data.drop('loan_approval', axis=1)
y = synthetic_data['loan_approval']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

# Actual execution
X = synthetic_data.drop('loan_approval', axis=1)
y = synthetic_data['loan_approval']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Training set size: {len(X_train)}")
st.write(f"Testing set size: {len(X_test)}")

st.markdown("""
The code first separates the features (`X`) from the target variable (`y`) in our `synthetic_data` DataFrame. The `loan_approval` column is dropped from `X` to become our `y`.

Then, it uses `train_test_split` to divide `X` and `y` into training and testing sets. We allocated 80% of the data for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). The `random_state=42` ensures that this split is consistent every time the code is run.

The printed output shows the number of samples in the training and testing sets, confirming that our data has been successfully partitioned. For instance, with 1000 total samples, you would expect approximately 800 samples in the training set and 200 in the testing set. This prepares our data for the model training and evaluation phases.
""")
```

### 7. Model Training

```python
st.header("7. Model Training")

st.markdown("""
**Business Value**: The core objective of this notebook is to demonstrate bias detection and mitigation in an AI model. Training a model is the foundational step that creates the system we intend to analyze for fairness. The predictive power of this model, even if initially biased, provides the baseline against which we will measure the impact of our bias mitigation strategies. Ultimately, a well-trained model, free from unfair biases, delivers accurate and equitable decisions, which is a key business value.

**Technical Implementation**: We choose **Logistic Regression** as our classification model for this demonstration. Here's why:

*   **Simplicity and Interpretability**: Logistic Regression is a relatively simple linear model, making its behavior and feature importances easier to understand compared to more complex black-box models. This is beneficial for an educational notebook focused on understanding bias.
*   **Binary Classification**: It is inherently well-suited for binary classification problems, such as our `loan_approval` target variable (approved/not approved).
*   **Probability Output**: Logistic Regression outputs probabilities, which can be useful for various downstream analyses, including some bias metrics.

**Model Training Steps**:

1.  **Instantiation**: We create an instance of `LogisticRegression` with `random_state=42` for reproducibility of the model's internal randomness.
2.  **Fitting**: The `model.fit(X_train, y_train)` method trains the model using the training features (`X_train`) and their corresponding true labels (`y_train`). During this process, the model learns the relationship between the input features and the likelihood of loan approval.
3.  **Prediction**: After training, `model.predict(X_test)` generates binary predictions (0 or 1) on the unseen test data (`X_test`). `model.predict_proba(X_test)[:, 1]` gives the probability of the positive class (loan approval).
4.  **Evaluation**: We evaluate the model's performance using two common metrics:
    *   **Accuracy Score**: $$ Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
        Accuracy measures the proportion of correctly classified instances (both approvals and rejections).
    *   **AUC-ROC Score (Area Under the Receiver Operating Characteristic Curve)**: This metric assesses the model's ability to distinguish between the two classes across various classification thresholds. An AUC-ROC of 0.5 indicates no discrimination ability, while 1.0 indicates perfect discrimination.

These metrics provide a quantitative measure of our baseline model's predictive capability before any specific bias mitigation is applied.
""")

# Code Stub for model training
st.subheader("Model Training Code")
st.code("""
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
""", language="python")

# Actual execution
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

st.write(f"Model Accuracy: {accuracy:.4f}")
st.write(f"AUC-ROC Score: {auc_roc:.4f}")

st.markdown("""
The code above trains a Logistic Regression model using our prepared training data and then evaluates its performance on the test data.

1.  **Model Initialization and Training**: A `LogisticRegression` model is initialized with `random_state=42` for reproducibility. The `model.fit(X_train, y_train)` command trains the model on the `X_train` features and `y_train` target labels.
2.  **Prediction**: After training, `model.predict(X_test)` generates binary predictions (0 or 1) for the loan approval outcome on the unseen test set. `model.predict_proba(X_test)[:, 1]` extracts the predicted probabilities of loan approval for each instance in the test set.
3.  **Evaluation Metrics**: The `accuracy_score` calculates the proportion of correctly predicted instances, while the `roc_auc_score` measures the area under the Receiver Operating Characteristic curve, indicating the model's ability to discriminate between positive and negative classes.

The printed output displays the `Model Accuracy` and `AUC-ROC Score`. These values represent the baseline performance of our model *before* any explicit bias detection or mitigation techniques are applied. A higher accuracy and AUC-ROC score generally indicate a better performing model. However, these metrics alone do not tell us about the fairness of the predictions, which we will address in the next sections.
""")
```

### 8. Bias Detection: Statistical Parity Difference

```python
st.header("8. Bias Detection: Statistical Parity Difference")

st.markdown("""
**Business Value**: Statistical Parity Difference (SPD) is a crucial metric for identifying disparate impact, a form of discrimination where a policy or practice results in a disproportionate negative effect on a protected group. In business, understanding SPD helps organizations ensure their AI-driven decisions (e.g., loan approvals, hiring recommendations) do not inadvertently disadvantage certain demographics, thereby maintaining ethical standards, complying with regulations (like non-discrimination laws), and preserving brand reputation.

**Technical Implementation**: Statistical Parity Difference (SPD) measures the difference in the rate of a favorable outcome (e.g., loan approval) between a privileged group and an unprivileged (or unprivileged) group. It is calculated as:

$$ SPD = P(outcome=1 | group=A) - P(outcome=1 | group=B) $$

Where:
*   $P(outcome=1 | group=A)$ is the probability of the favorable outcome (e.g., loan approved) for the **privileged group (A)**.
*   $P(outcome=1 | group=B)$ is the probability of the favorable outcome for the **unprivileged group (B)**.

A SPD value of **0** indicates perfect statistical parity. A positive SPD means the privileged group has a higher rate of favorable outcomes, while a negative SPD means the unprivileged group has a higher rate. In our synthetic data, we have set `gender` as the sensitive attribute, and `gender=1` (Male, after encoding) is considered the **privileged group**, while `gender=0` (Female, after encoding) is the **unprivileged group** for the purpose of demonstrating bias.

Our `statistical_parity_difference` function takes the DataFrame, the group column (e.g., 'gender'), the outcome column (e.g., 'loan_approval'), and the identifier for the privileged group (e.g., 1 for 'Male' after encoding) as input.
""")

# Code Stub for statistical_parity_difference
st.subheader("`statistical_parity_difference` function definition")
st.code("""
def statistical_parity_difference(df, group_col, outcome_col, privileged_group):
    \"\"\"Calculates the Statistical Parity Difference (SPD), a bias metric.
    SPD measures the difference in the proportion of favorable outcomes between a privileged group and an unprivileged group.
    A value of 0 indicates no statistical parity difference.

    Arguments:
        df (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column defining groups.
        outcome_col (str): The name of the column representing the binary outcome (1 for favorable).
        privileged_group (any): The specific value in `group_col` that identifies the privileged group.

    Output:
        float: The calculated Statistical Parity Difference. Returns NaN if a group is absent.
    \"\"\"

    privileged_df = df[df[group_col] == privileged_group]
    unprivileged_df = df[df[group_col] != privileged_group]

    prob_privileged = privileged_df[outcome_col].mean()
    prob_unprivileged = unprivileged_df[outcome_col].mean()

    spd = prob_privileged - prob_unprivileged

    return spd
""", language="python")

# Actual execution
spd = statistical_parity_difference(synthetic_data, 'gender', 'loan_approval', 1) # 1 represents 'Male' (privileged)
st.write(f"Statistical Parity Difference: {spd:.4f}")

st.markdown("""
The code executes the `statistical_parity_difference` function to quantify the bias in our `synthetic_data`.

*   `df=synthetic_data`: The DataFrame containing our preprocessed data.
*   `group_col='gender'`: The sensitive attribute we are examining for bias.
*   `outcome_col='loan_approval'`: The target variable, representing a favorable outcome (loan approval).
*   `privileged_group=1`: After `LabelEncoder`, 'Male' was likely encoded as 1 (and 'Female' as 0), making 'Male' our privileged group in this scenario.

The printed output displays the calculated Statistical Parity Difference (SPD). A positive value indicates that the privileged group (Male) has a higher probability of receiving a loan approval than the unprivileged group (Female). This directly reflects the bias we intentionally introduced during data generation. For example, an SPD of 0.2 means males are 20% more likely to get a loan approved compared to females, demonstrating a clear disparate impact.
""")
```

### 9. Bias Detection: Equal Opportunity Difference

```python
st.header("9. Bias Detection: Equal Opportunity Difference")

st.markdown("""
**Business Value**: Equal Opportunity Difference (EOD) focuses on a specific aspect of fairness: ensuring that individuals who are truly qualified (i.e., should have a positive outcome) have an equal chance of receiving that positive outcome, regardless of their sensitive group membership. In a business context, this translates to ensuring that equally creditworthy individuals, equally qualified job applicants, or equally healthy patients receive the same positive treatment from an AI system, preventing false negatives for specific groups. Addressing EOD is vital for ethical decision-making and preventing a form of unfairness that can lead to significant real-world harm.

**Technical Implementation**: Equal Opportunity Difference (EOD) measures the difference in the true positive rate (or recall) between a privileged group and an unprivileged group. It specifically looks at the rate of favorable outcomes *among those who actually deserve the favorable outcome*. It is calculated as:

$$ EOD = P(outcome=1 | group=A, actual=1) - P(outcome=1 | group=B, actual=1) $$

Where:
*   $P(outcome=1 | group=A, actual=1)$ is the probability of the favorable outcome for the **privileged group (A)**, given that their true (actual) outcome is 1 (favorable).
*   $P(outcome=1 | group=B, actual=1)$ is the probability of the favorable outcome for the **unprivileged group (B)**, given that their true (actual) outcome is 1 (favorable).

An EOD value of **0** indicates equal opportunity. A positive EOD means the privileged group has a higher rate of correct positive predictions among those truly positive, while a negative EOD means the unprivileged group performs better in this regard. Again, in our context, `gender=1` (Male) is the privileged group, and `gender=0` (Female) is the unprivileged group.

Our `equal_opportunity_difference` function takes the DataFrame, the group column, the outcome column, and the identifier for the privileged group. It specifically filters for instances where `outcome_col` is 1 (actual positive outcomes) before calculating the probabilities.
""")

# Code Stub for equal_opportunity_difference
st.subheader("`equal_opportunity_difference` function definition")
st.code("""
def equal_opportunity_difference(df, group_col, outcome_col, privileged_group):
    \"\"\"
    Calculates the Equal Opportunity Difference (EOD).
    Measures the difference in P(outcome=1 | actual=1) between privileged and unprivileged groups.

    Arguments:
    df (pd.DataFrame): DataFrame with group and outcome columns.
    group_col (str): Name of the group column.
    outcome_col (str): Name of the binary outcome column (1 for favorable).
    privileged_group (any): Value identifying the privileged group.

    Returns:
    float: The calculated Equal Opportunity Difference.
    \"\"\"
    favorable_outcome_df = df[df[outcome_col] == 1]

    privileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] == privileged_group]
    rate_privileged = privileged_favorable_df[outcome_col].mean()

    unprivileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] != privileged_group]
    rate_unprivileged = unprivileged_favorable_df[outcome_col].mean()

    eod = rate_privileged - rate_unprivileged

    return eod
""", language="python")

# Actual execution
eod = equal_opportunity_difference(synthetic_data, 'gender', 'loan_approval', 1) # 1 represents 'Male' (privileged)
st.write(f"Equal Opportunity Difference: {eod:.4f}")

st.markdown("""
The code calculates the Equal Opportunity Difference (EOD) using our `equal_opportunity_difference` function. Similar to SPD, it uses:

*   `df=synthetic_data`: The preprocessed DataFrame.
*   `group_col='gender'`: The sensitive group column.
*   `outcome_col='loan_approval'`: The target outcome.
*   `privileged_group=1`: 'Male' as the privileged group.

The printed output displays the calculated EOD. A positive EOD indicates that among those who *actually* deserve a loan (i.e., their `loan_approval` is 1), the privileged group (Males) are being correctly identified at a higher rate than the unprivileged group (Females). This metric is crucial because it highlights if the model is systematically failing to grant favorable outcomes to qualified individuals from the unprivileged group, which is a significant fairness concern.
""")
```

### 10. Bias Mitigation: Reweighting

```python
st.header("10. Bias Mitigation: Reweighting")

st.markdown("""
**Business Value**: Reweighting is a practical and interpretable bias mitigation technique that directly addresses dataset imbalances. In many real-world scenarios, historical data may reflect existing societal biases, leading to underrepresentation or under-selection of certain groups. By reweighting, businesses can proactively adjust their training data to promote fairness, leading to models that make more equitable decisions. This not only enhances ethical compliance but can also improve model performance for historically underserved groups, broadening market reach and improving user satisfaction.

**Technical Implementation**: The Reweighting technique works by assigning different weights to individual data points, typically increasing the influence of underrepresented or disadvantaged groups during model training. The goal is to create a more balanced dataset in terms of group representation and outcome distribution, without physically altering the feature values.

Our `reweight` function implements a specific form of reweighting, by **duplicating rows** of the underrepresented group. Here's the concept:

1.  **Identify Underrepresented Group**: The function first compares the counts of the privileged and non-privileged groups to determine which one is underrepresented based on initial data distribution.
2.  **Calculate Duplication Target**: It then calculates how many rows of the underrepresented group should be duplicated based on a `weight` factor and the total dataset size. The number of rows to add is constrained by `min(reweighted_size, group_B_size)`, implying it aims to increase the presence of the underrepresented group without making it disproportionately dominant over the *original* size of the larger group.
3.  **Duplicate Rows**: If `rows_to_duplicate` is greater than zero, it randomly samples rows from the underrepresented group *with replacement* and concatenates them back to the original DataFrame.

This process effectively increases the presence of the underrepresented group in the training data, allowing the model to learn more from these instances and potentially reduce bias in its predictions. The `weight` parameter controls the extent of this reweighting, with higher values leading to more duplication and a stronger push towards balancing the groups.

Mathematically, this can be seen as altering the effective sample size for different subgroups, thereby influencing the empirical probabilities that the model learns:

$$ P_{reweighted}(Y=y, A=a) = \frac{\sum_{i \\in (Y=y, A=a)} w_i}{\sum_{i} w_i} $$

Where $w_i$ are the weights assigned to each sample. In our duplication method, $w_i$ for duplicated samples is effectively $>1$, while for others it is $1$.
""")

# Code Stub for reweight function
st.subheader("`reweight` function definition")
st.code("""
def reweight(df, group_col, outcome_col, privileged_group, weight):
    \"\"\"Applies a reweighting technique to the input DataFrame to mitigate bias by duplicating samples from the underrepresented group.

    Arguments:
    df (pd.DataFrame): The original DataFrame to be reweighted.
    group_col (str): The name of the column representing the sensitive group.
    outcome_col (str): The name of the column representing the outcome variable.
    privileged_group (any): The value in `group_col` that identifies the privileged group.
    weight (float): A factor used to determine the number of rows to duplicate from the underrepresented group.

    Output:
    pd.DataFrame: A new DataFrame with samples reweighted to mitigate bias, potentially having more rows than the original DataFrame.
    \"\"\"

    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"Column '{outcome_col}' not found in DataFrame.")

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
""", language="python")

# Actual execution
reweighted_data = reweight(synthetic_data, 'gender', 'loan_approval', 1, 0.2) # Weight for 'Female' is 0.2
st.write(f"Original data size: {len(synthetic_data)}")
st.write(f"Reweighted data size: {len(reweighted_data)}")

st.markdown("""
The code executes the `reweight` function to mitigate bias in our `synthetic_data`.

*   `df=synthetic_data`: The original DataFrame.
*   `group_col='gender'`: The sensitive attribute.
*   `outcome_col='loan_approval'`: The outcome variable.
*   `privileged_group=1`: 'Male' as the privileged group. This means the other group (Females, encoded as 0) will be considered for reweighting.
*   `weight=0.2`: A weighting factor. This determines the extent of duplication for the underrepresented group. In this implementation, it means we aim to add up to 20% of the original DataFrame's length in duplicated rows from the underrepresented group.

The printed output shows the size of the `original data` and the `reweighted data`. You will observe that the `reweighted_data` DataFrame has a larger number of rows compared to the original. This increase in size is due to the duplication of samples from the underrepresented group (Females in this case), thereby increasing their representation in the dataset. This modified dataset will then be used to retrain our model, aiming to reduce the observed bias.
""")
```

### 11. Model Training and Evaluation after Reweighting

```python
st.header("11. Model Training and Evaluation after Reweighting")

st.markdown("""
**Business Value**: After applying a bias mitigation technique like reweighting, it's essential to retrain the model on the adjusted data and re-evaluate its performance and fairness. This step directly assesses whether the mitigation strategy was effective in reducing bias without significantly sacrificing predictive accuracy. From a business perspective, this ensures that efforts to improve fairness are validated and that the deployed AI system remains both equitable and performant, maintaining trust and regulatory compliance.

**Technical Implementation**: This section performs the following steps:

1.  **Data Splitting (Reweighted Data)**: The `reweighted_data` is split into new training and testing sets (`X_train_reweighted`, `X_test_reweighted`, `y_train_reweighted`, `y_test_reweighted`). It's crucial to split the *reweighted* data to ensure the model learns from the adjusted distribution.

2.  **Model Retraining**: A new `LogisticRegression` model (`model_reweighted`) is instantiated and trained using the `X_train_reweighted` and `y_train_reweighted`. The model now learns from the data where the underrepresented group has increased presence.

3.  **Model Evaluation**: The retrained model's performance is evaluated on `X_test_reweighted` using:
    *   **Accuracy Score**: `accuracy_score(y_test_reweighted, y_pred_reweighted)`
    *   **AUC-ROC Score**: `roc_auc_score(y_test_reweighted, model_reweighted.predict_proba(X_test_reweighted)[:, 1])`

4.  **Bias Metrics Re-evaluation**: The `statistical_parity_difference` and `equal_opportunity_difference` functions are called again, but this time on the `reweighted_data` (or predictions made on its test split), to see how the bias metrics have changed after mitigation.

By comparing the accuracy, AUC-ROC, SPD, and EOD values before and after reweighting, we can assess the trade-offs. Ideally, bias metrics should move closer to zero (indicating fairness), while accuracy should remain high. A significant drop in accuracy after mitigation might indicate an overcorrection or a need for a different mitigation strategy.
""")

# Code Stub for reweighted model training and evaluation
st.subheader("Reweighted Model Training and Evaluation Code")
st.code("""
# Split the reweighted data
X_reweighted = reweighted_data.drop('loan_approval', axis=1)
y_reweighted = reweighted_data['loan_approval']

X_train_reweighted, X_test_reweighted, y_train_reweighted, y_test_reweighted = train_test_split(X_reweighted, y_reweighted, test_size=0.2, random_state=42)

# Train a new model
model_reweighted = LogisticRegression(random_state=42)
model_reweighted.fit(X_train_reweighted, y_train_reweighted)

# Evaluate the new model
y_pred_reweighted = model_reweighted.predict(X_test_reweighted)
accuracy_reweighted = accuracy_score(y_test_reweighted, y_pred_reweighted)
auc_roc_reweighted = roc_auc_score(y_test_reweighted, model_reweighted.predict_proba(X_test_reweighted)[:, 1])

# Calculate bias metrics on reweighted data
spd_reweighted = statistical_parity_difference(reweighted_data, 'gender', 'loan_approval', 1)
eod_reweighted = equal_opportunity_difference(reweighted_data, 'gender', 'loan_approval', 1)
""", language="python")

# Actual execution
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

st.markdown("""
The code section performs the critical step of retraining our model on the reweighted data and then re-evaluating its performance and fairness metrics.

1.  **Data Preparation**: The `reweighted_data` DataFrame, which now has an increased representation of the previously underrepresented group, is split into new training and testing sets.
2.  **Model Retraining**: A fresh `LogisticRegression` model (`model_reweighted`) is trained on this reweighted training data. This new model learns from the adjusted distribution, aiming to reduce bias.
3.  **Performance Evaluation**: The retrained model's `accuracy` and `AUC-ROC score` are calculated on the reweighted test set. These metrics are then printed, allowing for a direct comparison with the original model's performance.
4.  **Bias Re-evaluation**: Crucially, the `statistical_parity_difference` and `equal_opportunity_difference` functions are called again, this time using the `reweighted_data`. The new SPD and EOD values reflect the impact of the reweighting mitigation strategy.

By comparing these `Reweighted Model Accuracy`, `Reweighted AUC-ROC Score`, `Reweighted Statistical Parity Difference`, and `Reweighted Equal Opportunity Difference` values with their original counterparts, we can analyze the trade-off. Ideally, we would see the bias metrics (SPD and EOD) move closer to zero (indicating reduced bias), while the accuracy and AUC-ROC scores remain comparable or improve. This comparison helps us understand the effectiveness of reweighting in promoting fairness without unduly compromising predictive power.
""")
```

### 12. Visualization: Bias Metrics Comparison

```python
st.header("12. Visualization: Bias Metrics Comparison")

st.markdown("""
**Business Value**: Visualizing bias metrics before and after mitigation is essential for clear communication and impact assessment. Numerical metrics can be abstract, but a compelling bar chart immediately highlights the reduction in unfairness achieved by mitigation strategies. This visual evidence supports ethical decision-making, stakeholder communication, and demonstrates accountability in building fair AI systems. It allows practitioners and non-technical stakeholders alike to quickly grasp the effectiveness of fairness interventions.

**Technical Implementation**: This section generates a bar chart to visually compare the Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) before and after applying the reweighting mitigation technique.

*   **Data Preparation**: We gather the calculated `spd`, `eod` (original values) and `spd_reweighted`, `eod_reweighted` (values after mitigation).
*   **Bar Chart Creation**: `matplotlib.pyplot` is used to create a bar chart. Two sets of bars are plotted for each metric (SPD and EOD):
    *   One set represents the 'Original' bias metric values.
    *   The second set represents the 'Reweighted' bias metric values.
*   **Labels and Title**: The chart is appropriately titled 'Bias Metrics Comparison', with 'Difference' on the y-axis and the specific 'Metrics' on the x-axis.
*   **Legend**: A legend clarifies which bars correspond to 'Original' and 'Reweighted' metrics.

This bar chart provides an intuitive and immediate visual comparison, allowing us to quickly assess the effectiveness of the reweighting strategy in reducing the observed biases. Ideally, the bars for the 'Reweighted' metrics should be closer to zero compared to the 'Original' metrics, indicating a successful reduction in bias.
""")

# Code Stub for plotting bias metrics
st.subheader("Bias Metrics Comparison Plotting Code")
st.code("""
metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
original_values = [spd, eod]
reweighted_values = [spd_reweighted, eod_reweighted]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_values, width, label='Original')
rects2 = ax.bar(x + width/2, reweighted_values, width, label='Reweighted')

ax.set_ylabel('Difference')
ax.set_title('Bias Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Line for ideal fairness (0 difference)

fig.tight_layout()
st.pyplot(fig)
""", language="python")

# Actual execution
metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
original_values = [spd, eod]
reweighted_values = [spd_reweighted, eod_reweighted]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_values, width, label='Original')
rects2 = ax.bar(x + width/2, reweighted_values, width, label='Reweighted')

ax.set_ylabel('Difference')
ax.set_title('Bias Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

fig.tight_layout()
st.pyplot(fig)

st.markdown("""
The code generates a bar chart comparing the two key bias metrics—Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD)—before and after applying the reweighting mitigation technique.

*   The x-axis displays the names of the bias metrics.
*   The y-axis represents the 'Difference' value for each metric.
*   For each metric, two bars are shown: an 'Original' bar (representing the bias before mitigation) and a 'Reweighted' bar (representing the bias after mitigation).

The purpose of this chart is to visually demonstrate the impact of reweighting. By observing the height of the bars, we can easily see if the mitigation strategy has successfully moved the bias metrics closer to zero. A significant reduction in the magnitude of the 'Reweighted' bars compared to the 'Original' bars indicates that the reweighting technique has been effective in reducing the unfairness in the model's predictions with respect to the `gender` attribute.
""")
```

### 13. Visualization: Feature Importances

```python
st.header("13. Visualization: Feature Importances")

st.markdown("""
**Business Value**: Understanding which features most influence a model's decisions is paramount for transparency, interpretability, and debugging. In the context of AI bias, visualizing feature importances can reveal if sensitive attributes (or proxies for them) are disproportionately driving biased outcomes. This insight allows data scientists to identify the root causes of bias, guide feature engineering efforts, and build more ethical and explainable AI systems. For business stakeholders, it provides confidence in knowing *why* a model makes certain predictions.

**Technical Implementation**: This section generates a heatmap to visualize the coefficients of our Logistic Regression model, which serve as indicators of feature importance.

*   **Feature Importances Extraction**: For a linear model like Logistic Regression, the absolute values of the coefficients (`model.coef_[0]`) directly reflect the strength and direction of each feature's influence on the outcome. A larger absolute coefficient implies a greater impact.
*   **DataFrame Creation**: A Pandas DataFrame (`feature_importances_df`) is created to store the feature names and their corresponding importance scores.
*   **Sorting**: The features are sorted by their importance in descending order, making it easy to identify the most influential factors.
*   **Heatmap Visualization**: `seaborn.heatmap` is used to create the visualization:
    *   The heatmap displays the `Importance` values, with `Feature` names as labels.
    *   `annot=True` shows the numerical importance values on the heatmap cells.
    *   `cmap='viridis'` sets the color scheme.
    *   `fmt=".3f"` formats the annotations to three decimal places.

This heatmap helps us understand which features the model relies on most heavily. If a sensitive feature (like 'gender') or a feature highly correlated with it (a 'proxy' feature) shows a high importance, it further supports the finding of bias and points to areas for further investigation or mitigation.
""")

# Code Stub for feature importances plot
st.subheader("Feature Importances Plotting Code")
st.code("""
feature_importances = abs(model.coef_[0])

feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
sns.heatmap(feature_importances_df[['Importance']].set_index(feature_importances_df['Feature']).T, annot=True, cmap='viridis', fmt=".3f", ax=ax_imp)
ax_imp.set_title('Feature Importances')
st.pyplot(fig_imp)
""", language="python")

# Actual execution
feature_importances = abs(model.coef_[0])

feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
sns.heatmap(feature_importances_df[['Importance']].set_index(feature_importances_df['Feature']).T, annot=True, cmap='viridis', fmt=".3f", ax=ax_imp)
ax_imp.set_title('Feature Importances')
st.pyplot(fig_imp)

st.markdown("""
The code generates a heatmap that visually represents the importance of each feature in the Logistic Regression model.

1.  **Extracting Importances**: It retrieves the absolute coefficients of the trained `model`. For linear models like Logistic Regression, the magnitude of these coefficients indicates how much each feature contributes to the prediction. A larger absolute value means a stronger influence.
2.  **Structuring for Visualization**: These importances are then organized into a DataFrame along with their corresponding feature names, and sorted in descending order of importance.
3.  **Heatmap Generation**: `seaborn.heatmap` is used to create the visualization. The heatmap displays the feature names on one axis and their importance values. The color intensity and numerical annotations (`fmt=".3f"`) make it easy to quickly identify which features have the highest impact.

By examining this heatmap, we can gain insights into which features the model primarily relies on to make its `loan_approval` predictions. If the `gender` feature (or any other feature that might serve as a proxy for it) shows a high level of importance, it reinforces our understanding of where the model's bias might be originating. This visualization is crucial for understanding the model's decision-making process and for pinpointing areas that might require further attention in bias mitigation efforts.
""")
```

### 14. Visualization: Interactivity

```python
st.header("14. Visualization: Interactivity")

st.markdown("""
**Business Value**: Interactivity in an AI bias detection tool is invaluable for exploring the complex interplay between initial data biases, mitigation strategies, and resulting model fairness and performance. It transforms a static analysis into a dynamic learning environment, enabling users to gain deeper, intuitive understanding of these trade-offs. This directly supports research, development, and ethical AI deployment by allowing for rapid experimentation and visual assessment of different scenarios.
""")

# Helper Functions for interactive analysis (re-defined or ensured to be accessible for interactive_analysis)
# These will be embedded within the Streamlit app's main script for accessibility.

def _generate_synthetic_data_interactive(num_samples, bias_factor):
    """Generates synthetic data with controlled bias based on 'gender'."""
    np.random.seed(42) # for reproducibility

    data = {}
    data['age'] = np.random.normal(40, 10, num_samples).astype(int)
    data['education'] = np.random.randint(1, 5, num_samples) # 1=High School, 2=Bachelors, 3=Masters, 4=PhD
    data['income'] = np.random.lognormal(mean=11, sigma=0.8, size=num_samples).astype(int)
    data['work_hours'] = np.random.normal(40, 8, num_samples).astype(int)

    # Sensitive feature: gender (0 for male, 1 for female)
    data['gender'] = np.random.randint(0, 2, num_samples)

    df = pd.DataFrame(data)

    # Base probability for loan approval
    base_prob = 0.5
    # Influence of features (example coefficients)
    df['loan_prob'] = (base_prob
                       + 0.005 * (df['age'] - 40)
                       + 0.02 * (df['education'] - 2)
                       + 0.00001 * (df['income'] - df['income'].mean())
                       + 0.01 * (df['work_hours'] - 40)
                      )

    # Introduce bias: decrease loan approval probability for 'gender' == 1 (female)
    df.loc[df['gender'] == 1, 'loan_prob'] -= bias_factor * 0.25

    # Clip probabilities to be within [0, 1]
    df['loan_prob'] = df['loan_prob'].clip(0.05, 0.95)

    df['loan_approved'] = (np.random.rand(num_samples) < df['loan_prob']).astype(int)

    return df.drop(columns=['loan_prob'])


def _preprocess_data_interactive(df):
    """Preprocesses the raw synthetic data by encoding target, scaling numerical features."""

    df_processed = df.copy()

    sensitive_feature_col = 'gender'

    # Encode target variable
    le = LabelEncoder()
    df_processed['loan_approved_encoded'] = le.fit_transform(df_processed['loan_approved'])

    # Scale numerical features
    numerical_cols = ['age', 'income', 'work_hours']
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

    X = df_processed.drop(columns=['loan_approved', 'loan_approved_encoded'])
    y = df_processed['loan_approved_encoded']

    feature_names = X.columns.tolist()

    return X, y, sensitive_feature_col, feature_names


def _apply_reweighting_interactive(X_train, y_train, sensitive_feature_name, reweighting_factor):
    """Calculates sample weights for reweighting mitigation based on the sensitive feature and target."""
    if reweighting_factor == 0.0:
        return np.ones(len(y_train))

    sensitive_group = X_train[sensitive_feature_name]

    n_total = len(y_train)

    p_y0_a0 = ((y_train == 0) & (sensitive_group == 0)).sum() / n_total
    p_y0_a1 = ((y_train == 0) & (sensitive_group == 1)).sum() / n_total
    p_y1_a0 = ((y_train == 1) & (sensitive_group == 0)).sum() / n_total
    p_y1_a1 = ((y_train == 1) & (sensitive_group == 1)).sum() / n_total

    p_y0 = (y_train == 0).sum() / n_total
    p_y1 = (y_train == 1).sum() / n_total

    p_a0 = (sensitive_group == 0).sum() / n_total
    p_a1 = (sensitive_group == 1).sum() / n_total

    weights_map = {}
    if p_y0_a0 > 0: weights_map[(0, 0)] = (p_y0 * p_a0) / p_y0_a0
    if p_y0_a1 > 0: weights_map[(0, 1)] = (p_y0 * p_a1) / p_y0_a1
    if p_y1_a0 > 0: weights_map[(1, 0)] = (p_y1 * p_a0) / p_y1_a0
    if p_y1_a1 > 0: weights_map[(1, 1)] = (p_y1 * p_a1) / p_y1_a1

    sample_weights = np.ones(n_total)
    for i in range(n_total):
        key = (y_train.iloc[i], sensitive_group.iloc[i])
        if key in weights_map:
            sample_weights[i] = 1 + (weights_map[key] - 1) * reweighting_factor

    return sample_weights


def _train_model_interactive(X_train, y_train, sample_weights=None):
    """Trains a Logistic Regression model with optional sample weights."""
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def _calculate_bias_metrics_interactive(model, X_data, y_true, sensitive_feature_name):
    """Calculates Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD).
    Assumes 'gender' == 1 is the unprivileged group and 'gender' == 0 is the privileged group
    (as per the bias introduced in data generation).
    Metrics = Unprivileged_group_metric - Privileged_group_metric.
    """

    y_pred = model.predict(X_data)

    group_unprivileged_idx = (X_data[sensitive_feature_name] == 1)
    group_privileged_idx = (X_data[sensitive_feature_name] == 0)

    prob_positive_unprivileged = 0
    if group_unprivileged_idx.sum() > 0:
        prob_positive_unprivileged = y_pred[group_unprivileged_idx].sum() / group_unprivileged_idx.sum()

    prob_positive_privileged = 0
    if group_privileged_idx.sum() > 0:
        prob_positive_privileged = y_pred[group_privileged_idx].sum() / group_privileged_idx.sum()

    spd = prob_positive_unprivileged - prob_positive_privileged

    true_positive_unprivileged_idx = group_unprivileged_idx & (y_true == 1)
    true_positive_privileged_idx = group_privileged_idx & (y_true == 1)

    prob_eod_unprivileged = 0
    if true_positive_unprivileged_idx.sum() > 0:
        pred_positive_given_true_positive_unprivileged = y_pred[true_positive_unprivileged_idx].sum()
        prob_eod_unprivileged = pred_positive_given_true_positive_unprivileged / true_positive_unprivileged_idx.sum()

    prob_eod_privileged = 0
    if true_positive_privileged_idx.sum() > 0:
        pred_positive_given_true_positive_privileged = y_pred[true_positive_privileged_idx].sum()
        prob_eod_privileged = pred_positive_given_true_positive_privileged / true_positive_privileged_idx.sum()

    eod = prob_eod_unprivileged - prob_eod_privileged

    return {'SPD': spd, 'EOD': eod}


def _plot_bias_metrics_interactive(original_metrics, reweighted_metrics):
    """Visualizes the comparison of bias metrics using a bar chart."""
    metrics_data = {
        'Metric': ['SPD', 'EOD', 'SPD', 'EOD'],
        'Value': [original_metrics['SPD'], original_metrics['EOD'],
                  reweighted_metrics['SPD'], reweighted_metrics['EOD']],
        'Type': ['Original', 'Original', 'Reweighted', 'Reweighted']
    }
    metrics_df = pd.DataFrame(metrics_data)

    fig_interactive_bias, ax_interactive_bias = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Metric', y='Value', hue='Type', data=metrics_df, palette='viridis', ax=ax_interactive_bias)
    ax_interactive_bias.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax_interactive_bias.set_title('Bias Metrics Comparison: Original vs. Reweighted')
    ax_interactive_bias.set_ylabel('Metric Value')
    ax_interactive_bias.set_xlabel('Bias Metric')
    ax_interactive_bias.legend(title='Mitigation Type')
    fig_interactive_bias.tight_layout()
    st.pyplot(fig_interactive_bias)


def _plot_feature_importances_interactive(model, feature_names):
    """Visualizes feature importances (Logistic Regression coefficients)."""
    if hasattr(model, 'coef_') and model.coef_.shape[0] > 0:
        importances = pd.Series(model.coef_[0], index=feature_names)
        importances = importances.sort_values(ascending=False)

        fig_interactive_imp, ax_interactive_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette='coolwarm', ax=ax_interactive_imp)
        ax_interactive_imp.set_title('Feature Importances (Logistic Regression Coefficients)')
        ax_interactive_imp.set_xlabel('Coefficient Value')
        ax_interactive_imp.set_ylabel('Feature')
        fig_interactive_imp.tight_layout()
        st.pyplot(fig_interactive_imp)


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
    reweighted_model = _train_model_interactive(X_train, y_train, sample_weights=sample_weights_train)
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

# Streamlit UI for interactive analysis
st.subheader("Interactive Controls")

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

st.markdown("""
The interactive section, driven by the sliders, allows you to dynamically experiment with different scenarios of bias and mitigation.

By adjusting the **Bias Factor** slider, you can increase or decrease the initial bias injected into the synthetic data generation. This helps you understand how varying levels of inherent data bias impact the model's fairness.

Similarly, by moving the **Reweight Factor** slider, you can control the strength of the reweighting mitigation applied to the dataset. You can observe in real-time how increasing the reweighting factor attempts to reduce the Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD).

After each adjustment of the sliders, the entire analysis pipeline (data generation, preprocessing, model training, bias detection, reweighting, and re-evaluation) is re-run, and the following outputs are updated:

*   **Original Model Accuracy and Bias Metrics**: These show the baseline performance and fairness given the current `bias_factor`.
*   **Reweighted Model Accuracy and Bias Metrics**: These show the impact of the `reweighting_factor` on both performance and fairness.
*   **Bias Metrics Comparison Plot**: This bar chart visually updates to show the changes in SPD and EOD before and after mitigation, helping you see the effectiveness of the reweighting.
*   **Feature Importances Plot**: This heatmap shows the updated feature importances for the reweighted model.

This interactive experience provides immediate feedback on the trade-offs between model accuracy and fairness. You can observe how a higher `bias_factor` might lead to larger original bias metrics, and how increasing the `reweighting_factor` generally pushes these metrics closer to zero, potentially with some impact on accuracy. This hands-on exploration deepens the understanding of bias dynamics and mitigation strategies in AI.
""")
```

### 15. Conclusion

```python
st.header("15. Conclusion")

st.markdown("""
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
""")
```
