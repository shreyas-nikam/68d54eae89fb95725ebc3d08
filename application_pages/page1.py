
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

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
        st.error(f"Validation Error: Missing expected columns: {', '.join(sorted(list(missing_columns)))}\")
        return False

    critical_columns_with_nan = []
    for col in expected_columns:
        if df[col].isnull().any():
            critical_columns_with_nan.append(col)

    if critical_columns_with_nan:
        st.error(f"Validation Error: Missing values found in critical columns: {', '.join(sorted(critical_columns_with_nan)))}\")
        return False

    if 'age' in df.columns:
        if not pd.api.types.is_integer_dtype(df['age']):
            st.error("Type Error: Incorrect data type for 'age' column. Expected integer type.")
            return False

    st.success("Validation successful.")
    st.subheader("Summary statistics for numeric columns:")
    st.dataframe(df.describe(include=np.number))

    return True


def run_page1():
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
    *   **Aequitas**: An open-source toolkit for bias and fairness auditing.
    *   **Fairlearn**: A Python package for assessing and improving fairness of AI systems.
    *   **IBM AI Fairness 360 (AIF360)**: An extensible open-source toolkit that helps detect and mitigate bias in machine learning models.
    """)

    st.header("3. Synthetic Data Generation")

    st.markdown("""
    To effectively demonstrate AI bias and its mitigation, we need a dataset that is both controllable and interpretable. Real-world datasets can be complex and difficult to attribute bias to specific factors. Therefore, we will generate synthetic data.

    **Business Value**: By creating synthetic data, we can explicitly inject bias related to a sensitive attribute (e.g., gender) into a decision-making process (e.g., loan approval). This allows us to clearly observe how a model trained on such data will exhibit biased behavior and provides a controlled environment to test bias detection and mitigation techniques. This step is crucial for research, development, and demonstrating the impact of unfair data practices without relying on sensitive real-world information.

    **Technical Implementation**: The `generate_synthetic_data` function will create a DataFrame with a specified number of samples. It includes:

    *   **Numeric Features**: `age` (integer, e.g., 18-65) and `income` (float, e.g., normal distribution around a mean).
    *   **Categorical Features**: `location` (e.g., 'Urban', 'Suburban', 'Rural') and `gender` (e.g., 'Male', 'Female').
    *   **Target Variable**: `loan_approval` (binary: 0 or 1). We introduce bias such that the probability of loan approval is higher for one gender group (e.g., 'Male') by a `bias_factor`. This simulates a scenario where historical data might show preferential treatment.

    The formula for loan approval probability ($P(loan_{approval})$) is designed to be influenced by age, income, and critically, gender:

    $$ P(loan_{approval} | age, income, gender) = BaseProb + BiasFactor \times I(gender=Male) + f(age) + g(income) $$

    Where:
    - $BaseProb$ is a baseline approval probability.
    - $I(gender=Male)$ is an indicator function, equal to 1 if the gender is Male, and 0 otherwise. This is where the `bias_factor` explicitly influences the outcome based on gender.
    - $f(age)$ and $g(income)$ are functions representing the influence of age and income on loan approval, generally increasing with higher age and income within reasonable ranges.

    By fixing a `seed`, we ensure that the generated data is reproducible, allowing for consistent experimentation and demonstration of results.
    """)

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

    num_samples_fixed = 1000
    bias_factor_fixed = 0.3
    seed_fixed = 42

    synthetic_data = generate_synthetic_data(num_samples=num_samples_fixed, bias_factor=bias_factor_fixed, seed=seed_fixed)
    st.markdown(f"Generated synthetic data with `num_samples={num_samples_fixed}`, `bias_factor={bias_factor_fixed}`, `seed={seed_fixed}`.")
    st.subheader("First 5 rows of Synthetic Data:")
    st.dataframe(synthetic_data.head())
    st.session_state.synthetic_data = synthetic_data # Store the generated data in session state

    st.markdown("""
    The code above generates a synthetic dataset. We called the `generate_synthetic_data` function with:

    *   `num_samples = 1000`: Creating 1000 rows of data.
    *   `bias_factor = 0.3`: Introducing a significant bias where 'Male' individuals have a 0.3 higher baseline probability of loan approval compared to 'Female' individuals.
    *   `seed = 42`: Ensuring reproducibility, so running this code multiple times will yield the same dataset.

    The printed output shows the first 5 rows of the `synthetic_data` DataFrame. You can observe the generated features (`age`, `income`, `location`, `gender`) and the biased target variable (`loan_approval`). Notice how `loan_approval` is a binary (0 or 1) outcome.
    """)

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
        st.error(f"Validation Error: Missing expected columns: {', '.join(sorted(list(missing_columns)))}\")
        return False

    critical_columns_with_nan = []
    for col in expected_columns:
        if df[col].isnull().any():
            critical_columns_with_nan.append(col)

    if critical_columns_with_nan:
        st.error(f"Validation Error: Missing values found in critical columns: {', '.join(sorted(critical_columns_with_nan)))}\")
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
