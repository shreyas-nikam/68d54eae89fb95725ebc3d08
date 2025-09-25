import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
        st.error(f"Validation Error: Missing expected columns: {\