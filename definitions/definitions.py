import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples, bias_factor, seed):
    """Generates synthetic data with a specified bias."""
    try:
        seed = int(seed)
    except ValueError:
        seed = 42

    np.random.seed(seed)

    if num_samples <= 0:
        return pd.DataFrame()

    age = np.random.randint(20, 60, num_samples)
    income = np.random.randint(30000, 100000, num_samples)
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], num_samples)
    gender = np.random.choice(['Male', 'Female'], num_samples)

    loan_approval = []
    for i in range(num_samples):
        # Introduce bias: Females are less likely to get a loan
        if gender[i] == 'Female':
            probability = 0.4 - bias_factor
        else:
            probability = 0.6 + bias_factor

        # Ensure probability stays within [0, 1]
        probability = max(0, min(1, probability))
        loan_approval.append(np.random.choice([0, 1], p=[1 - probability, probability]))

    data = {
        'age': age,
        'income': income,
        'location': location,
        'gender': gender,
        'loan_approval': loan_approval
    }

    df = pd.DataFrame(data)
    return df

import pandas as pd

def validate_data(df, expected_columns):
    """Validates DataFrame for expected columns, dtypes, and missing values."""

    # Check for missing columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError("Missing expected columns: {}".format(missing_cols))

    # Check for incorrect data types
    expected_dtypes = {'age': ['int64', 'float64'], 'income': ['int64', 'float64'],
                       'location': ['object', 'string'], 'gender': ['object', 'string'],
                       'loan_approval': ['int64', 'float64']}

    for col in expected_columns:
        if col in expected_dtypes:
            if df[col].dtype not in expected_dtypes[col]:
                raise TypeError("Incorrect data type for '{}' column".format(col))

    # Check for missing values
    if df[expected_columns].isnull().any().any():
        raise ValueError("Missing values found in critical columns")

    return True

import pandas as pd

def statistical_parity_difference(df, group_col, outcome_col, privileged_group):
    """Calculates the Statistical Parity Difference."""
    
    # Calculate the proportion of positive outcomes for the privileged group
    privileged_count = df[df[group_col] == privileged_group][outcome_col].mean()
    
    # Calculate the proportion of positive outcomes for the unprivileged group
    unprivileged_count = df[df[group_col] != privileged_group][outcome_col].mean()
    
    # Handle cases where there are no unprivileged or privileged groups
    if pd.isna(privileged_count):
        privileged_count = 0
    if pd.isna(unprivileged_count):
        unprivileged_count = 0
    
    # Return the Statistical Parity Difference
    return privileged_count - unprivileged_count

def equal_opportunity_difference(df, group_col, outcome_col, privileged_group):
    """Calculates the Equal Opportunity Difference."""

    # Filter for positive outcomes
    df_positive_outcome = df[df[outcome_col] == 1]

    # Calculate approval rate for the privileged group
    privileged_positive = df_positive_outcome[df_positive_outcome[group_col] == privileged_group][outcome_col].count()
    privileged_total = df[df[group_col] == privileged_group][outcome_col].count()
    if privileged_total == 0:
        privileged_rate = 0.0
    else:
        privileged_rate = privileged_positive / privileged_total if privileged_total > 0 else 0.0

    # Calculate approval rate for the unprivileged group
    unprivileged_positive = df_positive_outcome[df_positive_outcome[group_col] != privileged_group][outcome_col].count()
    unprivileged_total = df[df[group_col] != privileged_group][outcome_col].count()
    if unprivileged_total == 0:
        unprivileged_rate = 0.0
    else:
        unprivileged_rate = unprivileged_positive / unprivileged_total if unprivileged_total > 0 else 0.0

    # Calculate the difference
    return privileged_rate - unprivileged_rate

import pandas as pd

def reweight(df, group_col, outcome_col, privileged_group, weight):
    """Reweights the data to mitigate bias."""
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column '{outcome_col}' not found in DataFrame.")

    if weight <= 0:
        return df.copy()

    underrepresented_group = df[df[group_col] != privileged_group].copy()

    num_to_duplicate = int(len(underrepresented_group) * weight)

    if num_to_duplicate > 0:
        duplicated_rows = underrepresented_group.sample(n=num_to_duplicate, replace=True)
        df = pd.concat([df, duplicated_rows], ignore_index=True)
    
    return df

def interactive_analysis(bias_factor, reweighting_factor):
                """Re-runs analysis with bias and reweighting."""
                print(f"Running analysis with bias_factor={bias_factor} and reweighting_factor={reweighting_factor}")
                # Simulate some analysis steps. Replace with actual analysis.
                result = bias_factor + reweighting_factor
                print(f"Simulated result: {result}")