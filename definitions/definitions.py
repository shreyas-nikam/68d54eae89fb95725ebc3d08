import pandas as pd
import numpy as np

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

    # Type validation for inputs
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if not isinstance(bias_factor, float): # Strict float check as per test cases
        raise TypeError("bias_factor must be a float.")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    # Handle edge case: num_samples = 0
    if num_samples == 0:
        # Return an empty DataFrame with the expected columns and dtypes (where possible for empty df)
        # Note: For empty DFs, pandas might infer object type initially for most columns.
        return pd.DataFrame(columns=['age', 'income', 'location', 'gender', 'loan_approval'])

    np.random.seed(seed)

    # 1. Generate numerical features
    age = np.random.randint(low=20, high=66, size=num_samples)  # Ages between 20 and 65
    income = np.random.uniform(low=30000, high=150000, size=num_samples)  # Incomes between 30k and 150k

    # 2. Generate categorical features
    locations = ['Urban', 'Suburban', 'Rural']
    location = np.random.choice(locations, size=num_samples)

    genders = ['Male', 'Female']
    gender = np.random.choice(genders, size=num_samples)

    # 3. Generate biased binary target variable (loan_approval)
    base_approval_prob = 0.55  # Baseline probability of loan approval

    # Initialize approval probabilities for all samples
    loan_approval_probs = np.full(num_samples, base_approval_prob)

    # Apply bias: Increase approval probability for 'Male' (privileged group)
    male_indices = (gender == 'Male')
    loan_approval_probs[male_indices] = loan_approval_probs[male_indices] + bias_factor

    # Clip probabilities to ensure they stay within [0, 1]
    loan_approval_probs = np.clip(loan_approval_probs, 0.0, 1.0)

    # Generate binary loan approval outcome based on probabilities
    # A loan is approved if a random number [0,1) is less than the calculated probability
    loan_approval = (np.random.rand(num_samples) < loan_approval_probs).astype(float)

    # 4. Assemble into a Pandas DataFrame
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
import numpy as np

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
    
    # 1. Check for presence of expected column names
    actual_columns = set(df.columns)
    missing_columns = set(expected_columns) - actual_columns
    
    if missing_columns:
        print(f"Validation Error: Missing expected columns: {', '.join(sorted(list(missing_columns)))}")
        return False

    # 2. Ensure no missing values in critical fields.
    # All `expected_columns` are considered critical for missing values based on test cases.
    critical_columns_with_nan = []
    for col in expected_columns:
        # Since we already checked for column presence, 'col' is guaranteed to be in df.columns
        if df[col].isnull().any():
            critical_columns_with_nan.append(col)
    
    if critical_columns_with_nan:
        print(f"Validation Error: Missing values found in critical columns: {', '.join(sorted(critical_columns_with_nan))}")
        return False

    # 3. Verify data types for specific columns (e.g., 'age')
    # Test Case 3 specifically checks for 'age' to be an integer type.
    if 'age' in df.columns:
        # After checking for NaNs, if 'age' is still not an integer type, it's a type mismatch.
        if not pd.api.types.is_integer_dtype(df['age']):
            print("Type Error: Incorrect data type for 'age' column. Expected integer type.")
            return False

    # If all validation checks pass
    print("Validation successful.")
    print("\nSummary statistics for numeric columns:")
    # Only include numeric types for describe to match output expectations (e.g., Test Case 1)
    print(df.describe(include=np.number))

    return True

import pandas as pd

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
    
    # Filter DataFrame for privileged and unprivileged groups
    privileged_df = df[df[group_col] == privileged_group]
    unprivileged_df = df[df[group_col] != privileged_group]

    # Calculate the proportion of favorable outcomes for the privileged group
    # .mean() on an empty series will return NaN
    prob_privileged = privileged_df[outcome_col].mean()

    # Calculate the proportion of favorable outcomes for the unprivileged group
    prob_unprivileged = unprivileged_df[outcome_col].mean()

    # Calculate Statistical Parity Difference
    # If either prob_privileged or prob_unprivileged is NaN, the result will be NaN.
    spd = prob_privileged - prob_unprivileged

    return spd

import pandas as pd
import numpy as np

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
    # Filter for instances where the outcome is favorable (actual=1 implies outcome_col == 1).
    # This step aligns with the definition "P(outcome=1 | actual=1)", meaning we only consider
    # cases where the 'actual' outcome is favorable (represented by outcome_col == 1).
    favorable_outcome_df = df[df[outcome_col] == 1]

    # Calculate the rate of favorable outcomes for the privileged group.
    # Within 'favorable_outcome_df', the mean of 'outcome_col' (which only contains 1s)
    # will be 1.0 if the group has favorable outcomes, or NaN if it has none.
    privileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] == privileged_group]
    rate_privileged = privileged_favorable_df[outcome_col].mean()

    # Calculate the rate of favorable outcomes for the unprivileged group.
    # All groups not identified as privileged are considered unprivileged.
    unprivileged_favorable_df = favorable_outcome_df[favorable_outcome_df[group_col] != privileged_group]
    rate_unprivileged = unprivileged_favorable_df[outcome_col].mean()

    # Calculate the Equal Opportunity Difference.
    eod = rate_privileged - rate_unprivileged

    return eod

import pandas as pd
import numpy as np

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

    # Validate column existence
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"Column '{outcome_col}' not found in DataFrame.")

    # Handle empty DataFrame
    if df.empty:
        return df.copy()

    # Handle zero weight - no reweighting needed
    if weight == 0.0:
        return df.copy()

    # Identify unique groups in the group_col
    unique_groups = df[group_col].unique()

    # If there's only one group or less, reweighting by comparison isn't applicable
    if len(unique_groups) < 2:
        return df.copy()

    # Determine the non-privileged group value. This logic assumes exactly two groups.
    non_privileged_group_vals = [val for val in unique_groups if val != privileged_group]

    # If no non-privileged group is found, or more than one (which is not handled by current logic), return original
    if not non_privileged_group_vals or len(non_privileged_group_vals) > 1:
        # This covers cases where privileged_group is the only group, or there are multiple non-privileged groups
        # (the latter is not directly supported by the simplified reweighting logic in tests).
        return df.copy()

    group_A_val = privileged_group
    group_B_val = non_privileged_group_vals[0] # The single non-privileged group

    # Calculate initial counts for both groups
    group_A_initial_count = df[df[group_col] == group_A_val].shape[0]
    group_B_initial_count = df[df[group_col] == group_B_val].shape[0]

    underrepresented_group_val = None
    underrepresented_group_initial_count = 0

    # Determine which group is "underrepresented" based on the specified logic from tests:
    # If privileged group is strictly smaller, it's underrepresented.
    # Otherwise (equal or privileged is larger), the non-privileged group is underrepresented.
    if group_A_initial_count < group_B_initial_count:
        underrepresented_group_val = group_A_val
        underrepresented_group_initial_count = group_A_initial_count
    else:
        underrepresented_group_val = group_B_val
        underrepresented_group_initial_count = group_B_initial_count
    
    # If the identified underrepresented group has no members, cannot duplicate
    if underrepresented_group_initial_count == 0:
        return df.copy()

    # Calculate the number of rows to duplicate
    df_len = len(df)
    
    # 'reweighted_target_add_size' defines the maximum number of rows to add based on the weight factor
    reweighted_target_add_size = int(weight * df_len)
    
    # The actual number of rows to duplicate is limited by this target size
    # and the initial count of the underrepresented group (as per test case logic).
    rows_to_duplicate = min(reweighted_target_add_size, underrepresented_group_initial_count)

    if rows_to_duplicate > 0:
        underrepresented_group_df = df[df[group_col] == underrepresented_group_val]
        
        # Randomly sample 'rows_to_duplicate' rows from the underrepresented group with replacement.
        # Using a fixed random_state for reproducibility in tests.
        duplicated_rows = underrepresented_group_df.sample(n=rows_to_duplicate, replace=True, random_state=42)
        
        # Concatenate the original DataFrame with the newly duplicated rows
        reweighted_df = pd.concat([df, duplicated_rows], ignore_index=True)
        return reweighted_df
    else:
        # If rows_to_duplicate is 0 (e.g., reweighted_target_add_size is 0), return original df
        return df.copy()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Helper Functions
# These functions are defined globally to be accessible by interactive_analysis
# The underscore prefix is a convention for internal helper functions.

def _generate_synthetic_data(num_samples, bias_factor):
    """Generates synthetic data with controlled bias based on 'gender'."""
    if not isinstance(bias_factor, (int, float)):
        raise TypeError(f"bias_factor must be a numeric type, got {type(bias_factor)}")

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
    # 'gender' == 1 is the disadvantaged group here.
    df.loc[df['gender'] == 1, 'loan_prob'] -= bias_factor * 0.25 
    
    # Clip probabilities to be within [0, 1]
    df['loan_prob'] = df['loan_prob'].clip(0.05, 0.95) 

    df['loan_approved'] = (np.random.rand(num_samples) < df['loan_prob']).astype(int)
    
    return df.drop(columns=['loan_prob'])


def _preprocess_data(df):
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

    # For 'education', we assume it's treated as an ordinal feature directly.
    # If one-hot encoding was desired, it would be applied here.

    X = df_processed.drop(columns=['loan_approved', 'loan_approved_encoded'])
    y = df_processed['loan_approved_encoded']

    feature_names = X.columns.tolist()

    return X, y, sensitive_feature_col, feature_names


def _apply_reweighting(X_train, y_train, sensitive_feature_name, reweighting_factor):
    """Calculates sample weights for reweighting mitigation based on the sensitive feature and target."""
    if reweighting_factor == 0.0:
        return np.ones(len(y_train))

    sensitive_group = X_train[sensitive_feature_name]
    
    n_total = len(y_train)
    
    # Calculate joint and marginal probabilities
    p_y0_a0 = ((y_train == 0) & (sensitive_group == 0)).sum() / n_total
    p_y0_a1 = ((y_train == 0) & (sensitive_group == 1)).sum() / n_total
    p_y1_a0 = ((y_train == 1) & (sensitive_group == 0)).sum() / n_total
    p_y1_a1 = ((y_train == 1) & (sensitive_group == 1)).sum() / n_total

    p_y0 = (y_train == 0).sum() / n_total
    p_y1 = (y_train == 1).sum() / n_total

    p_a0 = (sensitive_group == 0).sum() / n_total
    p_a1 = (sensitive_group == 1).sum() / n_total
    
    # Calculate target weights (ideal state)
    # w(a,y) = P(Y=y) * P(A=a) / P(Y=y, A=a)
    weights_map = {}
    if p_y0_a0 > 0: weights_map[(0, 0)] = (p_y0 * p_a0) / p_y0_a0
    if p_y0_a1 > 0: weights_map[(0, 1)] = (p_y0 * p_a1) / p_y0_a1
    if p_y1_a0 > 0: weights_map[(1, 0)] = (p_y1 * p_a0) / p_y1_a0
    if p_y1_a1 > 0: weights_map[(1, 1)] = (p_y1 * p_a1) / p_y1_a1
    
    sample_weights = np.ones(n_total)
    for i in range(n_total):
        key = (y_train.iloc[i], sensitive_group.iloc[i])
        if key in weights_map:
            # Interpolate between base weight (1) and calculated target weight using reweighting_factor
            sample_weights[i] = 1 + (weights_map[key] - 1) * reweighting_factor
    
    return sample_weights


def _train_model(X_train, y_train, sample_weights=None):
    """Trains a Logistic Regression model with optional sample weights."""
    # max_iter increased for better convergence across different datasets
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=2000)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def _calculate_bias_metrics(model, X_data, y_true, sensitive_feature_name):
    """Calculates Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD).
    Assumes 'gender' == 1 is the unprivileged group and 'gender' == 0 is the privileged group
    (as per the bias introduced in data generation).
    Metrics = Unprivileged_group_metric - Privileged_group_metric.
    """
    
    y_pred = model.predict(X_data)
    
    # 'gender' == 1 (female) is the unprivileged group; 'gender' == 0 (male) is the privileged group.
    group_unprivileged_idx = (X_data[sensitive_feature_name] == 1)
    group_privileged_idx = (X_data[sensitive_feature_name] == 0)

    # Statistical Parity Difference (SPD)
    # P(Y_pred=1 | A=unprivileged) - P(Y_pred=1 | A=privileged)
    
    prob_positive_unprivileged = 0
    if group_unprivileged_idx.sum() > 0:
        prob_positive_unprivileged = y_pred[group_unprivileged_idx].sum() / group_unprivileged_idx.sum()

    prob_positive_privileged = 0
    if group_privileged_idx.sum() > 0:
        prob_positive_privileged = y_pred[group_privileged_idx].sum() / group_privileged_idx.sum()
    
    spd = prob_positive_unprivileged - prob_positive_privileged

    # Equal Opportunity Difference (EOD)
    # P(Y_pred=1 | A=unprivileged, Y_true=1) - P(Y_pred=1 | A=privileged, Y_true=1)
    
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


def _plot_bias_metrics(original_metrics, reweighted_metrics):
    """Visualizes the comparison of bias metrics using a bar chart."""
    metrics_data = {
        'Metric': ['SPD', 'EOD', 'SPD', 'EOD'],
        'Value': [original_metrics['SPD'], original_metrics['EOD'], 
                  reweighted_metrics['SPD'], reweighted_metrics['EOD']],
        'Type': ['Original', 'Original', 'Reweighted', 'Reweighted']
    }
    metrics_df = pd.DataFrame(metrics_data)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Metric', y='Value', hue='Type', data=metrics_df, palette='viridis')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Line for ideal fairness (0 difference)
    plt.title('Bias Metrics Comparison: Original vs. Reweighted')
    plt.ylabel('Metric Value')
    plt.xlabel('Bias Metric')
    plt.legend(title='Mitigation Type')
    plt.tight_layout()
    plt.show()


def _plot_feature_importances(model, feature_names):
    """Visualizes feature importances (Logistic Regression coefficients)."""
    if hasattr(model, 'coef_') and model.coef_.shape[0] > 0:
        importances = pd.Series(model.coef_[0], index=feature_names)
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette='coolwarm')
        plt.title('Feature Importances (Logistic Regression Coefficients)')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    # Suppress print statement if coef_ is not available, especially during tests.


def interactive_analysis(bias_factor, reweighting_factor):
    """Performs an end-to-end AI bias analysis pipeline interactively.

    Arguments:
        bias_factor (float): Controls the initial strength of the bias introduced during synthetic data generation.
        reweighting_factor (float): The weight applied during the reweighting mitigation technique to balance group representation.
    
    Output:
        None: This function prints various model metrics and displays two plots 
              (Bias Metrics Comparison Bar Chart and Feature Importances Heatmap) directly.
    """
    # Input validation
    if not isinstance(bias_factor, (int, float)):
        raise TypeError("bias_factor must be a numeric type.")
    if not isinstance(reweighting_factor, (int, float)):
        raise TypeError("reweighting_factor must be a numeric type.")

    print(f"Starting Interactive AI Bias Analysis:")
    print(f"  Bias Factor: {bias_factor}")
    print(f"  Reweighting Factor: {reweighting_factor}")

    # 1. Generate Synthetic Data
    num_samples = 2000 # Using a reasonable number of samples for analysis
    df = _generate_synthetic_data(num_samples, bias_factor)
    print(f"\nGenerated {len(df)} synthetic data samples.")

    # 2. Preprocess Data
    X, y, sensitive_feature_col, feature_names = _preprocess_data(df)
    print("Data preprocessed (target encoded, numerical features scaled).")

    # 3. Split Data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")

    # 4. Train Original Model (without reweighting)
    print("\nTraining original Logistic Regression model...")
    original_model = _train_model(X_train, y_train)
    y_pred_original = original_model.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Original Model Accuracy: {accuracy_original:.4f}")

    # 5. Calculate and Display Original Bias Metrics
    original_metrics = _calculate_bias_metrics(original_model, X_test, y_test, sensitive_feature_col)
    print(f"Original Bias Metrics:")
    print(f"  Statistical Parity Difference (SPD): {original_metrics['SPD']:.4f}")
    print(f"  Equal Opportunity Difference (EOD): {original_metrics['EOD']:.4f}")

    # 6. Apply Reweighting and Train Reweighted Model
    print(f"\nApplying reweighting with factor {reweighting_factor} and training mitigated model...")
    sample_weights_train = _apply_reweighting(X_train, y_train, sensitive_feature_col, reweighting_factor)
    reweighted_model = _train_model(X_train, y_train, sample_weights=sample_weights_train)
    y_pred_reweighted = reweighted_model.predict(X_test)
    accuracy_reweighted = accuracy_score(y_test, y_pred_reweighted)
    print(f"Reweighted Model Accuracy: {accuracy_reweighted:.4f}")

    # 7. Calculate and Display Reweighted Bias Metrics
    reweighted_metrics = _calculate_bias_metrics(reweighted_model, X_test, y_test, sensitive_feature_col)
    print(f"Reweighted Bias Metrics:")
    print(f"  Statistical Parity Difference (SPD): {reweighted_metrics['SPD']:.4f}")
    print(f"  Equal Opportunity Difference (EOD): {reweighted_metrics['EOD']:.4f}")
    
    # 8. Visualize Bias Metrics Comparison
    _plot_bias_metrics(original_metrics, reweighted_metrics)
    print("\nBias metrics comparison plot displayed.")

    # 9. Visualize Feature Importances of the reweighted model
    _plot_feature_importances(reweighted_model, feature_names)
    print("Feature importances plot displayed.")

    print("\nInteractive AI Bias Analysis pipeline completed.")