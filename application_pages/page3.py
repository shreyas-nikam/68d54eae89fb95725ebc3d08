
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

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

    fig_interactive_bias = px.bar(metrics_df, x='Metric', y='Value', color='Type', barmode='group',
                                  title='Bias Metrics Comparison: Original vs. Reweighted',
                                  labels={'Value': 'Metric Value', 'Metric': 'Bias Metric'},
                                  color_discrete_map={'Original': '#440154', 'Reweighted': '#21908d'})
    fig_interactive_bias.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Ideal Fairness (0 Difference)", annotation_position="bottom right")
    st.plotly_chart(fig_interactive_bias)


def _plot_feature_importances_interactive(model, feature_names):
    """Visualizes feature importances (Logistic Regression coefficients)."""
    if hasattr(model, 'coef_') and model.coef_.shape[0] > 0:
        importances = pd.Series(model.coef_[0], index=feature_names)
        importances = importances.sort_values(ascending=True) # Sort ascending for horizontal bar plot

        fig_interactive_imp = px.bar(x=importances.values, y=importances.index,
                                     orientation='h',
                                     title='Feature Importances (Logistic Regression Coefficients)',
                                     labels={'x': 'Coefficient Value', 'y': 'Feature'},
                                     color=importances.values,
                                     color_continuous_scale=px.colors.diverging.Coolwarm)
        st.plotly_chart(fig_interactive_imp)


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

def run_page3():
    if st.session_state.synthetic_data_preprocessed is None or st.session_state.model is None:
        st.warning("Please complete 'Overview and Data Generation' and 'Model Training and Bias Detection' pages first.")
        return

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

    $$ P_{reweighted}(Y=y, A=a) = \frac{\sum_{i \in (Y=y, A=a)} w_i}{\sum_{i} w_i} $$

    Where $w_i$ are the weights assigned to each sample. In our duplication method, $w_i$ for duplicated samples is effectively $>1$, while for others it is $1$.
    """)

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
    if st.session_state.synthetic_data_preprocessed is not None:
        reweighted_data = reweight(st.session_state.synthetic_data_preprocessed.copy(), 'gender', 'loan_approval', 1, 0.2) # Weight for 'Female' is 0.2
        st.write(f"Original data size: {len(st.session_state.synthetic_data_preprocessed)}")
        st.write(f"Reweighted data size: {len(reweighted_data)}")
        st.session_state.reweighted_data = reweighted_data
    else:
        st.warning("Please preprocess data first.")
        return

    st.markdown("""
    The code executes the `reweight` function to mitigate bias in our `synthetic_data`.

    *   `df=synthetic_data`: The original DataFrame.
    *   `group_col='gender'`: The sensitive attribute.
    *   `outcome_col='loan_approval'`: The outcome variable.
    *   `privileged_group=1`: 'Male' as the privileged group. This means the other group (Females, encoded as 0) will be considered for reweighting.
    *   `weight=0.2`: A weighting factor. This determines the extent of duplication for the underrepresented group. In this implementation, it means we aim to add up to 20% of the original DataFrame's length in duplicated rows from the underrepresented group.

    The printed output shows the size of the `original data` and the `reweighted data`. You will observe that the `reweighted_data` DataFrame has a larger number of rows compared to the original. This increase in size is due to the duplication of samples from the underrepresented group (Females in this case), thereby increasing their representation in the dataset. This modified dataset will then be used to retrain our model, aiming to reduce the observed bias.
    """)

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
    if st.session_state.reweighted_data is not None and st.session_state.synthetic_data_preprocessed is not None:
        X_reweighted = st.session_state.reweighted_data.drop('loan_approval', axis=1)
        y_reweighted = st.session_state.reweighted_data['loan_approval']
        X_train_reweighted, X_test_reweighted, y_train_reweighted, y_test_reweighted = train_test_split(X_reweighted, y_reweighted, test_size=0.2, random_state=42)
        model_reweighted = LogisticRegression(random_state=42)
        model_reweighted.fit(X_train_reweighted, y_train_reweighted)
        y_pred_reweighted = model_reweighted.predict(X_test_reweighted)
        accuracy_reweighted = accuracy_score(y_test_reweighted, y_pred_reweighted)
        auc_roc_reweighted = roc_auc_score(y_test_reweighted, model_reweighted.predict_proba(X_test_reweighted)[:, 1])
        # Need to re-import these functions from page2 or define them here if not imported globally
        from application_pages.page2 import statistical_parity_difference, equal_opportunity_difference
        spd_reweighted = statistical_parity_difference(st.session_state.reweighted_data, 'gender', 'loan_approval', 1)
        eod_reweighted = equal_opportunity_difference(st.session_state.reweighted_data, 'gender', 'loan_approval', 1)

        st.write(f"Reweighted Model Accuracy: {accuracy_reweighted:.4f}")
        st.write(f"Reweighted AUC-ROC Score: {auc_roc_reweighted:.4f}")
        st.write(f"Reweighted Statistical Parity Difference: {spd_reweighted:.4f}")
        st.write(f"Reweighted Equal Opportunity Difference: {eod_reweighted:.4f}")

        st.session_state.accuracy_reweighted = accuracy_reweighted
        st.session_state.auc_roc_reweighted = auc_roc_reweighted
        st.session_state.spd_reweighted = spd_reweighted
        st.session_state.eod_reweighted = eod_reweighted
        st.session_state.model_reweighted = model_reweighted
    else:
        st.warning("Please reweight data first.")
        return

    st.markdown("""
    The code section performs the critical step of retraining our model on the reweighted data and then re-evaluating its performance and fairness metrics.

    1.  **Data Preparation**: The `reweighted_data` DataFrame, which now has an increased representation of the previously underrepresented group, is split into new training and testing sets.
    2.  **Model Retraining**: A fresh `LogisticRegression` model (`model_reweighted`) is trained on this reweighted training data. This new model learns from the adjusted distribution, aiming to reduce bias.
    3.  **Performance Evaluation**: The retrained model's `accuracy` and `AUC-ROC score` are calculated on the reweighted test set. These metrics are then printed, allowing for a direct comparison with the original model's performance.
    4.  **Bias Re-evaluation**: Crucially, the `statistical_parity_difference` and `equal_opportunity_difference` functions are called again, this time using the `reweighted_data`. The new SPD and EOD values reflect the impact of the reweighting mitigation strategy.

    By comparing these `Reweighted Model Accuracy`, `Reweighted AUC-ROC Score`, `Reweighted Statistical Parity Difference`, and `Reweighted Equal Opportunity Difference` values with their original counterparts, we can analyze the trade-off. Ideally, we would see the bias metrics (SPD and EOD) move closer to zero (indicating reduced bias), while the accuracy and AUC-ROC scores remain comparable or improve. This comparison helps us understand the effectiveness of reweighting in promoting fairness without unduly compromising predictive power.
    """)

    st.header("12. Visualization: Bias Metrics Comparison")

    st.markdown("""
    **Business Value**: Visualizing bias metrics before and after mitigation is essential for clear communication and impact assessment. Numerical metrics can be abstract, but a compelling bar chart immediately highlights the reduction in unfairness achieved by mitigation strategies. This visual evidence supports ethical decision-making, stakeholder communication, and demonstrates accountability in building fair AI systems. It allows practitioners and non-technical stakeholders alike to quickly grasp the effectiveness of fairness interventions.

    **Technical Implementation**: This section generates a bar chart to visually compare the Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) before and after applying the reweighting mitigation technique.

    *   **Data Preparation**: We gather the calculated `spd`, `eod` (original values) and `spd_reweighted`, `eod_reweighted` (values after mitigation).
    *   **Bar Chart Creation**: `plotly.graph_objects` is used to create a bar chart. Two sets of bars are plotted for each metric (SPD and EOD):
        *   One set represents the 'Original' bias metric values.
        *   The second set represents the 'Reweighted' bias metric values.
    *   **Labels and Title**: The chart is appropriately titled 'Bias Metrics Comparison', with 'Difference' on the y-axis and the specific 'Metrics' on the x-axis.
    *   **Legend**: A legend clarifies which bars correspond to 'Original' and 'Reweighted' metrics.

    This bar chart provides an intuitive and immediate visual comparison, allowing us to quickly assess the effectiveness of the reweighting strategy in reducing the observed biases. Ideally, the bars for the 'Reweighted' metrics should be closer to zero compared to the 'Original' metrics, indicating a successful reduction in bias.
    """)

    st.subheader("Bias Metrics Comparison Plotting Code")
    st.code("""
metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
original_values = [st.session_state.spd, st.session_state.eod]
reweighted_values = [st.session_state.spd_reweighted, st.session_state.eod_reweighted]

df_plot = pd.DataFrame({
    "Metric": metrics_names * 2,
    "Value": original_values + reweighted_values,
    "Type": ["Original"] * len(metrics_names) + ["Reweighted"] * len(metrics_names)
})

fig = go.Figure()
fig.add_trace(go.Bar(name='Original', x=df_plot[df_plot['Type'] == 'Original']['Metric'], y=df_plot[df_plot['Type'] == 'Original']['Value']))
fig.add_trace(go.Bar(name='Reweighted', x=df_plot[df_plot['Type'] == 'Reweighted']['Metric'], y=df_plot[df_plot['Type'] == 'Reweighted']['Value']))

fig.update_layout(
    title='Bias Metrics Comparison',
    yaxis_title='Difference',
    barmode='group'
)
fig.add_shape(type="line", x0=-0.5, y0=0, x1=len(metrics_names)-0.5, y1=0, xref="x", yref="y", line=dict(color="grey", width=1, dash="dash"))

st.plotly_chart(fig)
""", language="python")

    # Actual execution
    if st.session_state.spd is not None and st.session_state.eod is not None and st.session_state.spd_reweighted is not None and st.session_state.eod_reweighted is not None:
        metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
        original_values = [st.session_state.spd, st.session_state.eod]
        reweighted_values = [st.session_state.spd_reweighted, st.session_state.eod_reweighted]

        df_plot = pd.DataFrame({
            "Metric": metrics_names * 2,
            "Value": original_values + reweighted_values,
            "Type": ["Original"] * len(metrics_names) + ["Reweighted"] * len(metrics_names)
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=df_plot[df_plot['Type'] == 'Original']['Metric'], y=df_plot[df_plot['Type'] == 'Original']['Value']))
        fig.add_trace(go.Bar(name='Reweighted', x=df_plot[df_plot['Type'] == 'Reweighted']['Metric'], y=df_plot[df_plot['Type'] == 'Reweighted']['Value']))

        fig.update_layout(
            title='Bias Metrics Comparison',
            yaxis_title='Difference',
            barmode='group'
        )
        fig.add_shape(type="line", x0=-0.5, y0=0, x1=len(metrics_names)-0.5, y1=0, xref="x", yref="y", line=dict(color="grey", width=1, dash="dash"))

        st.plotly_chart(fig)
    else:
        st.warning("Please complete model training and reweighting first.")
        return

    st.markdown("""
    The code generates a bar chart comparing the two key bias metrics—Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD)—before and after applying the reweighting mitigation technique.

    *   The x-axis displays the names of the bias metrics.
    *   The y-axis represents the 'Difference' value for each metric.
    *   For each metric, two bars are shown: an 'Original' bar (representing the bias before mitigation) and a 'Reweighted' bar (representing the bias after mitigation).

    The purpose of this chart is to visually demonstrate the impact of reweighting. By observing the height of the bars, we can easily see if the mitigation strategy has successfully moved the bias metrics closer to zero. A significant reduction in the magnitude of the 'Reweighted' bars compared to the 'Original' bars indicates that the reweighting technique has been effective in reducing the unfairness in the model's predictions with respect to the `gender` attribute.
    """)

    st.header("13. Visualization: Feature Importances")

    st.markdown("""
    **Business Value**: Understanding which features most influence a model's decisions is paramount for transparency, interpretability, and debugging. In the context of AI bias, visualizing feature importances can reveal if sensitive attributes (or proxies for them) are disproportionately driving biased outcomes. This insight allows data scientists to identify the root causes of bias, guide feature engineering efforts, and build more ethical and explainable AI systems. For business stakeholders, it provides confidence in knowing *why* a model makes certain predictions.

    **Technical Implementation**: This section generates a heatmap to visualize the coefficients of our Logistic Regression model, which serve as indicators of feature importance.

    *   **Feature Importances Extraction**: For a linear model like Logistic Regression, the absolute values of the coefficients (`model.coef_[0]`) directly reflect the strength and direction of each feature's influence on the outcome. A larger absolute coefficient implies a greater impact.
    *   **DataFrame Creation**: A Pandas DataFrame (`feature_importances_df`) is created to store the feature names and their corresponding importance scores.
    *   **Sorting**: The features are sorted by their importance in descending order, making it easy to identify the most influential factors.
    *   **Heatmap Visualization**: `plotly.express` is used to create the visualization as a bar chart for clearer representation of 1D importances:
        *   The bar chart displays the `Importance` values, with `Feature` names as labels.
        *   The bars are colored by importance value using a diverging color scale.

    This chart helps us understand which features the model relies on most heavily. If a sensitive feature (like 'gender') or a feature highly correlated with it (a 'proxy' feature) shows a high importance, it further supports the finding of bias and points to areas for further investigation or mitigation.
    """)

    st.subheader("Feature Importances Plotting Code")
    st.code("""
feature_importances = abs(st.session_state.model.coef_[0])

feature_importances_df = pd.DataFrame({
    'Feature': st.session_state.X.columns,
    'Importance': feature_importances
})

feature_importances_df = feature_importances_df.sort_values('Importance', ascending=True) # Sort ascending for horizontal bar plot

fig_imp = px.bar(feature_importances_df, x='Importance', y='Feature',
                 orientation='h',
                 title='Feature Importances (Logistic Regression Coefficients)',
                 color='Importance',
                 color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_imp)
""", language="python")

    # Actual execution
    if st.session_state.model is not None and st.session_state.X is not None:
        feature_importances = abs(st.session_state.model.coef_[0])

        feature_importances_df = pd.DataFrame({
            'Feature': st.session_state.X.columns,
            'Importance': feature_importances
        })

        feature_importances_df = feature_importances_df.sort_values('Importance', ascending=True) # Sort ascending for horizontal bar plot

        fig_imp = px.bar(feature_importances_df, x='Importance', y='Feature',
                         orientation='h',
                         title='Feature Importances (Logistic Regression Coefficients)',
                         color='Importance',
                         color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_imp)
    else:
        st.warning("Please complete model training first.")
        return

    st.markdown("""
    The code generates a bar chart that visually represents the importance of each feature in the Logistic Regression model.

    1.  **Extracting Importances**: It retrieves the absolute coefficients of the trained `model`. For linear models like Logistic Regression, the magnitude of these coefficients indicates how much each feature contributes to the prediction. A larger absolute value means a stronger influence.
    2.  **Structuring for Visualization**: These importances are then organized into a DataFrame along with their corresponding feature names, and sorted in descending order of importance.
    3.  **Bar Chart Generation**: `plotly.express` is used to create the visualization. The bar chart displays the feature names on one axis and their importance values. The color intensity and numerical annotations make it easy to quickly identify which features have the highest impact.

    By examining this bar chart, we can gain insights into which features the model primarily relies on to make its `loan_approval` predictions. If the `gender` feature (or any other feature that might serve as a proxy for it) shows a high level of importance, it reinforces our understanding of where the model's bias might be originating. This visualization is crucial for understanding the model's decision-making process and for pinpointing areas that might require further attention in bias mitigation efforts.
    """)

    st.header("14. Visualization: Interactivity")

    st.markdown("""
    **Business Value**: Interactivity in an AI bias detection tool is invaluable for exploring the complex interplay between initial data biases, mitigation strategies, and resulting model fairness and performance. It transforms a static analysis into a dynamic learning environment, enabling users to gain deeper, intuitive understanding of these trade-offs. This directly supports research, development, and ethical AI deployment by allowing for rapid experimentation and visual assessment of different scenarios.
    """)

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
    *   **Feature Importances Plot**: This bar chart shows the updated feature importances for the reweighted model.

    This interactive experience provides immediate feedback on the trade-offs between model accuracy and fairness. You can observe how a higher `bias_factor` might lead to larger original bias metrics, and how increasing the `reweighting_factor` generally pushes these metrics closer to zero, potentially with some impact on accuracy. This hands-on exploration deepens the understanding of bias dynamics and mitigation strategies in AI.
    """)

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
    *   **Visual Analysis**: We utilized visualizations (bar charts for bias comparison, bar charts for feature importances) to clearly communicate the presence of bias and the impact of mitigation strategies.
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

