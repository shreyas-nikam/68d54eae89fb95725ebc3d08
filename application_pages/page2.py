import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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

def run_page2():
    if 'synthetic_data' not in st.session_state:
        st.warning("Please navigate to 'Data Generation & Baseline Model' first to generate data and train the initial model.")
        return
    
    synthetic_data = st.session_state['synthetic_data']
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']

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

    spd = statistical_parity_difference(synthetic_data, 'gender', 'loan_approval', 1)
    st.write(f"Statistical Parity Difference: {spd:.4f}")

    st.markdown("""
    The code executes the `statistical_parity_difference` function to quantify the bias in our `synthetic_data`.

    *   `df=synthetic_data`: The DataFrame containing our preprocessed data.
    *   `group_col='gender'`: The sensitive attribute we are examining for bias.
    *   `outcome_col='loan_approval'`: The target variable, representing a favorable outcome (loan approval).
    *   `privileged_group=1`: After `LabelEncoder`, 'Male' was likely encoded as 1 (and 'Female' as 0), making 'Male' our privileged group in this scenario.

    The printed output displays the calculated Statistical Parity Difference (SPD). A positive value indicates that the privileged group (Male) has a higher probability of receiving a loan approval than the unprivileged group (Female). This directly reflects the bias we intentionally introduced during data generation. For example, an SPD of 0.2 means males are 20% more likely to get a loan approved compared to females, demonstrating a clear disparate impact.
    """)
    st.session_state['spd'] = spd # Store for next steps

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

    eod = equal_opportunity_difference(synthetic_data, 'gender', 'loan_approval', 1)
    st.write(f"Equal Opportunity Difference: {eod:.4f}")

    st.markdown("""
    The code calculates the Equal Opportunity Difference (EOD) using our `equal_opportunity_difference` function. Similar to SPD, it uses:

    *   `df=synthetic_data`: The preprocessed DataFrame.
    *   `group_col='gender'`: The sensitive group column.
    *   `outcome_col='loan_approval'`: The target outcome.
    *   `privileged_group=1`: 'Male' as the privileged group.

    The printed output displays the calculated EOD. A positive EOD indicates that among those who *actually* deserve a loan (i.e., their `loan_approval` is 1), the privileged group (Males) are being correctly identified at a higher rate than the unprivileged group (Females). This metric is crucial because it highlights if the model is systematically failing to grant favorable outcomes to qualified individuals from the unprivileged group, which is a significant fairness concern.
    """)
    st.session_state['eod'] = eod # Store for next steps

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
""", language="python")

    reweighted_data = reweight(synthetic_data, 'gender', 'loan_approval', 1, 0.2)
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
    st.session_state['reweighted_data'] = reweighted_data # Store for next steps

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

    st.session_state['model_reweighted'] = model_reweighted
    st.session_state['accuracy_reweighted'] = accuracy_reweighted
    st.session_state['auc_roc_reweighted'] = auc_roc_reweighted
    st.session_state['spd_reweighted'] = spd_reweighted
    st.session_state['eod_reweighted'] = eod_reweighted

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
