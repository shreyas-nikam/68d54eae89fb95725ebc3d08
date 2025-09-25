import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

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

def run_page2():
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
    if st.session_state.synthetic_data is not None:
        synthetic_data = st.session_state.synthetic_data.copy()
        label_encoder = LabelEncoder()
        synthetic_data['gender'] = label_encoder.fit_transform(synthetic_data['gender'])
        synthetic_data['location'] = label_encoder.fit_transform(synthetic_data['location'])

        numerical_features = ['age', 'income']
        scaler = StandardScaler()
        synthetic_data[numerical_features] = scaler.fit_transform(synthetic_data[numerical_features])

        st.subheader("First 5 rows of Preprocessed Data:")
        st.dataframe(synthetic_data.head())
        st.session_state.synthetic_data_preprocessed = synthetic_data # Store preprocessed data
    else:
        st.warning("Please generate data on the 'Overview and Data Generation' page first.")
        return

    st.markdown("""
    The code performs the necessary preprocessing steps on the `synthetic_data` DataFrame.

    First, `LabelEncoder` is applied to the `gender` and `location` columns. This converts the categorical string values (e.g., 'Male', 'Female', 'Urban', 'Suburban', 'Rural') into numerical labels. For example, 'Female' and 'Male' might be converted to 0 and 1 respectively, and similarly for 'location'. This is crucial because machine learning models require numerical input.

    Second, `StandardScaler` is applied to the `age` and `income` columns. These numerical features are scaled to have a mean of 0 and a standard deviation of 1. This standardization ensures that features with larger numerical ranges do not unduly influence the model's learning process.

    The printed output displays the first five rows of the preprocessed `synthetic_data` DataFrame. You can observe that the `gender` and `location` columns now contain integer values, and the `age` and `income` columns contain scaled numerical values, typically centered around zero with small standard deviations. This transformation prepares the data for effective model training.
    """)

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

    st.subheader("Data Splitting Code")
    st.code("""
X = synthetic_data.drop('loan_approval', axis=1)
y = synthetic_data['loan_approval']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

    # Actual execution
    if st.session_state.synthetic_data_preprocessed is not None:
        X = st.session_state.synthetic_data_preprocessed.drop('loan_approval', axis=1)
        y = st.session_state.synthetic_data_preprocessed['loan_approval']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Training set size: {len(X_train)}")
        st.write(f"Testing set size: {len(X_test)}")

        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    else:
        st.warning("Please preprocess data first.")
        return

    st.markdown("""
    The code first separates the features (`X`) from the target variable (`y`) in our `synthetic_data` DataFrame. The `loan_approval` column is dropped from `X` to become our `y`.

    Then, it uses `train_test_split` to divide `X` and `y` into training and testing sets. We allocated 80% of the data for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). The `random_state=42` ensures that this split is consistent every time the code is run.

    The printed output shows the number of samples in the training and testing sets, confirming that our data has been successfully partitioned. For instance, with 1000 total samples, you would expect approximately 800 samples in the training set and 200 in the testing set. This prepares our data for the model training and evaluation phases.
    """)

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

    st.subheader("Model Training Code")
    st.code("""
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
""", language="python")

    # Actual execution
    if st.session_state.X_train is not None:
        model = LogisticRegression(random_state=42)
        model.fit(st.session_state.X_train, st.session_state.y_train)
        y_pred = model.predict(st.session_state.X_test)
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        auc_roc = roc_auc_score(st.session_state.y_test, model.predict_proba(st.session_state.X_test)[:, 1])

        st.write(f"Model Accuracy: {accuracy:.4f}")
        st.write(f"AUC-ROC Score: {auc_roc:.4f}")

        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.auc_roc = auc_roc
    else:
        st.warning("Please split data first.")
        return

    st.markdown("""
    The code above trains a Logistic Regression model using our prepared training data and then evaluates its performance on the test data.

    1.  **Model Initialization and Training**: A `LogisticRegression` model is initialized with `random_state=42` for reproducibility. The `model.fit(X_train, y_train)` command trains the model on the `X_train` features and `y_train` target labels.
    2.  **Prediction**: After training, `model.predict(X_test)` generates binary predictions (0 or 1) for the loan approval outcome on the unseen test set. `model.predict_proba(X_test)[:, 1]` extracts the predicted probabilities of loan approval for each instance in the test set.
    3.  **Evaluation Metrics**: The `accuracy_score` calculates the proportion of correctly predicted instances, while the `roc_auc_score` measures the area under the Receiver Operating Characteristic curve, indicating the model's ability to discriminate between positive and negative classes.

    The printed output displays the `Model Accuracy` and `AUC-ROC Score`. These values represent the baseline performance of our model *before* any explicit bias detection or mitigation techniques are applied. A higher accuracy and AUC-ROC score generally indicate a better performing model. However, these metrics alone do not tell us about the fairness of the predictions, which we will address in the next sections.
    """)

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

    # Actual execution
    if st.session_state.synthetic_data_preprocessed is not None:
        spd = statistical_parity_difference(st.session_state.synthetic_data_preprocessed, 'gender', 'loan_approval', 1) # 1 represents 'Male' (privileged)
        st.write(f"Statistical Parity Difference: {spd:.4f}")
        st.session_state.spd = spd
    else:
        st.warning("Please preprocess data first.")
        return

    st.markdown("""
    The code executes the `statistical_parity_difference` function to quantify the bias in our `synthetic_data`.

    *   `df=synthetic_data`: The DataFrame containing our preprocessed data.
    *   `group_col='gender'`: The sensitive attribute we are examining for bias.
    *   `outcome_col='loan_approval'`: The target variable, representing a favorable outcome (loan approval).
    *   `privileged_group=1`: After `LabelEncoder`, 'Male' was likely encoded as 1 (and 'Female' as 0), making 'Male' our privileged group in this scenario.

    The printed output displays the calculated Statistical Parity Difference (SPD). A positive value indicates that the privileged group (Male) has a higher probability of receiving a loan approval than the unprivileged group (Female). This directly reflects the bias we intentionally introduced during data generation. For example, an SPD of 0.2 means males are 20% more likely to get a loan approved compared to females, demonstrating a clear disparate impact.
    """)

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

    # Actual execution
    if st.session_state.synthetic_data_preprocessed is not None:
        eod = equal_opportunity_difference(st.session_state.synthetic_data_preprocessed, 'gender', 'loan_approval', 1) # 1 represents 'Male' (privileged)
        st.write(f"Equal Opportunity Difference: {eod:.4f}")
        st.session_state.eod = eod
    else:
        st.warning("Please preprocess data first.")
        return

    st.markdown("""
    The code calculates the Equal Opportunity Difference (EOD) using our `equal_opportunity_difference` function. Similar to SPD, it uses:

    *   `df=synthetic_data`: The preprocessed DataFrame.
    *   `group_col='gender'`: The sensitive group column.
    *   `outcome_col='loan_approval'`: The target outcome.
    *   `privileged_group=1`: 'Male' as the privileged group.

    The printed output displays the calculated EOD. A positive EOD indicates that among those who *actually* deserve a loan (i.e., their `loan_approval` is 1), the privileged group (Males) are being correctly identified at a higher rate than the unprivileged group (Females). This metric is crucial because it highlights if the model is systematically failing to grant favorable outcomes to qualified individuals from the unprivileged group, which is a significant fairness concern.
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

df_plot = pd.DataFrame({
    "Metric": metrics_names,
    "Value": original_values,
    "Type": ["Original"] * len(metrics_names)
})

fig = px.bar(df_plot, x='Metric', y='Value', color='Type', barmode='group',
             title='Bias Metrics Comparison: Original',
             labels={'Value': 'Metric Value', 'Metric': 'Bias Metric'},
             color_discrete_map={'Original': '#440154'})
fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Ideal Fairness (0 Difference)", annotation_position="bottom right")

st.plotly_chart(fig)
""", language="python")

    # Actual execution
    if st.session_state.spd is not None and st.session_state.eod is not None:
        metrics_names = ['Statistical Parity Difference', 'Equal Opportunity Difference']
        original_values = [st.session_state.spd, st.session_state.eod]

        df_plot = pd.DataFrame({
            "Metric": metrics_names,
            "Value": original_values,
            "Type": ["Original"] * len(metrics_names)
        })

        fig = px.bar(df_plot, x='Metric', y='Value', color='Type', barmode='group',
                     title='Bias Metrics Comparison: Original',
                     labels={'Value': 'Metric Value', 'Metric': 'Bias Metric'},
                     color_discrete_map={'Original': '#440154'})
        fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Ideal Fairness (0 Difference)", annotation_position="bottom right")

        st.plotly_chart(fig)
    else:
        st.warning("Please complete model training first.")
        return

    st.markdown("""
    The code generates a bar chart comparing the two key bias metrics—Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD)—before applying the reweighting mitigation technique.

    *   The x-axis displays the names of the bias metrics.
    *   The y-axis represents the 'Difference' value for each metric.
    *   Only 'Original' bars are shown here, representing the bias before mitigation.

    The purpose of this chart is to visually demonstrate the initial bias. By observing the height of the bars, we can easily see the magnitude of the unfairness in the model's predictions with respect to the `gender` attribute.
    """)

    st.header("13. Visualization: Feature Importances")

    st.markdown("""
    **Business Value**: Understanding which features most influence a model's decisions is paramount for transparency, interpretability, and debugging. In the context of AI bias, visualizing feature importances can reveal if sensitive attributes (or proxies for them) are disproportionately driving biased outcomes. This insight allows data scientists to identify the root causes of bias, guide feature engineering efforts, and build more ethical and explainable AI systems. For business stakeholders, it provides confidence in knowing *why* a model makes certain predictions.

    **Technical Implementation**: This section generates a bar chart to visualize the coefficients of our Logistic Regression model, which serve as indicators of feature importance.

    *   **Feature Importances Extraction**: For a linear model like Logistic Regression, the absolute values of the coefficients (`model.coef_[0]`) directly reflect the strength and direction of each feature's influence on the outcome. A larger absolute coefficient implies a greater impact.
    *   **DataFrame Creation**: A Pandas DataFrame (`feature_importances_df`) is created to store the feature names and their corresponding importance scores.
    *   **Sorting**: The features are sorted by their importance in descending order, making it easy to identify the most influential factors.
    *   **Bar Chart Visualization**: `plotly.express` is used to create the visualization as a bar chart for clearer representation of 1D importances:
        *   The bar chart displays the `Importance` values, with `Feature` names as labels.
        *   The bars are colored by importance value using a sequential color scale.

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

