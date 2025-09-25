
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab - Explainable AI: AI Bias Detection Tool")
st.divider()
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
*   **plotly**: For data visualization.
*   **Aequitas**: An open-source toolkit for bias and fairness auditing.
*   **Fairlearn**: A Python package for assessing and improving fairness of AI systems.
*   **IBM AI Fairness 360 (AIF360)**: An extensible open-source toolkit that helps detect and mitigate bias in machine learning models.
""")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Generation & Baseline Model", "Bias Detection & Mitigation", "Visualizations & Interactivity"])
if page == "Data Generation & Baseline Model":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Bias Detection & Mitigation":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Visualizations & Interactivity":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
