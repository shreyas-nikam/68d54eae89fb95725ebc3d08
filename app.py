
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will explore the critical aspects of Explainable AI (XAI) focusing on **AI Bias Detection and Mitigation**.
Artificial Intelligence (AI) models are increasingly integrated into critical decision-making processes,
and ensuring their fairness and transparency is paramount. This application provides an interactive tool
to understand, identify, and mitigate different types of bias in machine learning models, specifically
demonstrated through a synthetic loan approval scenario.

We will cover the entire pipeline, from generating synthetic data with intentional bias to applying
mitigation strategies and evaluating their impact using various metrics and visualizations.
""")

# Initialize session state for data storage
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'synthetic_data_preprocessed' not in st.session_state:
    st.session_state.synthetic_data_preprocessed = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'auc_roc' not in st.session_state:
    st.session_state.auc_roc = None
if 'spd' not in st.session_state:
    st.session_state.spd = None
if 'eod' not in st.session_state:
    st.session_state.eod = None
if 'reweighted_data' not in st.session_state:
    st.session_state.reweighted_data = None
if 'accuracy_reweighted' not in st.session_state:
    st.session_state.accuracy_reweighted = None
if 'auc_roc_reweighted' not in st.session_state:
    st.session_state.auc_roc_reweighted = None
if 'spd_reweighted' not in st.session_state:
    st.session_state.spd_reweighted = None
if 'eod_reweighted' not in st.session_state:
    st.session_state.eod_reweighted = None
if 'model_reweighted' not in st.session_state:
    st.session_state.model_reweighted = None

page = st.sidebar.selectbox(label="Navigation", options=["Overview and Data Generation", "Model Training and Bias Detection", "Bias Mitigation and Interactivity"])
if page == "Overview and Data Generation":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Model Training and Bias Detection":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Bias Mitigation and Interactivity":
    from application_pages.page3 import run_page3
    run_page3()
