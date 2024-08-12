import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import os
import dtale
from streamlit_pandas_profiling import st_profile_report
# Machine Learning Staff
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, pull as pull_reg, save_model as save_model_reg, create_model as create_model_reg
from pycaret.classification import setup as setup_cls, compare_models as compare_models_cls, pull as pull_cls, save_model as save_model_cls, create_model as create_model_cls
with st.sidebar:
    st.image('/home/cse/Projects/autoML/images/automl.png')
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, dtale and PyCaret")

df = pd.DataFrame()

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col= None)
   
if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    d = dtale.show(df, open_browser=False)
    components.iframe(d._url, width=1000, height=600)

if choice == "ML":
    st.title("Machine Learning")
    ml_type = st.selectbox("Choose ML Type", ["Regression", "Classification"])
    target = st.selectbox("Select Your Target", df.columns)
    if ml_type == "Regression":
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net", "Decision Tree", "Random Forest", "Support Vector Regression", "Gradient Boosting Regressor", "Extreme Gradient Boosting Regressor", "Light Gradient Boosting Regressor"])
    elif ml_type == "Classification":
        model_type = st.selectbox("Choose Model Type", ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine", "Naive Bayes", "Gradient Boosting", "Extreme Gradient Boosting", "Light Gradient Boosting", "CatBoost Classifier"])
    col1, col2, col3, col4 = st.columns([1,5,5,5])
    if col2.button("Train Model"):
        if ml_type == "Regression":
            setup_reg = setup_reg(df, target=target, verbose=False)
            setup_df = pull_reg()
            st.info("This is the Regression Experiment settings")
            st.dataframe(setup_df)
            if model_type == "Linear Regression":
                best_model = create_model_reg('lr')
            elif model_type == "Ridge Regression":
                best_model = create_model_reg('ridge')
            elif model_type == "Lasso Regression":
                best_model = create_model_reg('lasso')
            if model_type == "Elastic Net":
                best_model = create_model_reg('en')
            elif model_type == "Decision Tree":
                best_model = create_model_reg('dt')
            elif model_type == "Random Forest":
                best_model = create_model_reg('rf')
            elif model_type == "Support Vector Regression":
                best_model = create_model_reg('svm')
            elif model_type == "Gradient Boosting Regressor":
                best_model = create_model_reg('gbr')
            elif model_type == "Extreme Gradient Boosting Regressor":
                best_model = create_model_reg('xgboost')
            elif model_type == "Light Gradient Boosting Regressor":
                best_model = create_model_reg('lightgbm')
            compare_df = pull_reg()
            st.info("This is the Regression Model")
            st.dataframe(compare_df)
            best_model
            save_model_reg(best_model, 'best_model')
        elif ml_type == "Classification":
            setup_cls = setup_cls(df, target=target, verbose=False)
            setup_df = pull_cls()
            st.info("This is the Classification Experiment settings")
            st.dataframe(setup_df)
            if model_type == "Logistic Regression":
                best_model = create_model_cls('lr')
            elif model_type == "K-Nearest Neighbors":
                best_model = create_model_cls('knn')
            elif model_type == "Decision Tree":
                best_model = create_model_cls('dt')
            elif model_type == "Naive Bayes":
                best_model = create_model_cls('nb')
            elif model_type == "Support Vector Machine":
                best_model = create_model_cls('svm')
            elif model_type == "Random Forest":
                best_model = create_model_cls('rf')
            elif model_type == "Gradient Boosting":
                best_model = create_model_cls('gbc')
            elif model_type == "Extreme Gradient Boosting":
                best_model = create_model_cls('xgboost')
            elif model_type == "Light Gradient Boosting":
                best_model = create_model_cls('lightgbm')
            elif model_type == "CatBoost Classifier":
                best_model = create_model_cls('catboost')
            compare_df = pull_cls()
            st.info("This is the Classification Model")
            st.dataframe(compare_df)
            best_model
            save_model_cls(best_model, 'best_model')
    if col3.button("Train Model On All"):
        if ml_type == "Regression":
            setup_reg = setup_reg(df, target=target, verbose=False)
            setup_df = pull_reg()
            st.info("This is the Regression Experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models_reg()
            compare_df = pull_reg()
            st.info("This is the Regression Model")
            st.dataframe(compare_df)
            best_model
            save_model_reg(best_model, 'best_model')
        elif ml_type == "Classification":
            setup_cls = setup_cls(df, target=target, verbose=False)
            setup_df = pull_cls()
            st.info("This is the Classification Experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models_cls()
            compare_df = pull_cls()
            st.info("This is the Classification Model")
            st.dataframe(compare_df)
            best_model
            save_model_cls(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
