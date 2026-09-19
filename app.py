import streamlit as st

import utils

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.project_sidebar()

st.title("🚗 Used Car Price Prediction")
st.subheader("Machine Learning Regression Capstone")

st.markdown("""
### Project at a glance

This project builds a machine learning model to predict the listed price of a used car
from characteristics such as:

- Brand and model
- Model year and mileage
- Horsepower, engine size and cylinders
- Fuel type and transmission
- Exterior/interior colour
- Accident history
- Clean-title status

The Streamlit application is designed to replace the notebook during the final demonstration.
Use the **sidebar** to walk through the project from the raw dataset all the way to the
interactive price prediction.
""")

st.info(
    "💡 Presentation flow: Introduction → Dataset → Data Cleaning → EDA → Feature Engineering "
    "→ Model Development → Model Comparison → Final Model → Price Predictor"
)

st.markdown("### What this application demonstrates")
st.markdown("""
1. **Understanding the data** — what the dataset contains and what needed attention.
2. **Cleaning** — converting text values into usable numeric/categorical data.
3. **EDA** — identifying relationships, distributions and patterns.
4. **Feature engineering** — extracting useful information from the raw engine and transmission fields.
5. **Model development** — comparing six regression algorithms using 5-fold cross-validation.
6. **Hyperparameter tuning** — tuning the strongest candidates.
7. **Evaluation** — testing the final model on an untouched test set and investigating its errors.
8. **Prediction** — allowing a user to enter/select a vehicle and receive an estimated price.
""")

st.success("Start with **Introduction** in the sidebar.")
