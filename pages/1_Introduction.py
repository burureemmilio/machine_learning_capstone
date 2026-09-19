import streamlit as st

import utils

st.set_page_config(page_title="Introduction", page_icon="📌", layout="wide")
utils.project_sidebar()

utils.page_header(
    "📌 Introduction",
    "Why predict used car prices, who it helps, and how the project is built.",
    section="Section 0",
)

st.markdown("""
### The problem

A used car has no list price. The same model, same year, can trade thousands of dollars apart
depending on mileage, engine, condition and title history — and both sides of a sale are usually
guessing. Dealers price from experience, private sellers price from hope, and buyers have no
independent reference point.

This project treats that as a **supervised regression problem**: learn the relationship between a
car's observable characteristics and its listed price from 4,008 real listings, then use it to
estimate a price for a car the model has never seen.
""")

st.markdown("### Why this is needed, and who it helps")
st.markdown("""
- **Private sellers** price from hope or from a single comparable listing. Overpricing leaves a
  car sitting unsold for months; underpricing quietly loses thousands of dollars.
- **Buyers** have no independent reference for whether an asking price is fair, so negotiation
  ends up driven by confidence rather than evidence.
- **Dealers and marketplaces** price from individual experience, which does not scale and varies
  from one appraiser to the next.
- **Lenders and insurers** need a defensible collateral value for financing and claims, not a
  subjective guess.

A model that turns a car's characteristics into a consistent, repeatable price estimate gives all
four groups the same independent reference point.
""")

st.markdown("### Objectives")
st.markdown("""
1. Turn a raw, text-heavy listings dataset into something a regression model can actually consume.
2. Extract usable numeric signal from free-text fields — particularly the `engine` column, which
   hides horsepower, displacement and cylinder count inside a single descriptive string.
3. Compare several regression algorithms fairly, using the same cross-validation scheme and the
   same preprocessing for each.
4. Tune the strongest candidates and select a final model on cross-validated error, not on a
   guess about which family should win.
5. Evaluate honestly on a test set held out from the very beginning, and investigate *where* the
   model fails rather than reporting a single headline number.
6. Ship the result as an interactive tool a non-technical user can operate.
""")

st.markdown("### Structure of this app")

st.markdown(f"""
| Page | What it covers |
|---|---|
| **Dataset** | The raw data: shape, columns, types, cardinality. |
| **Data Cleaning** | Parsing text into numbers, handling missing values, the one row removed. |
| **EDA** | Target distribution, correlations, and how price relates to every modelling feature. |
| **Feature Engineering** | The {len(utils.MODEL_FEATURES)} model features and the encoding strategy for each. |
| **Model Development** | Six candidate algorithms, the evaluation scheme, and hyperparameter tuning. |
| **Model Comparison** | Cross-validated results, before and after tuning, and how they are measured. |
| **Final Model** | Held-out test performance, error analysis, feature importance, limitations. |
| **Price Predictor** | Enter a car, get an estimate. |
""")

st.success("Next: **Dataset** in the sidebar.")
