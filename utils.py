"""
Shared helpers for the Used Car Price Prediction Streamlit app.

Everything the pages need lives here: data loading, the cleaning/feature-engineering
pipeline (kept identical to the notebook so the app and the saved model agree), the
recorded results of the notebook run, and small formatting helpers.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

EXPECTED_COLUMNS = {
    "brand", "model", "model_year", "milage", "fuel_type", "engine",
    "transmission", "ext_col", "int_col", "accident", "clean_title", "price",
}

# The notebook defined car_age as (max model_year in the data) + 1 - model_year.
# The maximum model year in this dataset is 2024, so the reference year is 2025.
# This is FROZEN as a constant on purpose: the saved model learned car_age on this
# scale, so recomputing it from whatever rows happen to be loaded (or from today's
# date) would silently shift every prediction.
REFERENCE_YEAR = 2025

# The exact 18 columns the saved pipeline expects, in the order it was fitted on.
MODEL_FEATURES = [
    "brand", "model", "model_year", "milage", "fuel_type", "ext_col", "int_col",
    "accident", "is_electric", "horsepower", "engine_liters", "cylinders",
    "hp_missing", "liters_missing", "cyl_missing", "car_age",
    "transmission_group", "clean_title_flag",
]

# The single data-entry error removed in Section 7 of the notebook: a 2005 Maserati
# Quattroporte Base listed at $2,954,083, when every other Quattroporte in the data
# sits between $7,999 and $44,999.
OUTLIER_PRICE = 2_954_083.0

RAW_ROWS = 4009
CLEAN_ROWS = 4008
TRAIN_ROWS = 3206
TEST_ROWS = 802

# ---------------------------------------------------------------------------
# Results recorded from the notebook run. These are reported, not recomputed,
# so the app never has to retrain anything to show its own numbers.
# ---------------------------------------------------------------------------

FINAL_MODEL_NAME = "HistGradientBoosting (tuned)"

TEST_METRICS = {
    "r2": 0.2526,
    "mae": 12697,
    "rmse": 84906,
    "mape": 20.6,
    "median_ae": 4154,
}

# Same held-out test set, minus the two one-of-a-kind exotics (Bugatti Veyron,
# Porsche Carrera GT) that each appear exactly once in the entire dataset.
TEST_METRICS_EX_EXOTICS = {
    "r2": 0.788,
    "mae": 8642,
    "rmse": 21184,
}

CV_RESULTS = pd.DataFrame(
    [
        ("Random Forest",        0.783691, 0.050500,  8883.62, 23613.30, 22.8),
        ("HistGradientBoosting", 0.779507, 0.045141,  8900.87, 23864.00,  1.3),
        ("XGBoost",              0.768569, 0.053886,  9230.51, 24389.89,  1.9),
        ("Linear Regression",    0.673934, 0.098548, 11649.29, 28779.44,  0.2),
        ("Ridge Regression",     0.673073, 0.098548, 11651.56, 28817.67,  0.1),
        ("KNN Regressor",        0.546203, 0.073722, 11921.94, 34570.00,  0.2),
    ],
    columns=["Model", "R² (mean)", "R² (std)", "MAE (mean $)", "RMSE (mean $)", "CV time (s)"],
)

TUNING_COMPARISON = pd.DataFrame(
    [
        ("HistGradientBoosting (tuned)",   22144.37),
        ("Random Forest (default params)", 23613.30),
        ("Random Forest (tuned)",          23672.03),
        ("HistGradientBoosting (default)", 23864.00),
    ],
    columns=["Model", "CV RMSE ($)"],
)

BEST_PARAMS_HGB = {
    "learning_rate": 0.15,
    "max_iter": 200,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 10,
    "max_depth": None,
    "l2_regularization": 0.1,
}

BEST_PARAMS_RF = {
    "n_estimators": 150,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "max_samples": 0.9,
}

# Permutation importance on the test set: mean drop in R² when the column is shuffled.
PERMUTATION_IMPORTANCE = pd.DataFrame(
    [
        ("milage", 0.140073, 0.025991),
        ("horsepower", 0.048885, 0.019408),
        ("brand", 0.042137, 0.009260),
        ("engine_liters", 0.036733, 0.012329),
        ("model", 0.028874, 0.010648),
        ("cylinders", 0.013279, 0.001347),
        ("ext_col", 0.009899, 0.004223),
        ("int_col", 0.006551, 0.003848),
        ("accident", 0.005297, 0.005580),
        ("hp_missing", 0.004186, 0.006527),
        ("transmission_group", 0.002250, 0.001230),
        ("clean_title_flag", 0.000470, 0.003967),
        ("liters_missing", 0.000027, 0.000019),
        ("model_year", 0.000000, 0.000000),
        ("is_electric", -0.000032, 0.000332),
        ("car_age", -0.000181, 0.015114),
        ("fuel_type", -0.000338, 0.001501),
        ("cyl_missing", -0.000816, 0.000245),
    ],
    columns=["Feature", "Importance (drop in R²)", "Std"],
)

# Test-set accuracy sliced by price bracket.
PRICE_BRACKET_REPORT = pd.DataFrame(
    [
        ("< $25K",      294, 0.468888,   3309.76,   4314.65, 26.89),
        ("$25K–$50K",   306, -0.007188,  5362.28,   7236.06, 15.01),
        ("$50K–$100K",  148, -0.238248, 11737.55,  16373.41, 16.74),
        ("$100K–$250K",  44, -0.707621, 38844.59,  49987.29, 25.60),
        ("> $250K",      10, -0.643288, 412301.10, 749034.26, 40.74),
    ],
    columns=["Price bracket", "Cars", "R²", "MAE ($)", "RMSE ($)", "Mean abs % error"],
)

WORST_ERRORS = pd.DataFrame(
    [
        ("Bugatti", "Veyron 16.4 Grand Sport", 2011, 1950995, 120079, 93.8, 1),
        ("Porsche", "Carrera GT Base", 2005, 1599000, 160124, 90.0, 1),
        ("Ford", "GT", 2005, 429998, 62122, 85.6, 1),
        ("Rolls-Royce", "Cullinan", 2022, 399950, 568368, 42.1, 3),
        ("Maserati", "MC20 MC20", 2022, 238900, 107095, 55.2, 1),
        ("Ferrari", "FF Base", 2016, 149900, 271780, 81.3, 1),
        ("Lamborghini", "Huracan Tecnica Coupe", 2023, 359991, 261362, 27.4, 2),
        ("Acura", "NSX Base", 2018, 143900, 45890, 68.1, 2),
        ("Aston", "Martin DBX Base", 2021, 159500, 256187, 60.6, 1),
    ],
    columns=["Brand", "Model", "Year", "Actual ($)", "Predicted ($)",
             "Abs % error", "Times in dataset"],
)

ENCODING_STRATEGY = pd.DataFrame(
    [
        ("milage, horsepower, engine_liters, cylinders, car_age, "
         "hp_missing, liters_missing, cyl_missing, is_electric",
         "StandardScaler",
         "Continuous numerics. Scaling is required by the linear and KNN models and "
         "harmless for the tree ensembles."),
        ("fuel_type, transmission_group, accident, clean_title_flag",
         "OneHotEncoder",
         "Low-cardinality nominal columns (2–7 levels each), so one column per level "
         "stays cheap and readable."),
        ("brand, model, ext_col, int_col",
         "TargetEncoder",
         "High cardinality — 57 brands, 1,898 models, 319 exterior and 156 interior "
         "colours. One-hot would explode the matrix; target encoding keeps one column each."),
        ("model_year", "Dropped",
         "Superseded by car_age, which carries the same information on a more directly "
         "useful scale."),
    ],
    columns=["Columns", "Treatment", "Why"],
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_data():
    """Find and load the raw used-cars CSV sitting next to the app."""
    search_dirs = [BASE_DIR, BASE_DIR.parent, Path.cwd()]
    seen = set()
    for directory in search_dirs:
        for path in sorted(directory.glob("*.csv")):
            if path in seen:
                continue
            seen.add(path)
            try:
                sample = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if EXPECTED_COLUMNS.issubset(sample.columns):
                return pd.read_csv(path), path.name
    raise FileNotFoundError(
        "No compatible used-cars CSV was found. Put used_cars.csv in the same folder as app.py."
    )


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained pipeline saved from the notebook."""
    import joblib

    model_path = BASE_DIR / "used_car_price_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "used_car_price_model.pkl was not found. Put the saved model in the same "
            "folder as app.py."
        )
    return joblib.load(model_path)


def get_data():
    """Raw frame, cleaned frame, and the CSV filename — what most pages want."""
    raw, filename = load_raw_data()
    return raw, prepare_data(raw), filename


# ---------------------------------------------------------------------------
# Parsers for the free-text engine column
# ---------------------------------------------------------------------------

def parse_hp(s):
    m = re.search(r"([\d\.]+)\s*HP", str(s))
    return float(m.group(1)) if m else np.nan


def parse_liters(s):
    m = re.search(r"([\d\.]+)\s*L(?:iter)?\b", str(s))
    return float(m.group(1)) if m else np.nan


def parse_cylinders(s):
    s = str(s)
    m = re.search(r"(\d+)\s*Cylinder", s)
    if m:
        return float(m.group(1))
    m = re.search(r"\bV(\d+)\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"\bI(\d+)\b", s)
    if m:
        return float(m.group(1))
    if "Electric" in s:
        return 0.0
    return np.nan


def bucket_transmission(s):
    s = str(s)
    if "CVT" in s:
        return "CVT"
    if "M/T" in s or "Manual" in s:
        return "Manual"
    if "Dual Shift" in s or "Dual-Shift" in s:
        return "Dual-Shift Automatic"
    if "A/T" in s or "Automatic" in s:
        return "Automatic"
    return "Other/Unknown"


TRANSMISSION_GROUPS = ["Automatic", "Dual-Shift Automatic", "Manual", "CVT", "Other/Unknown"]


# ---------------------------------------------------------------------------
# Cleaning & feature engineering — mirrors Section 5 of the notebook exactly
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_data(df, drop_outlier=True):
    """Turn the raw text-heavy frame into the 18 modelling features.

    Set drop_outlier=False to keep the mispriced Maserati row, which is useful on the
    cleaning page when showing what was removed and why.
    """
    data = df.copy()

    data["price"] = (
        data["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).astype(float)
    )
    data["milage"] = (
        data["milage"].astype(str).str.replace(" mi.", "", regex=False)
        .str.replace(",", "", regex=False).astype(float)
    )

    data["is_electric"] = data["engine"].astype(str).str.contains(
        "Electric", case=False, na=False
    ).astype(int)

    data["horsepower"] = data["engine"].apply(parse_hp)
    data["engine_liters"] = data["engine"].apply(parse_liters)
    data["cylinders"] = data["engine"].apply(parse_cylinders)

    # An electric motor genuinely has 0 litres and 0 cylinders — structural, not missing.
    data.loc[data["is_electric"] == 1, "engine_liters"] = (
        data.loc[data["is_electric"] == 1, "engine_liters"].fillna(0)
    )
    data.loc[data["is_electric"] == 1, "cylinders"] = (
        data.loc[data["is_electric"] == 1, "cylinders"].fillna(0)
    )

    # Record what was missing before filling it — missingness is itself informative.
    data["hp_missing"] = data["horsepower"].isna().astype(int)
    data["liters_missing"] = data["engine_liters"].isna().astype(int)
    data["cyl_missing"] = data["cylinders"].isna().astype(int)

    for col in ["horsepower", "engine_liters", "cylinders"]:
        data[col] = data[col].fillna(data[col].median())

    data["car_age"] = REFERENCE_YEAR - data["model_year"]

    data["transmission_group"] = data["transmission"].apply(bucket_transmission)

    data["fuel_type"] = data["fuel_type"].replace(["–", "not supported"], np.nan)
    data.loc[data["is_electric"] == 1, "fuel_type"] = "Electric"
    data["fuel_type"] = data["fuel_type"].fillna("Unknown")

    data["accident"] = data["accident"].fillna("Unknown")
    data["clean_title_flag"] = np.where(data["clean_title"] == "Yes", "Yes", "No/Unknown")

    if drop_outlier:
        data = data[data["price"] != OUTLIER_PRICE].reset_index(drop=True)

    return data


def build_feature_row(
    brand, model_name, model_year, milage, fuel_type, ext_col, int_col,
    accident, transmission_group, clean_title_flag,
    horsepower, engine_liters, cylinders,
    is_electric=0, hp_missing=0, liters_missing=0, cyl_missing=0,
):
    """Assemble a single-row frame with exactly the columns the model was fitted on."""
    row = {
        "brand": brand,
        "model": model_name,
        "model_year": int(model_year),
        "milage": float(milage),
        "fuel_type": fuel_type,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "is_electric": int(is_electric),
        "horsepower": float(horsepower),
        "engine_liters": float(engine_liters),
        "cylinders": float(cylinders),
        "hp_missing": int(hp_missing),
        "liters_missing": int(liters_missing),
        "cyl_missing": int(cyl_missing),
        "car_age": REFERENCE_YEAR - int(model_year),
        "transmission_group": transmission_group,
        "clean_title_flag": clean_title_flag,
    }
    return pd.DataFrame([row])[MODEL_FEATURES]


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def format_money(value):
    return f"${value:,.0f}"


def safe_options(series):
    return sorted(series.dropna().astype(str).unique().tolist())


def page_header(title, subtitle, section=None):
    """Consistent page heading, with the matching notebook section for cross-reference."""
    st.title(title)
    if section:
        st.caption(f"Notebook {section}")
    st.markdown(f"**{subtitle}**")
    st.divider()


def project_sidebar():
    st.sidebar.markdown("## 🚗 Used Car Price Prediction")
    st.sidebar.caption("Machine Learning Capstone")
    st.sidebar.divider()
    st.sidebar.markdown(f"**Raw records:** {RAW_ROWS:,}")
    st.sidebar.markdown(f"**Clean records:** {CLEAN_ROWS:,}")
    st.sidebar.markdown(f"**Train / test:** {TRAIN_ROWS:,} / {TEST_ROWS:,}")
    st.sidebar.markdown(f"**Final model:** {FINAL_MODEL_NAME}")
