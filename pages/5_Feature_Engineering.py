import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Feature Engineering", page_icon="🛠️", layout="wide")
utils.project_sidebar()

utils.page_header(
    "🛠️ Feature Engineering & Encoding",
    "The 18 columns the model sees, where each came from, and how each is encoded.",
    section="Sections 9–11",
)

try:
    raw, _ = utils.load_raw_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = utils.prepare_data(raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw columns", raw.shape[1])
c2.metric("Engineered", 9)
c3.metric("Dropped", 4)
c4.metric("Features to model", len(utils.MODEL_FEATURES))

st.divider()
st.markdown("## Where each feature came from")

origins = pd.DataFrame([
    ("brand", "Raw", "Manufacturer, used as-is."),
    ("model", "Raw", "Model and trim string, used as-is."),
    ("model_year", "Raw", "Kept in the frame but dropped by the ColumnTransformer — see below."),
    ("milage", "Parsed", 'Stripped of " mi." and commas, cast to float.'),
    ("fuel_type", "Cleaned", 'Placeholders unified; Electric recovered from engine text; '
                             'remainder → "Unknown".'),
    ("ext_col", "Raw", "Exterior colour."),
    ("int_col", "Raw", "Interior colour."),
    ("accident", "Cleaned", 'Missing → "Unknown".'),
    ("is_electric", "Engineered", 'Binary flag from "Electric" appearing in the engine text.'),
    ("horsepower", "Engineered", "Regex-extracted from engine text, median-filled."),
    ("engine_liters", "Engineered", "Regex-extracted; 0 for electric; median-filled otherwise."),
    ("cylinders", "Engineered", "Regex-extracted; 0 for electric; median-filled otherwise."),
    ("hp_missing", "Engineered", "1 if horsepower had to be imputed."),
    ("liters_missing", "Engineered", "1 if displacement had to be imputed."),
    ("cyl_missing", "Engineered", "1 if cylinder count had to be imputed."),
    ("car_age", "Engineered", f"{utils.REFERENCE_YEAR} − model_year."),
    ("transmission_group", "Engineered", "62 raw strings bucketed into 5 groups."),
    ("clean_title_flag", "Engineered", 'Binary: "Yes" vs "No/Unknown".'),
], columns=["Feature", "Origin", "How it was produced"])

st.dataframe(origins, width="stretch", hide_index=True, height=560)

st.markdown("### Columns deliberately dropped")
st.markdown("""
| Dropped | Why |
|---|---|
| `engine` | Fully superseded — its three numeric signals were extracted into `horsepower`, `engine_liters` and `cylinders`. As a raw string it has 1,146 near-unique values and no usable structure. |
| `transmission` | Superseded by `transmission_group`. |
| `clean_title` | Superseded by `clean_title_flag`. |
| `model_year` | Superseded by `car_age`, which carries identical information on a more useful scale. It is passed into the pipeline but the `ColumnTransformer` routes it to `remainder='drop'`. |
""")

st.info(
    "Keeping both `model_year` and `car_age` would be exact redundancy — they are a perfect "
    "linear function of each other. Dropping one costs nothing. This is visible in the results "
    "later: `model_year` scores exactly **0.000000** on permutation importance, because the "
    "model never receives it."
)

st.divider()
st.markdown("## Encoding strategy")

st.markdown(
    "Three different treatments, chosen by cardinality and by what each model family needs:"
)
st.dataframe(utils.ENCODING_STRATEGY, width="stretch", hide_index=True)

e1, e2, e3 = st.columns(3)
with e1:
    st.markdown("#### StandardScaler — 9 columns")
    st.markdown("""
Centres each column at 0 with unit variance.

**Needed by:** Linear Regression, Ridge and KNN. Ridge penalises coefficient magnitude, so
unscaled features get penalised in proportion to their units rather than their importance; KNN
measures distance, so an unscaled `milage` (0–400,000) would drown out `cylinders` (0–12)
entirely.

**Harmless for:** the tree ensembles, which split on thresholds and are invariant to monotone
rescaling. It is applied to everything anyway so that all six candidates run through one
identical pipeline and the comparison stays fair.
""")
with e2:
    st.markdown("#### OneHotEncoder — 4 columns")
    st.markdown("""
One binary column per category level.

Applied to `fuel_type`, `transmission_group`, `accident` and `clean_title_flag` — between 2 and 7
levels each, so this adds roughly 16 columns in total. Cheap, lossless, and no ordering is
implied between categories.

Configured with `handle_unknown='ignore'` so a category never seen in training does not crash a
prediction — it simply produces all-zero indicators.
""")
with e3:
    st.markdown("#### TargetEncoder — 4 columns")
    st.markdown("""
Replaces each category with a smoothed average of the target for that category.

Applied to `brand`, `model`, `ext_col` and `int_col`. One-hot encoding these would add roughly
**2,430 columns** to a 3,206-row training set — far wider than it is tall, and almost entirely
zeros.

Target encoding keeps one column each, and scikit-learn's implementation applies internal
cross-fitting plus shrinkage toward the global mean, so rare categories are pulled toward the
average instead of memorising a single row.
""")

st.divider()
st.markdown("## The leakage problem, and how the pipeline solves it")

st.error("""
**`TargetEncoder` uses the target.** That makes it the single most dangerous component in this
project. Fitting it once on the full dataset — the obvious, convenient thing to do — would encode
every `brand` and `model` using price information from rows that later appear in the test set.
Every score after that point would be inflated, and the model would look far better than it is.
""")

st.markdown("""
The protection is structural rather than procedural. The dataset is split into training and test
portions **before** any encoder or scaler is ever fitted, so the test rows are set aside first
and touched by nothing that learns from the data. Every scaling and encoding step then lives
inside one combined pipeline alongside the model itself, wrapped in the log-price transform. When
that pipeline is evaluated with cross-validation, the entire thing — encoders included — is
refit from scratch on each fold's training portion alone, so no fold ever leaks information about
the rows it will be scored against.
""")

st.success(f"""
The {utils.TEST_ROWS} test rows were separated before a single encoder was fitted and were not
examined again until final evaluation. Inside cross-validation, the `TargetEncoder` is refit on
each fold's training portion only. Nothing that learns from prices ever sees a price it will
later be scored against.
""")

sp1, sp2 = st.columns(2)
sp1.metric("Training rows", f"{utils.TRAIN_ROWS:,}", delta="80%")
sp2.metric("Test rows", f"{utils.TEST_ROWS:,}", delta="20% — untouched until the end")

st.divider()
st.markdown("## The transformed feature matrix")

st.markdown(
    f"This is what the model actually receives — {len(utils.MODEL_FEATURES)} columns, in this "
    "exact order:"
)
st.dataframe(df[utils.MODEL_FEATURES].head(8), width="stretch")

with st.expander("Feature dtypes as handed to the pipeline"):
    st.dataframe(
        pd.DataFrame({
            "Feature": utils.MODEL_FEATURES,
            "Dtype": [str(df[c].dtype) for c in utils.MODEL_FEATURES],
            "Distinct values": [df[c].nunique() for c in utils.MODEL_FEATURES],
        }),
        width="stretch", hide_index=True, height=460,
    )

st.success("Next: **Model Development** in the sidebar.")
