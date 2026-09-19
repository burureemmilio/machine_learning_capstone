import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Model Development", page_icon="⚙️", layout="wide")
utils.project_sidebar()

utils.page_header(
    "⚙️ Model Development",
    "Six candidates, one shared evaluation scheme, and the tuning that followed.",
    section="Sections 12, 14–16",
)

st.markdown("## The candidates")

st.markdown("""
Six algorithms, chosen to span **different families** rather than several variations on one idea.
A comparison between three flavours of linear model would tell you very little about whether the
problem is linear at all.
""")

candidates = pd.DataFrame([
    ("Linear Regression", "Linear", "The floor. Establishes what a straight-line relationship "
     "can achieve, so every other result has something to be measured against."),
    ("Ridge Regression", "Linear + L2", "Tests whether the plain linear model is overfitting. "
     "If Ridge ≈ Linear, the problem is underfitting, not variance."),
    ("K-Nearest Neighbors", "Instance-based", "A non-parametric, non-linear baseline that makes "
     "no assumption about functional form — it just finds similar cars."),
    ("Random Forest", "Bagged trees", "Averages many decorrelated deep trees. Captures "
     "interactions without being told they exist."),
    ("HistGradientBoosting", "Boosted trees", "scikit-learn's histogram-based booster, same "
     "family as LightGBM. Builds trees sequentially, each correcting the last."),
    ("XGBoost", "Boosted trees", "A separately-implemented booster, included to check that any "
     "boosting advantage is real and not an artefact of one library."),
], columns=["Model", "Family", "Why it is in the comparison"])

st.dataframe(candidates, width="stretch", hide_index=True)

st.info("""
**Every candidate runs through the identical pipeline** — same scaling, same encoding, same
log-target wrapper, same folds. The only thing that varies is the final estimator. Without that,
a difference in scores could just as easily reflect a difference in preprocessing.
""")

st.divider()
st.markdown("## The evaluation scheme")

st.markdown(f"""
**5-fold cross-validation on the training set only.** The {utils.TRAIN_ROWS:,} training rows are
split five different ways, and each model is fitted five times — each time on four-fifths of the
training rows, then scored on the remaining fifth it did not see during that fit. The reported
score for each metric is the average across all five fits, with the spread across folds showing
how stable that average really is.

The rows are shuffled before splitting, since the source file is not stored in a random order,
and the same shuffled split is reused for every candidate model. That means all six algorithms
are judged on exactly the same five partitions of data — the comparison between them is paired
rather than independent, which makes small differences between models meaningful rather than
noise from different splits.

The {utils.TEST_ROWS} held-out test rows take no part in any of this — they remain untouched
until the very end, on the Final Model page.
""")

st.markdown("**Three metrics, because they disagree in useful ways:** R² for variance explained, "
            "MAE for the typical dollar miss, RMSE for sensitivity to large misses. A model that "
            "wins on MAE but loses on RMSE is making fewer small errors and more big ones.")

st.divider()
st.markdown("## Shortlisting for tuning")

st.markdown("""
Rather than assuming a winner, the two models with the **lowest cross-validated RMSE** were
carried forward. On this run that was **Random Forest** and **HistGradientBoosting**, with
XGBoost a close third.
""")

st.markdown(
    "The two lowest cross-validated RMSE scores were simply read off the results table and "
    "carried forward — **Random Forest** and **HistGradientBoosting** — rather than being "
    "assumed in advance."
)

st.success("""
**All three tree ensembles beat both linear models and KNN** by a wide margin. That is a finding,
not just a ranking: it says the price/feature relationship has meaningful non-linearities and
interactions. The effect of mileage on price plausibly depends on the car's age and brand — 80,000
miles means something very different on a 3-year-old Bentley than on a 15-year-old Honda. A tree
can split on that; a single linear coefficient cannot represent it.
""")

st.divider()
st.markdown("## Hyperparameter tuning")

st.markdown("""
Both shortlisted models were tuned with `RandomizedSearchCV` over the **same 5 folds**, optimising
RMSE directly.
""")

st.markdown(f"""
**Why randomized rather than grid search?** The HistGradientBoosting grid alone has
5 × 4 × 4 × 4 × 4 × 5 = **6,400** combinations. At 5-fold CV that would be 32,000 model fits.
Forty random draws covers the space well enough to find a strong combination at 200 fits —
and in high-dimensional spaces random sampling explores each individual parameter's range more
thoroughly than a grid of the same size, because a grid wastes draws repeating values that don't
matter.
""")

t1, t2 = st.columns(2)

with t1:
    st.markdown("### HistGradientBoosting")
    st.markdown("""
| Parameter | What it controls |
|---|---|
| `learning_rate` | Step size. Lower generalises better but needs more iterations. |
| `max_iter` | Number of boosting rounds (trees). |
| `max_leaf_nodes` | Tree complexity — the main capacity control for this model. |
| `min_samples_leaf` | Regularisation: higher forces simpler trees. |
| `l2_regularization` | Explicit L2 penalty on leaf values. |
| `max_depth` | An additional depth cap on top of leaf-node count. |
""")
    st.markdown("**Best combination found**")
    st.dataframe(
        pd.DataFrame(
            [(k, str(v)) for k, v in utils.BEST_PARAMS_HGB.items()],
            columns=["Parameter", "Value"],
        ),
        width="stretch", hide_index=True,
    )
    st.metric("Best CV RMSE", "$22,144", delta="-$1,720 vs default", delta_color="inverse")

with t2:
    st.markdown("### Random Forest")
    st.markdown("""
| Parameter | What it controls |
|---|---|
| `n_estimators` | Number of trees. More reduces variance, costs compute. |
| `max_depth` | Maximum depth per tree — main capacity control. |
| `min_samples_split` | Minimum samples to split a node. |
| `min_samples_leaf` | Minimum samples at a leaf; higher regularises. |
| `max_features` | How many features each split considers — controls decorrelation. |
| `max_samples` | Bootstrap fraction per tree. |
""")
    st.markdown("**Best combination found**")
    st.dataframe(
        pd.DataFrame(
            [(k, str(v)) for k, v in utils.BEST_PARAMS_RF.items()],
            columns=["Parameter", "Value"],
        ),
        width="stretch", hide_index=True,
    )
    st.metric("Best CV RMSE", "$23,672", delta="+$59 vs default", delta_color="inverse")

st.warning("""
**Tuning helped one model and not the other.** HistGradientBoosting improved by about $1,720 in
cross-validated RMSE. Random Forest came out marginally *worse* than its own defaults — well
within fold-to-fold noise, which means scikit-learn's Random Forest defaults were already close
to optimal for this dataset, and the randomized search simply didn't find anything better in 40
draws.

That is a legitimate result worth reporting rather than hiding. Tuning is not guaranteed to pay,
and the honest version of this comparison shows both outcomes.
""")

st.markdown("""
Note also the **cost asymmetry** in the cross-validation timings: Random Forest took roughly 23
seconds per CV run against HistGradientBoosting's 1.3 seconds — about 18× the compute for a
slightly worse tuned score.
""")

st.success("Next: **Model Comparison** in the sidebar.")
