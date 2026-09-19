import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
utils.project_sidebar()

utils.page_header(
    "📈 Model Comparison",
    "Cross-validated results for all six candidates, then the tuned finalists.",
    section="Sections 13 & 17",
)

cv = utils.CV_RESULTS.copy()

st.markdown("## How the models are being judged")

m1, m2 = st.columns(2)
with m1:
    st.markdown("""
**Metrics used**

| Metric | What it tells you |
|---|---|
| **R²** | Share of price variance explained. Sensitive to a few large errors. |
| **MAE** | Average miss in dollars. Treats every error equally. |
| **RMSE** | Penalises large misses much more heavily than MAE. |
| **MAPE** | Average miss as a percentage — comparable across price ranges. |
| **Median AE** | The typical miss, unaffected by a handful of extreme rows. |
""")
with m2:
    st.info("""
**Why more than one metric?**

RMSE and R² both square the errors, so a single very large miss on one exotic car counts as much
as hundreds of small misses on ordinary cars. MAE, MAPE and median absolute error are far more
robust to that.

The tables and charts below report several of these side by side precisely because they can
disagree — and on the Final Model page, that disagreement turns out to be the most interesting
finding in the whole analysis.
""")

st.divider()
st.markdown("## 5-fold cross-validation — all six candidates")
st.caption(f"Training set only ({utils.TRAIN_ROWS:,} rows). The test set takes no part in this.")

display = cv.copy()
display["R² (mean)"] = display["R² (mean)"].map("{:.3f}".format)
display["R² (std)"] = display["R² (std)"].map("± {:.3f}".format)
display["MAE (mean $)"] = display["MAE (mean $)"].map("${:,.0f}".format)
display["RMSE (mean $)"] = display["RMSE (mean $)"].map("${:,.0f}".format)
display["CV time (s)"] = display["CV time (s)"].map("{:.1f}s".format)

st.dataframe(display, width="stretch", hide_index=True)

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ordered = cv.sort_values("RMSE (mean $)", ascending=False)
    colors = ["#55A868" if m in ("Random Forest", "HistGradientBoosting") else "#4C72B0"
              for m in ordered["Model"]]
    ax.barh(ordered["Model"], ordered["RMSE (mean $)"], color=colors)
    ax.set_xlabel("Cross-validated RMSE ($) — lower is better")
    ax.set_title("RMSE by model")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with g2:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ordered = cv.sort_values("R² (mean)")
    colors = ["#55A868" if m in ("Random Forest", "HistGradientBoosting") else "#4C72B0"
              for m in ordered["Model"]]
    ax.barh(ordered["Model"], ordered["R² (mean)"],
            xerr=ordered["R² (std)"], color=colors, capsize=3)
    ax.set_xlabel("Cross-validated R² — higher is better")
    ax.set_title("R² by model (error bars = std across folds)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.markdown("## Reading the results")

best_rmse = cv["RMSE (mean $)"].min()
worst_rmse = cv["RMSE (mean $)"].max()
gap = cv[cv["Model"] == "Linear Regression"]["RMSE (mean $)"].iloc[0] - best_rmse

r1, r2, r3 = st.columns(3)
r1.metric("Best (Random Forest)", f"${best_rmse:,.0f}", "RMSE")
r2.metric("Worst (KNN)", f"${worst_rmse:,.0f}", "RMSE")
r3.metric("Trees vs linear", f"${gap:,.0f}", "RMSE improvement", delta_color="off")

st.markdown(f"""
**1. Tree ensembles win decisively.** All three sit at R² ≈ 0.77–0.78, while the linear models
reach 0.67 and KNN only 0.55. The gap between the best tree and the best linear model is about
**${gap:,.0f} of RMSE** — roughly an 18% reduction in error. This confirms what the EDA suggested:
price responds to these features non-linearly, and depends on interactions between them.

**2. Linear and Ridge are almost identical** (R² 0.6739 vs 0.6731; the RMSE difference is $38).
Ridge adds L2 regularisation, so if the plain linear model were overfitting, Ridge would have
pulled ahead. It didn't. The linear models are **underfitting** — they are too simple for this
relationship, not too complex. More regularisation was never going to help.

**3. KNN is the clear loser** at R² 0.546. With 18 features, several of them target-encoded,
distance becomes a weak notion of similarity — the curse of dimensionality in action. Cars that
are close in this space are not reliably close in price.

**4. The three tree ensembles are within noise of each other.** Random Forest leads
HistGradientBoosting by $251 of RMSE, against a fold-to-fold standard deviation of ±0.05 in R².
That is not a meaningful lead. Which is exactly why both were taken forward to tuning rather than
crowning Random Forest immediately.

**5. Cost differs by more than accuracy does.** Random Forest needed 22.8s per CV run against
HistGradientBoosting's 1.3s — **18× the compute** for a statistically indistinguishable score.
""")

st.divider()
st.markdown("## After tuning")

tuned = utils.TUNING_COMPARISON.copy()
tuned_display = tuned.copy()
tuned_display["CV RMSE ($)"] = tuned_display["CV RMSE ($)"].map("${:,.0f}".format)

tc1, tc2 = st.columns([1, 1.2])
with tc1:
    st.dataframe(tuned_display, width="stretch", hide_index=True)
with tc2:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ordered = tuned.sort_values("CV RMSE ($)", ascending=False)
    colors = ["#55A868" if "tuned" in m and "Hist" in m else "#B0B0B0"
              for m in ordered["Model"]]
    ax.barh(ordered["Model"], ordered["CV RMSE ($)"], color=colors)
    ax.set_xlabel("Cross-validated RMSE ($)")
    ax.set_xlim(20000, 24500)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.success(f"""
**Final selection: {utils.FINAL_MODEL_NAME}**, at a cross-validated RMSE of **$22,144**.

Tuning moved HistGradientBoosting from 4th place to 1st — a $1,720 improvement that was enough
to overtake Random Forest. Random Forest's own tuning produced nothing: its tuned score ($23,672)
is fractionally worse than its defaults ($23,613), which is noise. Selection was made on
cross-validated RMSE alone, before the test set was touched.
""")

st.info("""
**Why RMSE was the selection criterion.** RMSE penalises large errors more heavily than MAE does.
For a pricing tool that is the right preference: being $30,000 wrong once is considerably worse
than being $3,000 wrong ten times, because a single large miss is what destroys trust in an
estimate. The consequence is that the model is deliberately cautious about the expensive tail —
and the next page shows exactly how well that worked, and where it didn't.
""")

st.success("Next: **Final Model** in the sidebar.")
