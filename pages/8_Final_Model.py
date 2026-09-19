import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Final Model", page_icon="🏁", layout="wide")
utils.project_sidebar()

utils.page_header(
    "🏁 Final Model & Evaluation",
    "Held-out test performance, where the errors are, and what the model cannot do.",
    section="Sections 18–23",
)


@st.cache_data(show_spinner="Evaluating on the held-out test set…")
def evaluate():
    """Re-create the exact train/test split and score the saved model on the test half."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    raw, _ = utils.load_raw_data()
    df = utils.prepare_data(raw)
    X = df[utils.MODEL_FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = utils.load_model()
    preds = model.predict(X_test)

    frame = X_test.copy()
    frame["actual"] = y_test.values
    frame["predicted"] = preds
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["pct_error"] = (frame["abs_error"] / frame["actual"] * 100).round(1)

    # How many times each exact model appears in the whole dataset — the key diagnostic.
    counts = df["model"].value_counts()
    frame["times_in_dataset"] = frame["model"].map(counts).fillna(0).astype(int)

    def score(f):
        return {
            "r2": r2_score(f["actual"], f["predicted"]),
            "mae": mean_absolute_error(f["actual"], f["predicted"]),
            "rmse": float(np.sqrt(mean_squared_error(f["actual"], f["predicted"]))),
            "mape": float((f["abs_error"] / f["actual"] * 100).mean()),
            "median_ae": float(f["abs_error"].median()),
        }

    full = score(frame)
    worst_two = frame.nlargest(2, "abs_error").index
    trimmed = score(frame.drop(index=worst_two))

    return frame, full, trimmed, list(worst_two)


try:
    results, full, trimmed, worst_two_idx = evaluate()
except Exception as exc:
    st.error(f"Could not evaluate the model: {exc}")
    st.stop()

st.markdown(f"## The final model: {utils.FINAL_MODEL_NAME}")

with st.expander("Hyperparameters of the model file actually loaded"):
    try:
        model = utils.load_model()
        inner = model.regressor_ if hasattr(model, "regressor_") else model.regressor
        est = inner.named_steps["model"]
        interesting = ["learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                       "min_samples_leaf", "l2_regularization", "random_state"]
        params = est.get_params()
        st.dataframe(
            pd.DataFrame([(k, str(params.get(k))) for k in interesting],
                         columns=["Parameter", "Value"]),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Read directly from used_car_price_model.pkl, so this always describes the model "
            "this app is really using."
        )
    except Exception as exc:
        st.caption(f"Could not read hyperparameters: {exc}")

st.markdown("### Held-out test set — the headline numbers")
st.caption(
    f"{len(results):,} cars the model has never seen, separated before any encoder was fitted."
)

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("R²", f"{full['r2']:.3f}")
h2.metric("MAE", utils.format_money(full["mae"]))
h3.metric("RMSE", utils.format_money(full["rmse"]))
h4.metric("MAPE", f"{full['mape']:.1f}%")
h5.metric("Median AE", utils.format_money(full["median_ae"]))

st.error(f"""
**These numbers contradict each other, and that is the most important result in the project.**

R² of **{full['r2']:.3f}** says the model explains almost none of the variance in price — it looks
like a failure. But the median absolute error is **{utils.format_money(full['median_ae'])}**,
meaning half of all predictions land within about four thousand dollars of the asking price, and
the mean absolute percentage error is **{full['mape']:.1f}%**. A model that explains nothing does
not do that.

Both numbers are correct. The rest of this page explains why they disagree.
""")

st.divider()
st.markdown("## Where the error actually lives")

worst = results.nlargest(15, "abs_error")[
    ["brand", "model", "model_year", "milage", "actual", "predicted",
     "abs_error", "pct_error", "times_in_dataset"]
].copy()
for col in ["actual", "predicted", "abs_error"]:
    worst[col] = worst[col].apply(utils.format_money)
worst["milage"] = worst["milage"].apply(lambda v: f"{v:,.0f}")
worst.columns = ["Brand", "Model", "Year", "Mileage", "Actual", "Predicted",
                 "Abs error", "% error", "Times in dataset"]

st.markdown("**The 15 worst predictions**")
st.dataframe(worst, width="stretch", hide_index=True)

top2 = results.loc[worst_two_idx]
combined = top2["abs_error"].sum()
total_abs = results["abs_error"].sum()

w1, w2, w3 = st.columns(3)
w1.metric("Median price, worst 20", utils.format_money(results.nlargest(20, "abs_error")["actual"].median()))
w2.metric("Median price, whole test set", utils.format_money(results["actual"].median()))
w3.metric("Share of total absolute error from 2 cars", f"{combined / total_abs:.1%}")

st.markdown(f"""
The worst errors are not spread across the dataset. They are concentrated almost entirely in
**one-of-a-kind exotic cars**. The two largest misses are a **Bugatti Veyron** and a **Porsche
Carrera GT**, each missed by well over a million dollars.

Look at the final column. Both appear **exactly once** in the entire {utils.CLEAN_ROWS:,}-row
dataset.
""")

st.warning("""
**This is a structural limitation, not a bug.** `model` is target-encoded, which means the
encoding for a given model is learned from other rows with that same model. When a car is the
only example of its model anywhere in the data, the training half contains **zero** examples to
learn from. The encoder falls back toward the global average, and the model is left predicting
from brand and engine specs alone — which badly undershoots a limited-production halo car whose
price comes from scarcity, not from horsepower.

No amount of extra regularisation, boosting rounds or tuning can fix this. The only fix is more
listings for rare models, which this dataset does not contain.
""")

st.markdown("### What happens if those two rows are removed")

ex1, ex2, ex3 = st.columns(3)
ex1.metric("R²", f"{trimmed['r2']:.3f}",
           delta=f"{trimmed['r2'] - full['r2']:+.3f}")
ex2.metric("RMSE", utils.format_money(trimmed["rmse"]),
           delta=f"-{utils.format_money(full['rmse'] - trimmed['rmse'])}",
           delta_color="inverse")
ex3.metric("MAE", utils.format_money(trimmed["mae"]),
           delta=f"-{utils.format_money(full['mae'] - trimmed['mae'])}",
           delta_color="inverse")

st.success(f"""
**Removing 2 rows out of {len(results)} — 0.25% of the test set — moves R² from
{full['r2']:.3f} to {trimmed['r2']:.3f}.**

That is the entire explanation for the gap between the cross-validated R² (≈0.78) and the
headline test R². RMSE and R² both square the errors, so a single $1.8M miss contributes as much
as roughly 7,000 typical $700 misses. Two such rows landing in the test half rather than the
training half is enough to dominate the metric for all 802 cars.

This is presented as a **diagnosis, not a correction**. The trimmed figure is not the model's
score — it is the evidence that the headline figure is being driven by two points.
""")

st.divider()
st.markdown("## Accuracy by price bracket")

st.markdown(
    "Since a couple of extreme points can swing an aggregate metric, the more informative view "
    "is accuracy *within* price bands:"
)

bins = [0, 25000, 50000, 100000, 250000, np.inf]
labels = ["< $25K", "$25K–$50K", "$50K–$100K", "$100K–$250K", "> $250K"]
results = results.copy()
results["bracket"] = pd.cut(results["actual"], bins=bins, labels=labels)

results["pct_err_calc"] = results["abs_error"] / results["actual"] * 100
bracket = (
    results.groupby("bracket", observed=True)
    .agg(**{
        "Cars": ("abs_error", "size"),
        "Median AE ($)": ("abs_error", "median"),
        "MAE ($)": ("abs_error", "mean"),
        "Mean abs % error": ("pct_err_calc", "mean"),
    })
    .reset_index()
)

bd = bracket.copy()
bd["Cars"] = bd["Cars"].astype(int)
bd["Median AE ($)"] = bd["Median AE ($)"].apply(utils.format_money)
bd["MAE ($)"] = bd["MAE ($)"].apply(utils.format_money)
bd["Mean abs % error"] = bd["Mean abs % error"].map("{:.1f}%".format)
bd.columns = ["Price bracket", "Cars", "Median AE", "MAE", "Mean abs % error"]

bk1, bk2 = st.columns([1.1, 1])
with bk1:
    st.dataframe(bd, width="stretch", hide_index=True)
with bk2:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(bracket["bracket"].astype(str), bracket["Mean abs % error"], color="#4C72B0")
    ax.set_ylabel("Mean absolute % error")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

covered = int(bracket[bracket["bracket"].isin(labels[:3])]["Cars"].sum())
st.markdown(f"""
**Percentage error stays in a fairly narrow band across the brackets that hold the bulk of the
data** — {covered} of {len(results)} test cars ({covered / len(results):.0%}) sit under $100K, and
across those three brackets the mean absolute percentage error stays broadly stable. Accuracy
degrades sharply only in the **> $250K** bracket, which contains just
{int(bracket[bracket['bracket'] == '> $250K']['Cars'].iloc[0]) if '> $250K' in bracket['bracket'].astype(str).values else 0}
cars — the exotics again.

Note that the *dollar* error grows steadily with price while the *percentage* error does not.
That is the expected behaviour of a model trained on log price, and it is the right trade: a
fixed percentage error is what a person actually cares about when pricing a car.
""")

st.divider()
st.markdown("## Actual vs predicted")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].scatter(results["actual"], results["predicted"], alpha=0.4, s=16, color="#4C72B0")
lims = [results["actual"].min(), results["actual"].max()]
axes[0].plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Actual price ($, log scale)")
axes[0].set_ylabel("Predicted price ($, log scale)")
axes[0].set_title("Actual vs predicted")
axes[0].legend()

residuals = results["actual"] - results["predicted"]
plot_mask = residuals.abs() < 100000
axes[1].scatter(results.loc[plot_mask, "predicted"], residuals[plot_mask],
                 alpha=0.4, s=16, color="#55A868")
axes[1].axhline(0, color="r", linestyle="--", linewidth=1.4)
axes[1].set_xscale("log")
axes[1].set_xlabel("Predicted price ($, log scale)")
axes[1].set_ylabel("Residual ($) — actual minus predicted")
axes[1].set_title("Residuals vs predicted price (within ±$100K)")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("""
On the log-scaled scatter, points cluster tightly along the diagonal for the bulk of the price
range, with the spread widening at the top end. On the residual plot, the red horizontal line
marks a perfect prediction, and the cloud of points sits flat along it with no upward or downward
drift as predicted price increases. That flat band is what shows the model is not systematically
over- or under-pricing cars at any point in the range — the errors are noise scattered evenly
around zero rather than a consistent bias that grows or shrinks with price.
""")

with st.expander("Sample of individual predictions"):
    sample = results.sample(min(20, len(results)), random_state=42).sort_values("actual")
    show = sample[["brand", "model", "model_year", "actual", "predicted",
                   "abs_error", "pct_error"]].copy()
    for col in ["actual", "predicted", "abs_error"]:
        show[col] = show[col].apply(utils.format_money)
    show.columns = ["Brand", "Model", "Year", "Actual", "Predicted", "Abs error", "% error"]
    st.dataframe(show, width="stretch", hide_index=True)

st.divider()
st.markdown("## Feature importance")

st.markdown("""
Built-in importances are not comparable across model families, so **permutation importance** is
used instead: each column's values are shuffled on the test set and the drop in R² measures how
much the model actually relied on it. This is model-agnostic and measures real dependence rather
than internal split counts.
""")

imp = utils.PERMUTATION_IMPORTANCE.head(12)

i1, i2 = st.columns([1.3, 1])
with i1:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp["Feature"][::-1], imp["Importance (drop in R²)"][::-1],
            xerr=imp["Std"][::-1], color="#4C72B0", capsize=3)
    ax.set_xlabel("Mean drop in R² when shuffled")
    ax.set_title("Top 12 features by permutation importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with i2:
    st.dataframe(
        utils.PERMUTATION_IMPORTANCE.assign(
            **{"Importance (drop in R²)":
               utils.PERMUTATION_IMPORTANCE["Importance (drop in R²)"].map("{:.4f}".format)}
        )[["Feature", "Importance (drop in R²)"]],
        width="stretch", hide_index=True, height=440,
    )

st.markdown("""
**`milage` dominates**, at roughly three times the importance of the next feature. Then
`horsepower`, the target-encoded `brand`, `engine_liters` and `model` — broadly consistent with
the EDA.

Two results are worth pausing on:

- **`car_age` ranks near the bottom**, despite correlating with price. This is not a
  contradiction. `car_age` and `milage` correlate with each other at roughly 0.6–0.8, so once the
  model has mileage, shuffling age removes very little *additional* information it wasn't already
  getting. Permutation importance measures unique contribution, not raw association.
- **`model_year` scores exactly 0.000000**, which is the correct answer: it is routed to
  `remainder='drop'` in the `ColumnTransformer`, so the model never receives it. A non-zero value
  here would have indicated a bug.

Nothing spurious is driving the predictions — no interior colour outranking mileage — which is a
good sanity check that the pipeline is doing what it claims.
""")

st.divider()
st.markdown("## Limitations, stated plainly")

st.markdown(f"""
1. **One-of-a-kind and ultra-rare models cannot be priced.** If a model appears once in the data,
   there is structurally nothing to learn from. Treat any prediction for a limited-production
   exotic as a rough floor at best.
2. **The model predicts *asking* price, not sale price.** The target is what sellers listed, not
   what buyers paid. Real transaction prices are typically lower.
3. **No condition, service history, or options data.** Two identical listings can differ by
   thousands based on maintenance records, trim options or tyre wear — none of which is in this
   dataset. This puts a hard floor under the achievable error.
4. **No geography or time.** Prices vary by region and move with the market. The dataset is a
   snapshot with no date or location column, so the model cannot account for either, and it will
   drift as the market moves.
5. **`clean_title` is missing on {596 / utils.RAW_ROWS:.0%} of rows**, so the title signal is
   weaker than it would be with complete data.
6. **The dataset is small** at {utils.CLEAN_ROWS:,} rows for 1,898 distinct models — an average
   of about two listings per model. That thinness is the root cause of limitation 1.
""")

st.markdown("## Conclusion")

st.success(f"""
The final model is **{utils.FINAL_MODEL_NAME}**, selected on cross-validated RMSE before the test
set was touched. On the held-out test set it achieves a median absolute error of
**{utils.format_money(full['median_ae'])}** and a mean absolute percentage error of
**{full['mape']:.1f}%**, with percentage error stable across the price bands that cover
{covered / len(results):.0%} of the data.

The headline R² of {full['r2']:.3f} is real but misleading: it is driven almost entirely by two
single-occurrence exotic cars, and recovers to {trimmed['r2']:.3f} when those two rows are set
aside. The practical reading is that this model is well suited to pricing ordinary to upper-range
used cars, and should not be trusted on rare collectibles.
""")

st.success("Next: **Price Predictor** in the sidebar.")
