import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
utils.project_sidebar()

utils.page_header(
    "🔍 Exploratory Data Analysis",
    "What the target looks like, and how every modelling feature relates to it.",
    section="Sections 6 & 8",
)

try:
    raw, _ = utils.load_raw_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = utils.prepare_data(raw)

st.markdown("## The target: `price`")

desc = df["price"].describe()
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Median", utils.format_money(desc["50%"]))
d2.metric("Mean", utils.format_money(desc["mean"]))
d3.metric("Min", utils.format_money(desc["min"]))
d4.metric("Max", utils.format_money(desc["max"]))
d5.metric("Std dev", utils.format_money(desc["std"]))

skew_raw = df["price"].skew()
skew_log = np.log1p(df["price"]).skew()

st.warning(f"""
**The mean sits {utils.format_money(desc['mean'] - desc['50%'])} above the median.** That gap is
the whole story of this target: a small number of very expensive cars drag the average away from
anything typical. Skewness confirms it — **{skew_raw:.2f}** for raw price, where 0 would be
symmetric.
""")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["price"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[0].set_title(f"Raw price (skew = {skew_raw:.2f})")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Listings")
axes[1].hist(np.log1p(df["price"]), bins=60, color="#55A868", edgecolor="white", linewidth=0.3)
axes[1].set_title(f"log1p(price) (skew = {skew_log:.2f})")
axes[1].set_xlabel("log1p(price)")
axes[1].set_ylabel("Listings")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.success(f"""
**This single chart justifies the core modelling decision.** Raw price is unusable as a
regression target — almost every listing is crushed into the left-hand bar while the axis
stretches to accommodate a handful of supercars. A log transform brings skewness from
{skew_raw:.2f} down to {skew_log:.2f}, close to a normal distribution. The model therefore
trains on the logged target and inverts the transform at prediction time, which means it is
effectively minimising *percentage* error rather than *dollar* error — treating a $2,000 miss on
a $10,000 car as comparably bad to a $20,000 miss on a $100,000 car.
""")

with st.expander("Where the listings actually sit — price percentiles"):
    pct = df["price"].quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("Price")
    pct.index = [f"{int(i * 100)}th percentile" for i in pct.index]
    pct["Price"] = pct["Price"].apply(utils.format_money)
    st.dataframe(pct, width="stretch")
    st.caption(
        "90% of this dataset is ordinary used cars. The long tail above the 99th percentile is "
        "where the model's accuracy eventually breaks down."
    )

st.divider()
st.markdown("## Correlations among the numeric features")

num_cols = ["price", "horsepower", "engine_liters", "cylinders", "milage", "car_age", "model_year"]
corr = df[num_cols].corr()

cc1, cc2 = st.columns([1.1, 1])
with cc1:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with cc2:
    price_corr = corr["price"].drop("price").sort_values(ascending=False)
    st.markdown("**Correlation with price**")
    st.dataframe(price_corr.round(3).rename("r").to_frame(), width="stretch")
    st.markdown(f"""
- **`horsepower` ({price_corr['horsepower']:+.2f})** is the strongest positive linear correlate.
- **`milage` ({price_corr['milage']:+.2f})** and
  **`car_age` ({price_corr['car_age']:+.2f})** are both negative, as expected.
- **`milage` and `car_age` correlate with each other at
  {df['milage'].corr(df['car_age']):.2f}** — older cars simply have more miles. Tree ensembles
  handle that redundancy gracefully; linear models split the credit between them, which is part
  of why the linear baselines underperform.
""")

st.info("""
**These are Pearson correlations, which only measure *linear* association.** The relationships
below are closer to exponential decay than straight lines, so the true predictive strength of
these features is higher than these numbers suggest — which is exactly why tree-based models beat
the linear ones in cross-validation.
""")

st.divider()
st.markdown("## Price against every numeric feature used in modelling")

sample = df.sample(min(1500, len(df)), random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
pairs = [
    ("milage", "Mileage", "#4C72B0"),
    ("car_age", "Car age (years)", "#C44E52"),
    ("horsepower", "Horsepower", "#55A868"),
    ("engine_liters", "Engine size (litres)", "#8172B2"),
]
for ax, (col, label, color) in zip(axes.flat, pairs):
    ax.scatter(sample[col], sample["price"], alpha=0.35, s=12, color=color)
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("Price ($, log scale)")
    ax.set_title(f"Price vs {label.lower()}")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("""
Note the **log y-axis** on every panel. On this scale the mileage and age relationships look
roughly linear, which means on a normal scale they are exponential decay — value falls fastest
early on and then flattens. Horsepower and engine size trend upward, as expected, but with far
more scatter: a large engine does not guarantee a high price the way low mileage reliably rules
out a very low one.
""")

fig, ax = plt.subplots(figsize=(9, 3.6))
cyl_stats = df.groupby("cylinders")["price"].median()
ax.bar(cyl_stats.index.astype(str), cyl_stats.values, color="#DD8452", width=0.6)
ax.set_xlabel("Cylinders")
ax.set_ylabel("Median price ($)")
ax.set_title("Median price by cylinder count")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("""
Cylinder count behaves more like a category than a smooth number — 0 cylinders (electric
vehicles) command a premium of their own, and price generally climbs with cylinder count up
through 8, then becomes noisy at the rare 10- and 12-cylinder end of the market where sample
sizes are tiny.
""")

st.divider()
st.markdown("## Price by brand and model")

top_brands = df["brand"].value_counts().head(12).index
brand_stats = (
    df[df["brand"].isin(top_brands)]
    .groupby("brand")["price"]
    .agg(["median", "count"])
    .sort_values("median", ascending=False)
)

bc1, bc2 = st.columns([1.4, 1])
with bc1:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(brand_stats.index[::-1], brand_stats["median"][::-1], color="#4C72B0")
    ax.set_xlabel("Median price ($)")
    ax.set_title("Median price — 12 most-listed brands")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with bc2:
    show = brand_stats.copy()
    show["median"] = show["median"].apply(utils.format_money)
    show.columns = ["Median price", "Listings"]
    st.dataframe(show, width="stretch")

ratio = brand_stats["median"].max() / brand_stats["median"].min()
st.success(f"""
**Brand matters enormously.** Median prices across just the 12 most-listed brands span a factor
of **{ratio:.1f}×** — and that is before including the luxury marques with only a handful of
listings each. This is the justification for target-encoding `brand` and `model` rather than
dropping them for being high-cardinality.
""")

top_models = df["model"].value_counts().head(10).index
model_stats = (
    df[df["model"].isin(top_models)]
    .groupby("model")["price"]
    .agg(["median", "count"])
    .sort_values("median", ascending=False)
)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.barh(model_stats.index[::-1], model_stats["median"][::-1], color="#64B5CD")
ax.set_xlabel("Median price ($)")
ax.set_title("Median price — 10 most-listed models")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Even among only the most common models, medians vary widely — confirming that `model` "
    "carries information beyond what `brand` alone provides."
)

st.divider()
st.markdown("## Price by condition, fuel type and drivetrain")

f1, f2 = st.columns(2)
with f1:
    acc = df.groupby("accident")["price"].agg(["median", "count"])
    acc_display = acc.copy()
    acc_display["median"] = acc_display["median"].apply(utils.format_money)
    acc_display.columns = ["Median price", "Listings"]
    st.markdown("**Accident history**")
    st.dataframe(acc_display, width="stretch")
    st.caption(
        "Cars with reported damage list lower — but the gap is smaller than intuition suggests, "
        "because damage is more common on older, already-cheaper cars."
    )
with f2:
    fuel = df.groupby("fuel_type")["price"].agg(["median", "count"]).sort_values(
        "median", ascending=False)
    fuel_display = fuel.copy()
    fuel_display["median"] = fuel_display["median"].apply(utils.format_money)
    fuel_display.columns = ["Median price", "Listings"]
    st.markdown("**Fuel type**")
    st.dataframe(fuel_display, width="stretch")
    st.caption(
        "Watch the listing counts as well as the medians — some categories rest on very few rows."
    )

f3, f4 = st.columns(2)
with f3:
    trans = df.groupby("transmission_group")["price"].agg(["median", "count"]).sort_values(
        "median", ascending=False)
    trans_display = trans.copy()
    trans_display["median"] = trans_display["median"].apply(utils.format_money)
    trans_display.columns = ["Median price", "Listings"]
    st.markdown("**Transmission group**")
    st.dataframe(trans_display, width="stretch")
    st.caption(
        "Manuals list lower on average, largely because they cluster among older and sportier "
        "niche listings rather than because a manual gearbox itself commands a discount."
    )
with f4:
    elec = df.groupby("is_electric")["price"].agg(["median", "count"])
    elec.index = elec.index.map({0: "Combustion / hybrid", 1: "Electric"})
    elec_display = elec.copy()
    elec_display["median"] = elec_display["median"].apply(utils.format_money)
    elec_display.columns = ["Median price", "Listings"]
    st.markdown("**Electric vs. combustion**")
    st.dataframe(elec_display, width="stretch")
    st.caption(
        "Electric vehicles list noticeably higher on median, reflecting both their relative "
        "newness and a market still weighted toward premium EV models."
    )

st.divider()
st.markdown("## Price by colour and title status")

cc3, cc4 = st.columns(2)
with cc3:
    top_ext = df["ext_col"].value_counts().head(8).index
    ext_stats = df[df["ext_col"].isin(top_ext)].groupby("ext_col")["price"].median().sort_values(
        ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(ext_stats.index[::-1], ext_stats.values[::-1], color="#937860")
    ax.set_xlabel("Median price ($)")
    ax.set_title("Median price — 8 most common exterior colours")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with cc4:
    top_int = df["int_col"].value_counts().head(8).index
    int_stats = df[df["int_col"].isin(top_int)].groupby("int_col")["price"].median().sort_values(
        ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(int_stats.index[::-1], int_stats.values[::-1], color="#8C8C8C")
    ax.set_xlabel("Median price ($)")
    ax.set_title("Median price — 8 most common interior colours")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.caption(
    "Colour differences are modest next to brand or mileage, but they are not flat — certain "
    "shades skew toward higher-end trims and inherit some of that price signal."
)

ct = df.groupby("clean_title_flag")["price"].agg(["median", "count"]).sort_values(
    "median", ascending=False)
ct_display = ct.copy()
ct_display["median"] = ct_display["median"].apply(utils.format_money)
ct_display.columns = ["Median price", "Listings"]
st.markdown("**Clean title status**")
st.dataframe(ct_display, width="stretch")
st.caption(
    "Confirmed clean-title listings command a real premium over the unconfirmed/other group, "
    "consistent with a buyer paying for documented certainty."
)

st.divider()
st.markdown("## Does missingness itself relate to price?")

miss_cols = [("hp_missing", "Horsepower"), ("liters_missing", "Engine size"),
             ("cyl_missing", "Cylinders")]
miss_rows = []
for col, label in miss_cols:
    grp = df.groupby(col)["price"].median()
    miss_rows.append({
        "Feature": label,
        "Median price — stated": utils.format_money(grp.get(0, float("nan"))),
        "Median price — missing/imputed": utils.format_money(grp.get(1, float("nan"))),
    })
st.dataframe(pd.DataFrame(miss_rows), width="stretch", hide_index=True)

st.markdown("""
Listings that omit engine specifications tend to price lower than listings that state them —
consistent with the idea that sparser, less detailed listings skew toward older or more casually
sold cars. This is exactly why those missingness flags were kept as their own features rather
than being discarded once the gaps were filled in.
""")

st.divider()
st.markdown("## Takeaways carried into modelling")
st.markdown(f"""
1. **The target must be log-transformed.** Skew {skew_raw:.2f} → {skew_log:.2f}.
2. **Relationships are non-linear.** Mileage and age decay exponentially, so tree ensembles that
   can split on thresholds should beat linear models — and in cross-validation they do.
3. **Brand and model carry real signal** worth the cost of target encoding.
4. **`milage` and `car_age` are redundant with each other** (r =
   {df['milage'].corr(df['car_age']):.2f}). This explains a result that looks odd later: `car_age`
   ranks near the bottom on permutation importance despite correlating with price, because once
   the model has mileage, shuffling age removes little *additional* information.
5. **Missingness carries a real price signal**, justifying the missingness flags as features.
6. **The expensive tail is thin and irregular.** Fewer than 1% of listings sit above
   {utils.format_money(df['price'].quantile(0.99))}, and those rows will dominate any squared-error
   metric.
""")

st.success("Next: **Feature Engineering** in the sidebar.")
