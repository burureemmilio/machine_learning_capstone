import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide")
utils.project_sidebar()

utils.page_header(
    "🧹 Data Cleaning",
    "Turning text into numbers, handling every gap deliberately, and removing exactly one row.",
    section="Sections 5 & 7",
)

try:
    raw, _ = utils.load_raw_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with_outlier = utils.prepare_data(raw, drop_outlier=False)
clean = utils.prepare_data(raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows in", f"{len(raw):,}")
c2.metric("Rows out", f"{len(clean):,}", delta=f"-{len(raw) - len(clean)}")
c3.metric("Columns in", raw.shape[1])
c4.metric("Columns out", clean.shape[1], delta=f"+{clean.shape[1] - raw.shape[1]}")

st.divider()
st.markdown("## 1. Parsing text into numbers")

p1, p2 = st.columns(2)
with p1:
    st.markdown(
        "**`price`** arrives with a dollar sign and thousands separators. Those characters are "
        "stripped away and the remainder is converted to a plain number, so `\"$10,300\"` becomes "
        "`10300.0` and can finally be plotted, averaged and modelled."
    )
    st.dataframe(
        pd.DataFrame({"Before": raw["price"].head(5).values,
                      "After": with_outlier["price"].head(5).values}),
        width="stretch", hide_index=True,
    )
with p2:
    st.markdown(
        "**`milage`** carries the unit suffix `\" mi.\"` alongside the same comma formatting. "
        "Both are removed the same way, turning `\"51,000 mi.\"` into `51000.0`."
    )
    st.dataframe(
        pd.DataFrame({"Before": raw["milage"].head(5).values,
                      "After": with_outlier["milage"].head(5).values}),
        width="stretch", hide_index=True,
    )

st.divider()
st.markdown("## 2. Pulling three numbers out of the `engine` text")

st.markdown("""
The `engine` column is the richest field in the dataset and the least usable as it stands. A
typical entry reads something like *"300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability"* — a
single sentence that actually contains three separate, useful measurements. Pattern matching is
used to pull each one out on its own: the horsepower number immediately before "HP", the litre
figure before "L" or "Liter", and the cylinder count from whichever of a few common phrasings the
listing happens to use — "6 Cylinder", "V6", "I4", and so on.
""")

demo = with_outlier[["engine", "horsepower", "engine_liters", "cylinders", "is_electric"]].head(8)
st.dataframe(demo, width="stretch", hide_index=True)

hp_found = int(with_outlier["hp_missing"].eq(0).sum())
lt_found = int(with_outlier["liters_missing"].eq(0).sum())
cy_found = int(with_outlier["cyl_missing"].eq(0).sum())

e1, e2, e3 = st.columns(3)
e1.metric("Horsepower parsed", f"{hp_found:,}", delta=f"{hp_found / len(raw):.0%} of rows")
e2.metric("Displacement parsed", f"{lt_found:,}", delta=f"{lt_found / len(raw):.0%} of rows")
e3.metric("Cylinders parsed", f"{cy_found:,}", delta=f"{cy_found / len(raw):.0%} of rows")

st.info(
    "**Electric vehicles are handled separately and deliberately.** An electric motor has "
    "genuinely zero cylinders and zero litres of displacement — that is a real value, not a "
    "missing one. So rows flagged as electric get `0`, not the median. Filling them with the "
    f"fleet median instead would have told the model that "
    f"{int(with_outlier['is_electric'].sum())} EVs have a "
    f"{with_outlier['cylinders'].median():.0f}-cylinder engine."
)

st.divider()
st.markdown("## 3. Missing values — what's actually there")

missing = raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing": missing.values,
    "% of rows": (missing.values / len(raw) * 100).round(1),
})

mcol1, mcol2 = st.columns([1, 1])
with mcol1:
    st.dataframe(missing_df, width="stretch", hide_index=True)
with mcol2:
    st.bar_chart(missing_df.set_index("Column")["% of rows"], height=240)

st.caption(
    "Only three columns have gaps at all, and the counts above understate the true picture: "
    "`fuel_type` also holds placeholder text such as a dash or \"not supported\" on top of its "
    "true blanks, which is missingness wearing a different disguise. Each column's gaps are "
    "handled on their own terms below."
)

st.markdown("### Numeric gaps, flagged before they are filled")

st.markdown("""
For `horsepower`, `engine_liters` and `cylinders`, whatever the text patterns above could not
find is left blank. Rather than filling those blanks immediately, a binary flag is created first
to record that the value was missing — and only afterwards is the gap filled with that column's
median. Recording the flag before filling matters because once a value has been imputed there is
no way to tell it apart from a real one; if the flag were skipped, that information would be lost
for good.

This is worth doing because **the missingness is itself informative**. A listing that omits
horsepower is a different kind of listing from one that states it — often an older, cheaper, or
more casually written entry. The flag lets the model use that signal instead of throwing it away.
It earns its place: the horsepower-missing flag ranks 10th of 18 features by permutation
importance, ahead of the transmission grouping and fuel type.
""")

flag_summary = pd.DataFrame({
    "Column": ["horsepower", "engine_liters", "cylinders"],
    "Was missing": [int(with_outlier["hp_missing"].sum()),
                    int(with_outlier["liters_missing"].sum()),
                    int(with_outlier["cyl_missing"].sum())],
    "Fill value (median)": [with_outlier["horsepower"].median(),
                            with_outlier["engine_liters"].median(),
                            with_outlier["cylinders"].median()],
    "Flag column": ["hp_missing", "liters_missing", "cyl_missing"],
})
st.dataframe(flag_summary, width="stretch", hide_index=True)

st.markdown("### Why the fill uses the median, not the mean")

st.markdown(
    "The distributions below are drawn only from the rows where each value was actually stated "
    "— not imputed — so they show what a real engine spec looks like across the fleet."
)

fill_cols = [
    ("horsepower", "hp_missing", "Horsepower", "#4C72B0"),
    ("engine_liters", "liters_missing", "Engine size (litres)", "#55A868"),
    ("cylinders", "cyl_missing", "Cylinders", "#C44E52"),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (col, flag_col, label, color) in zip(axes, fill_cols):
    observed = with_outlier.loc[with_outlier[flag_col] == 0, col]
    ax.hist(observed, bins=40, color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(observed.median(), color="black", linestyle="--", linewidth=1.4,
               label=f"Median: {observed.median():.1f}")
    ax.axvline(observed.mean(), color="darkorange", linestyle="--", linewidth=1.4,
               label=f"Mean: {observed.mean():.1f}")
    ax.set_title(f"{label} (skew = {observed.skew():.2f})")
    ax.set_xlabel(label)
    ax.set_ylabel("Listings")
    ax.legend(fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("""
**Why median and not mean?** The three distributions don't all misbehave in the same way, which
is exactly why it's worth looking rather than assuming:

- **Horsepower** is clearly right-skewed (skew ≈ 0.95) — a long tail of high-performance engines
  pulls the mean noticeably above the median, visible as the gap between the two dashed lines.
- **Engine size** is close to symmetric overall (skew ≈ 0.12), but it is also lumpy, with several
  small peaks around common displacements — the median still lands inside the densest cluster of
  real engines, which a mean smeared across the lumps would not.
- **Cylinders** isn't a smooth distribution at all — it's a small set of discrete values (0, 4, 6,
  8…) dominated by 6- and 8-cylinder engines, with a thin scatter of rare 10- and 12-cylinder
  cars pulling the mean down to a non-integer 5.9 that no actual engine has. The median lands
  exactly on 6, the single most common cylinder count in the data.

In every case the median produces a fill value that is an engine some real car actually has;
the mean, at best, only approximates one.
""")

st.divider()
st.markdown("## 4. Categorical gaps → explicit categories")

cat_rules = pd.DataFrame([
    ("fuel_type", 'Placeholder text recoded to a gap, Electric recovered from the engine '
                  'description, remainder labelled "Unknown"',
     "The dash and 'not supported' strings are missingness, not fuel types. Rows whose engine "
     "text says 'Electric Motor' are recoverable with certainty rather than guessed."),
    ("accident", 'Blank labelled "Unknown"',
     "A blank may mean 'none reported' or may mean nobody checked. Guessing either way would "
     "invent information; an explicit category lets the model learn what blanks are worth."),
    ("clean_title", 'Recoded into a Yes / No-or-Unknown flag',
     "The raw column only ever holds 'Yes'. The real signal is confirmed-clean versus "
     "not-confirmed, which is exactly what the binary flag captures."),
], columns=["Column", "Rule applied", "Reasoning"])
st.dataframe(cat_rules, width="stretch", hide_index=True)

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown("**`fuel_type` after cleaning**")
    st.dataframe(clean["fuel_type"].value_counts().rename("Rows").to_frame(), width="stretch")
with fc2:
    st.markdown("**`clean_title_flag` after cleaning**")
    st.dataframe(clean["clean_title_flag"].value_counts().rename("Rows").to_frame(),
                 width="stretch")

st.success(
    "**Decision carried through every column above:** no row is ever dropped for missingness. "
    "Every gap becomes either an explicit category or a median fill accompanied by a flag "
    "recording that it was filled. Dropping rows instead would have cost roughly 15% of the "
    "dataset."
)

st.divider()
st.markdown("## 5. Bucketing 62 transmission strings into 5 groups")

b1, b2 = st.columns([1, 1])
with b1:
    st.markdown("**Before — a sample of the 62 raw strings**")
    st.dataframe(
        raw["transmission"].value_counts().head(12).rename("Rows").to_frame(),
        width="stretch",
    )
with b2:
    st.markdown("**After — 5 meaningful groups**")
    st.dataframe(
        clean["transmission_group"].value_counts().rename("Rows").to_frame(),
        width="stretch",
    )

st.markdown("""
Strings like "6-Speed A/T", "8-Speed Automatic", "10-Speed Automatic" and plain "Automatic" are
four labels for one idea. Keeping them apart would mean the model learns *automatic gearbox* four
separate times, from a quarter of the data each. Instead, each raw string is scanned for a small
set of keywords and sorted into one of five groups — checking for the more specific terms like
"CVT" or "Dual-Shift" before falling back to the generic automatic or manual test, so those
distinct gearbox types are never miscategorised as plain automatics.
""")

st.divider()
st.markdown("## 6. Deriving `car_age`")

st.markdown(f"""
Rather than keeping the model year as-is, it is converted into an age in years by subtracting it
from a fixed reference point of **{utils.REFERENCE_YEAR}** — one year past the newest model year
in the data, {raw['model_year'].max()}. Age is a more directly useful predictor than an absolute
year: depreciation is driven by how old a car *is*, not by the number stamped on it.

The reference year is fixed as a constant rather than recalculated from today's date each time
the app runs. This matters because the saved model learned the relationship between age and
price using this exact scale — recomputing it later would silently shift every prediction the
model makes.
""")

st.dataframe(
    clean[["model_year", "car_age"]].drop_duplicates().sort_values("model_year",
                                                                  ascending=False).head(6),
    width="stretch", hide_index=True,
)

st.divider()
st.markdown("## 7. Outlier investigation — one row removed")

st.markdown("""
Prices here span three orders of magnitude, so a high price is not automatically an error. A
Bugatti really does cost more than a house. The job is to separate *legitimately expensive* from
*wrong*, which means checking suspicious rows against comparable listings rather than against a
blanket threshold.
""")

st.markdown("**The 10 most expensive listings in the raw data**")
top = with_outlier.nlargest(10, "price")[["brand", "model", "model_year", "milage", "price"]]
top["price"] = top["price"].apply(utils.format_money)
st.dataframe(top, width="stretch", hide_index=True)

st.markdown("**Cross-check: every Maserati Quattroporte in the dataset**")
quattro = with_outlier[with_outlier["model"].str.contains("Quattroporte", na=False)][
    ["brand", "model", "model_year", "milage", "price"]
].sort_values("price", ascending=False)
quattro_display = quattro.copy()
quattro_display["price"] = quattro_display["price"].apply(utils.format_money)
st.dataframe(quattro_display, width="stretch", hide_index=True)

others = quattro[quattro["price"] != utils.OUTLIER_PRICE]["price"]
st.error(f"""
**Verdict: data-entry error.** Every other Quattroporte in the dataset — including models newer
than the flagged one — sits between {utils.format_money(others.min())} and
{utils.format_money(others.max())}. A 2005 base Quattroporte listed at
{utils.format_money(utils.OUTLIER_PRICE)} is not a real price; it is almost certainly an extra
digit or two. This single row is removed.
""")

st.success(f"""
**Everything else in the top 10 stays.** The Bugatti Veyron, Porsche Carrera GT and Lamborghini
Aventador are correctly priced cars that happen to be expensive. Removing them would be
convenient — they cause most of the model's headline error — but it would be trimming the data to
flatter the result. They stay, and the Final Model page reports honestly on what they cost.
Rows in: {len(raw):,} → rows out: {len(clean):,}.
""")

st.divider()
st.markdown("## Result")

nulls = clean.isna().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    st.success("No nulls remain in any column used for modelling.")
else:
    st.info(
        "Remaining nulls sit only in `clean_title`, the original raw column — it has been "
        "superseded by `clean_title_flag` and is dropped before modelling."
    )
    st.dataframe(nulls.rename("Nulls").to_frame(), width="stretch")

st.dataframe(clean.head(8), width="stretch")

st.success("Next: **EDA** in the sidebar.")
