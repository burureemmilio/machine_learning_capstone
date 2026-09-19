import numpy as np
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")
utils.project_sidebar()

utils.page_header(
    "📊 The Dataset",
    "What arrived, in what condition, before anything was changed.",
    section="Sections 1–4",
)

try:
    raw, filename = utils.load_raw_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{raw.shape[0]:,}")
c2.metric("Columns", raw.shape[1])
c3.metric("Columns with gaps", int((raw.isna().sum() > 0).sum()))
c4.metric("Duplicate rows", int(raw.duplicated().sum()))

st.caption(f"Loaded from `{filename}`")

st.markdown("### Raw rows")
st.dataframe(raw.head(10), width="stretch")

st.markdown("### Columns and what they mean")
st.markdown("""
| Column | Meaning | Stored as |
|---|---|---|
| `brand` | Manufacturer | text |
| `model` | Model and trim, as written in the listing | text |
| `model_year` | Year of manufacture | integer |
| `milage` | Odometer reading (spelled this way in the source) | text — `"51,000 mi."` |
| `fuel_type` | Petrol, hybrid, diesel, electric, flex fuel | text |
| `engine` | Free-text engine description | text — see below |
| `transmission` | Gearbox description | text |
| `ext_col` / `int_col` | Exterior and interior colour | text |
| `accident` | Whether damage was reported | text |
| `clean_title` | Whether the title is clean | text — `"Yes"` or blank |
| `price` | **Target.** Listed asking price | text — `"$10,300"` |
""")

st.warning(
    "**Two of the most important columns are not numbers.** `price` and `milage` both arrive as "
    "text with currency symbols, commas and unit suffixes. Neither is usable — or even "
    "plottable — until it is parsed. That is the first job of the cleaning step, covered on the "
    "next page."
)

st.markdown("### Data types as loaded")
dtypes = pd.DataFrame({
    "Column": raw.columns,
    "Dtype": [str(t) for t in raw.dtypes],
    "Non-null": raw.notna().sum().values,
    "Unique values": [raw[c].nunique() for c in raw.columns],
})
st.dataframe(dtypes, width="stretch", hide_index=True)

st.info(
    f"Only **{(dtypes['Dtype'] != 'object').sum()}** of {raw.shape[1]} columns arrived as a "
    "numeric type. Everything else is text, including the target itself."
)

st.divider()
st.markdown("### Duplicates")
if raw.duplicated().sum() == 0:
    st.success("No fully duplicated rows — nothing to drop.")
else:
    st.warning(f"{raw.duplicated().sum()} duplicate rows found.")

st.divider()
st.markdown("### Cardinality — how many distinct values each text column holds")

cat_cols = ["brand", "model", "engine", "transmission", "ext_col", "int_col",
            "fuel_type", "accident", "clean_title"]
card = pd.DataFrame({
    "Column": cat_cols,
    "Distinct values": [raw[c].nunique() for c in cat_cols],
}).sort_values("Distinct values", ascending=False)


def encoding_plan(n):
    if n > 100:
        return "Target encoding"
    if n > 20:
        return "Bucket, then one-hot"
    return "One-hot encoding"


card["Planned treatment"] = card["Distinct values"].apply(encoding_plan)

k1, k2 = st.columns([1, 1])
with k1:
    st.dataframe(card, width="stretch", hide_index=True)
with k2:
    st.bar_chart(card.set_index("Column")["Distinct values"], height=320)

st.markdown(f"""
This single table decides the entire encoding strategy, covered in full on the Feature
Engineering page:

- **`model` ({raw['model'].nunique():,} values)**, **`ext_col` ({raw['ext_col'].nunique()})** and
  **`int_col` ({raw['int_col'].nunique()})** are far too high-cardinality for one-hot encoding —
  it would produce a matrix wider than the dataset is tall, most of it zeros. These get
  **target encoding**.
- **`brand` ({raw['brand'].nunique()})** is borderline but also goes to target encoding, since
  brand is a strong price signal and one-hot would still add 57 sparse columns.
- **`transmission` ({raw['transmission'].nunique()})** looks high-cardinality but isn't really:
  the 62 strings are mostly redundant spellings of four or five actual ideas — things like
  "6-Speed A/T", "8-Speed Automatic" and plain "Automatic" all mean the same thing. Bucketing it
  down first, then one-hot encoding the buckets, keeps the signal and drops the noise.
- **`fuel_type`, `accident`, `clean_title`** are genuinely low-cardinality → **one-hot**.
- **`engine` ({raw['engine'].nunique():,} values)** is not encoded at all. It is free text, and
  nearly every row is unique. Instead it gets *parsed* into three numeric columns — also covered
  on Data Cleaning.
""")

with st.expander("A sample of the raw `engine` strings — why parsing beats encoding"):
    st.dataframe(
        raw["engine"].drop_duplicates().head(15).to_frame("engine"),
        width="stretch", hide_index=True,
    )
    st.markdown("""
Each of these bundles three separate numeric facts into one string: horsepower, displacement in
litres, and cylinder count. Treated as a category, two nearly identical engines described with
slightly different wording become two unrelated labels. Parsed apart, they are recognised as
nearly the same car.
""")

st.markdown("### The target, pre-cleaning")
st.markdown(
    "`price` cannot be summarised numerically yet — every value is still stored as text with a "
    "dollar sign and commas, such as `\"$10,300\"`. That alone is the argument for the cleaning "
    "step coming first, before any distribution or correlation can even be plotted."
)

st.success("Next: **Data Cleaning** in the sidebar.")
