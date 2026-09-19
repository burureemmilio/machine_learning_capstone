import numpy as np
import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="Price Predictor", page_icon="💰", layout="wide")
utils.project_sidebar()

utils.page_header(
    "💰 Price Predictor",
    "Describe a car and get an estimated listing price from the trained model.",
    section="Section 24",
)

try:
    raw, _ = utils.load_raw_data()
    df = utils.prepare_data(raw)
    model = utils.load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

brands = utils.safe_options(df["brand"])
ext_colors = utils.safe_options(df["ext_col"])
int_colors = utils.safe_options(df["int_col"])
fuel_types = utils.safe_options(df["fuel_type"])
accidents = utils.safe_options(df["accident"])

st.markdown("### Describe the car")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Vehicle**")
    brand = st.selectbox(
        "Brand", brands,
        index=brands.index("Ford") if "Ford" in brands else 0,
        key="brand_select",
    )
    # Recomputed every time the brand changes, so the model list always matches it.
    brand_models = utils.safe_options(df.loc[df["brand"] == brand, "model"])
    model_name = st.selectbox(
        "Model", brand_models,
        key=f"model_select_{brand}",
        help="Only models listed under the selected brand are shown.",
    )
    model_year = st.slider(
        "Model year",
        int(df["model_year"].min()), utils.REFERENCE_YEAR - 1,
        2019,
    )
    milage = st.number_input(
        "Mileage", min_value=0, max_value=500_000, value=40_000, step=1_000,
        help="Odometer reading in miles.",
    )

with c2:
    st.markdown("**Engine & drivetrain**")
    known_specs = st.checkbox(
        "I know the engine specs", value=True,
        help="Untick if the specs are unknown — the model will impute them and flag that "
             "it did, exactly as it was trained to.",
    )
    if known_specs:
        horsepower = st.number_input("Horsepower", 50.0, 1500.0, 300.0, 10.0)
        engine_liters = st.number_input("Engine size (litres)", 0.0, 9.0, 3.0, 0.1)
        cylinders = st.number_input("Cylinders", 0, 16, 6, 1)
        hp_missing = liters_missing = cyl_missing = 0
    else:
        horsepower = float(df["horsepower"].median())
        engine_liters = float(df["engine_liters"].median())
        cylinders = float(df["cylinders"].median())
        hp_missing = liters_missing = cyl_missing = 1
        st.caption(
            f"Using dataset medians: {horsepower:.0f} HP, {engine_liters:.1f} L, "
            f"{cylinders:.0f} cylinders — with the missingness flags set."
        )

    fuel_type = st.selectbox(
        "Fuel type", fuel_types,
        index=fuel_types.index("Gasoline") if "Gasoline" in fuel_types else 0,
    )
    transmission_group = st.selectbox("Transmission", utils.TRANSMISSION_GROUPS)

with c3:
    st.markdown("**Condition & appearance**")
    ext_col = st.selectbox(
        "Exterior colour", ext_colors,
        index=ext_colors.index("Black") if "Black" in ext_colors else 0,
    )
    int_col = st.selectbox(
        "Interior colour", int_colors,
        index=int_colors.index("Black") if "Black" in int_colors else 0,
    )
    accident = st.selectbox("Accident history", accidents)
    clean_title_flag = st.selectbox(
        "Clean title", ["Yes", "No/Unknown"],
        help="'No/Unknown' covers both a non-clean title and a listing that never stated one.",
    )

submitted = st.button("Estimate price", type="primary", width="stretch")

if not submitted:
    st.info("Fill in the details above and select **Estimate price**.")
    st.stop()

is_electric = int("electric" in str(fuel_type).lower())
if is_electric:
    engine_liters, cylinders = 0.0, 0.0
    liters_missing = cyl_missing = 0

features = utils.build_feature_row(
    brand=brand,
    model_name=model_name,
    model_year=model_year,
    milage=milage,
    fuel_type=fuel_type,
    ext_col=ext_col,
    int_col=int_col,
    accident=accident,
    transmission_group=transmission_group,
    clean_title_flag=clean_title_flag,
    horsepower=horsepower,
    engine_liters=engine_liters,
    cylinders=cylinders,
    is_electric=is_electric,
    hp_missing=hp_missing,
    liters_missing=liters_missing,
    cyl_missing=cyl_missing,
)

try:
    prediction = float(model.predict(features)[0])
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.divider()
st.success(f"✅ **Successfully predicted a price for the {model_year} {brand} {model_name}.**")

st.markdown("### The car that was priced")
det1, det2, det3, det4 = st.columns(4)
det1.metric("Vehicle", f"{brand} {model_name}")
det2.metric("Year", model_year)
det3.metric("Mileage", f"{milage:,.0f} mi")
det4.metric("Car age", f"{utils.REFERENCE_YEAR - model_year} years")

det5, det6, det7, det8 = st.columns(4)
det5.metric("Engine", f"{horsepower:.0f} HP, {engine_liters:.1f} L")
det6.metric("Cylinders", f"{cylinders:.0f}" if not is_electric else "Electric")
det7.metric("Transmission", transmission_group)
det8.metric("Fuel type", fuel_type)

st.caption(
    f"Colours: {ext_col} exterior / {int_col} interior  ·  Accident history: {accident}  ·  "
    f"Clean title: {clean_title_flag}"
)

st.markdown("### Estimated price")

mape = utils.TEST_METRICS["mape"] / 100
low, high = prediction * (1 - mape), prediction * (1 + mape)

p1, p2 = st.columns(2)
p1.metric("Estimate", utils.format_money(prediction))
p2.metric("Typical range", f"{utils.format_money(low)} – {utils.format_money(high)}")

st.info(f"""
**Where the typical range comes from.** When the trained model was scored on the 802 cars it had
never seen during testing, its predictions missed the actual listed price by
**{utils.TEST_METRICS['mape']:.1f}% on average** — that figure is the model's mean absolute
percentage error, measured on the Final Model page. The range shown above simply applies that
same average percentage miss to this prediction: {utils.format_money(prediction)} minus
{utils.TEST_METRICS['mape']:.1f}% on the low end, and {utils.format_money(prediction)} plus
{utils.TEST_METRICS['mape']:.1f}% on the high end.

It is a rough guide to how far off a typical estimate tends to be, not a formal statistical
confidence interval — accuracy is noticeably better for ordinary cars than for rare or very
expensive ones, so the true uncertainty for any one prediction can sit above or below this band.
""")

if prediction > 150_000:
    st.warning(
        "Predictions above roughly $150,000 sit in the price bracket where the model's accuracy "
        "degrades — see the price-bracket breakdown on the Final Model page."
    )

st.divider()

with st.expander("What the model received"):
    st.caption(
        "The exact 18-column row passed to the pipeline. `car_age` and the missingness flags "
        "are derived automatically — they are not free-text inputs."
    )
    st.dataframe(features.T.rename(columns={0: "Value"}), width="stretch")
