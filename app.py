# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Labmentix\RealEstateInvestmentAdvisor")
DATA_PATH = ROOT / "data" / "cleaned_housing_data.csv"
CLF_PATH = ROOT / "models" / "investment_classifier_memsafe.pkl"
REG_PATH = ROOT / "models" / "future_price_regressor_memsafe.pkl"

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# -------------------------
# Utilities: load artifacts
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg

# frequency maps used in training (City, Locality, Amenities etc.)
@st.cache_data(show_spinner=False)
def build_freq_maps(df, high_card_cols):
    freq_maps = {}
    for c in high_card_cols:
        freq = df[c].value_counts(dropna=False)
        freq_maps[c] = freq.to_dict()
    return freq_maps

# apply freq encoding to single-row df (in-place)
def apply_freq_encoding(row_df, freq_maps):
    for col, fmap in freq_maps.items():
        fname = f"{col}_freq"
        fname_ratio = f"{col}_freq_ratio"
        val = row_df.iloc[0].get(col, np.nan)
        cnt = fmap.get(val, 0)
        row_df[fname] = cnt
        row_df[fname_ratio] = cnt / float(sum(fmap.values())) if sum(fmap.values()) > 0 else 0.0
    return row_df

# helper to compute price per sqft
def compute_price_per_sqft(price_lakhs, size_sqft):
    try:
        return (price_lakhs * 100000.0) / float(size_sqft)
    except Exception:
        return np.nan

# small helper to format currency
def format_inr_lakhs(x):
    return f"₹ {x:,.2f} Lakhs"

# -------------------------
# Load data & models
# -------------------------
df = load_data()
clf, reg = load_models()

# Determine which categorical cols were treated as high-card (same logic used in training)
# Threshold used in training scripts:
CARD_THRESHOLD = 30
all_cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
high_card_cols = [c for c in all_cat_cols if df[c].nunique() > CARD_THRESHOLD]
low_card_cols = [c for c in all_cat_cols if df[c].nunique() <= CARD_THRESHOLD]

freq_maps = build_freq_maps(df, high_card_cols)

# -------------------------
# Sidebar: quick info & examples
# -------------------------
st.sidebar.title("Real Estate Advisor")
st.sidebar.markdown(
    """
    Enter property details and get:
    - Whether this looks like a *Good Investment* (classifier)
    - Estimated price after **5 years** (regression)
    """
)
st.sidebar.write("Models loaded from:")
st.sidebar.write(f"- {CLF_PATH.name}")
st.sidebar.write(f"- {REG_PATH.name}")
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show sample dataset (first 5 rows)"):
    st.sidebar.dataframe(df.head())

# -------------------------
# Main: Header
# -------------------------
st.title("🏠 Real Estate Investment Advisor — Full App")
st.markdown(
    "Enter property details below. App uses your trained models and the project's preprocessing to give a recommendation."
)

# -------------------------
# Input form (left) + Output (right)
# -------------------------
with st.form("property_form", clear_on_submit=False):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Property Location & Type")
        state = st.selectbox("State", options=sorted(df["State"].dropna().unique().tolist()), index=0)
        city = st.text_input("City", value="Bengaluru")
        locality = st.text_input("Locality", value="Whitefield")
        property_type = st.selectbox("Property Type", options=sorted(df["Property_Type"].dropna().unique().tolist()), index=0)

        st.subheader("Physical Attributes")
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
        size_sqft = st.number_input("Size (SqFt)", min_value=200, max_value=20000, value=1200)
        price_lakhs = st.number_input("Current Price (Lakhs)", min_value=0.1, value=80.0, step=0.5)
        price_per_sqft = compute_price_per_sqft(price_lakhs, size_sqft)

    with col2:
        st.subheader("Amenities & Accessibility")
        furnished = st.selectbox("Furnished Status", options=sorted(df["Furnished_Status"].dropna().unique().tolist()))
        age = st.number_input("Age of Property (years)", min_value=0, max_value=200, value=5)
        nearby_schools = st.number_input("Nearby Schools (count)", min_value=0, max_value=100, value=3)
        nearby_hospitals = st.number_input("Nearby Hospitals (count)", min_value=0, max_value=100, value=2)
        public_transport = st.selectbox("Public Transport Accessibility", options=sorted(df["Public_Transport_Accessibility"].dropna().unique().tolist()))
        parking = st.selectbox("Parking Space", options=sorted(df["Parking_Space"].dropna().unique().tolist()))
        security = st.selectbox("Security", options=sorted(df["Security"].dropna().unique().tolist()))
        facing = st.selectbox("Facing", options=sorted(df["Facing"].dropna().unique().tolist()))
        owner_type = st.selectbox("Owner Type", options=sorted(df["Owner_Type"].dropna().unique().tolist()))
        availability = st.selectbox("Availability Status", options=sorted(df["Availability_Status"].dropna().unique().tolist()))
        amenities = st.text_input("Amenities (comma-separated)", value="Pool,Park")  # free text; will be mapped as string

    submitted = st.form_submit_button("Predict")

# -------------------------
# When submitted: build row, preprocess & predict
# -------------------------
if submitted:
    # Build a single-row dataframe using the same column names as in cleaned dataset
    # We will add missing columns with defaults if needed.
    input_row = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_per_sqft,
        "Furnished_Status": furnished,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        # keep Amenities as single string (exact column name used in training)
        "Amenities": amenities
    }

    # Add any other numeric columns present in training but not present above with default 0
    training_cols = df.columns.tolist()
    for c in training_cols:
        if c not in input_row and c not in ["Good_Investment", "Future_Price_5Y", "City_Median_Price_per_SqFt"]:
            # set sensible defaults depending on type
            if c in ["Price_per_SqFt", "Age_of_Property"]:
                input_row[c] = 0
            else:
                # fallback numeric default
                if df[c].dtype.kind in "biufc":
                    input_row[c] = 0
                else:
                    input_row[c] = df[c].mode().iloc[0] if not df[c].mode().empty else ""

    input_df = pd.DataFrame([input_row])

    # Ensure numeric columns dtype consistent
    numeric_cols_training = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in numeric_cols_training:
        if c in input_df.columns:
            try:
                input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0)
            except Exception:
                input_df[c] = 0

    # Apply same frequency encoding as training (using freq_maps built from training df)
    input_df = apply_freq_encoding(input_df, freq_maps)
    # -------------------------
    # Ensure engineered columns expected by pipelines exist
    # -------------------------
    # 1) Add City_Median_Price_per_SqFt (used in training)
    if "City" in input_df.columns and "Price_per_SqFt" in df.columns:
        # compute median price_per_sqft for the same city from training data
        city_name = input_df.loc[0, "City"]
        city_median = df.loc[df["City"] == city_name, "Price_per_SqFt"].median()
        # fallback to overall median if city not present in training data
        if pd.isna(city_median):
            city_median = df["Price_per_SqFt"].median()
        input_df["City_Median_Price_per_SqFt"] = city_median
    else:
        # fallback: overall median
        input_df["City_Median_Price_per_SqFt"] = df["Price_per_SqFt"].median()

    # 2) Double-check any other columns the pipeline might expect:
    #    If the model complains about any other missing column (KeyError/ValueError),
    #    we'll add it here with a sensible default derived from training df.
    expected_extra_cols = ["City_Median_Price_per_SqFt"]  # add more names if error shows them missing
    for c in expected_extra_cols:
        if c not in input_df.columns:
            if c in df.columns:
                input_df[c] = df[c].median()  # median as safe numeric default
            else:
                input_df[c] = 0

    # Ensure columns order/availability - pipeline will ignore extras and expect the feature names we've prepared
    # Use the classifier and regressor to predict
    try:
        clf_proba = clf.predict_proba(input_df)[0][1] if hasattr(clf, "predict_proba") else None
        clf_pred = int(clf.predict(input_df)[0])
    except Exception as e:
        st.error("Error running classification model. The app couldn't match input columns to the model's expected features.")
        st.exception(e)
        st.stop()

    try:
        reg_pred = reg.predict(input_df)[0]
    except Exception as e:
        st.error("Error running regression model. The app couldn't match input columns to the model's expected features.")
        st.exception(e)
        st.stop()

    # -------------------------
    # Output panel
    # -------------------------
    left, right = st.columns([1, 1])

    with left:
        st.markdown("## Recommendation")
        if clf_pred == 1:
            st.success(f"✅ This appears to be a GOOD investment (confidence: {clf_proba:.2f})")
        else:
            st.warning(f"⚠️ This may NOT be a good investment (confidence: {clf_proba:.2f})")

        st.markdown("### Estimated Price (5 years)")
        st.info(f"{format_inr_lakhs(reg_pred)}")

        st.markdown("### Summary")
        # Short rules to explain model decision (simple human-friendly heuristics)
        bullets = []
        bullets.append(f"- Current price per sqft: {input_df['Price_per_SqFt'].iloc[0]:.2f}")
        bullets.append(f"- BHK: {int(input_df['BHK'].iloc[0])}, Size: {int(input_df['Size_in_SqFt'].iloc[0])} sqft")
        bullets.append(f"- Nearby schools: {int(input_df['Nearby_Schools'].iloc[0])}, Nearby hospitals: {int(input_df['Nearby_Hospitals'].iloc[0])}")
        bullets.append(f"- Furnishing: {input_df['Furnished_Status'].iloc[0]}")
        st.markdown("\n".join(bullets))

    with right:
        st.markdown("## Visuals")
        # small bar showing current vs predicted
        fig, ax = plt.subplots(figsize=(5,3))
        current = input_df["Price_in_Lakhs"].iloc[0]
        predicted = reg_pred
        ax.bar(["Now"], [current], label="Current")
        ax.bar(["5 Yrs"], [predicted], label="Predicted")
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title("Current vs Predicted Price")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Model details")
        st.write("- Classifier: RandomForest (memory-safe pipeline)")
        st.write("- Regressor: RandomForest (memory-safe pipeline)")
        if st.checkbox("Show input dataframe used for prediction"):
            st.dataframe(input_df.T)

    # Save option
    if st.button("Save this prediction to CSV"):
        out_path = ROOT / "predictions" 
        out_path.mkdir(exist_ok=True, parents=True)
        fname = out_path / f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        input_df["Pred_Good_Investment"] = clf_pred
        input_df["Pred_5Y_Price_Lakhs"] = reg_pred
        input_df.to_csv(fname, index=False)
        st.success(f"Saved prediction to {fname}")

st.markdown("---")
st.write("Built with ❤️ — RealEstateInvestmentAdvisor")
