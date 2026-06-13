import streamlit as st
import pickle
import numpy as np

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanNest – Rent Predictor",
    page_icon="🏙️",
    layout="centered",
)

# ─── Load model and preprocessing artifacts ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained RF model and LabelEncoder mappings once per session.

    IMPORTANT: Run this app from the SAME Python environment (venv) where
    the model was trained. Mismatched scikit-learn versions cause pickle errors.
      Windows: .\\venv\\Scripts\\streamlit run app.py
      macOS/Linux: ./venv/bin/streamlit run app.py
    """
    with open("models/best_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/preprocess_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts


try:
    model, artifacts = load_artifacts()
    encoders         = artifacts["encoders"]          # {col: LabelEncoder}
    categorical_cols = artifacts["categorical_cols"]  # ordered list
    feature_cols     = artifacts["feature_cols"]      # all 14 feature columns
except Exception as e:
    st.error(
        f"**Failed to load model:** {e}\n\n"
        "**Fix:** Run the app from the **same venv** used to train the model:\n\n"
        "```\n.\\venv\\Scripts\\streamlit run app.py\n```"
    )
    st.stop()


def get_encoder_labels(col):
    """Return sorted list of original string labels for a categorical column."""
    return list(encoders[col].classes_)


# ─── UI Header ───────────────────────────────────────────────────────────────
st.title("🏙️ UrbanNest – Dynamic Rent Prediction Engine")
st.markdown(
    "Enter property details below and click **Predict Rent** to get the "
    "estimated monthly rent (₹)."
)
st.divider()

# ─── Input widgets for every feature ────────────────────────────────────────
# Categorical columns use st.selectbox; numeric columns use st.number_input.
st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    city          = st.selectbox("City", get_encoder_labels("city"))
    location      = st.selectbox("Location", get_encoder_labels("location"))
    status        = st.selectbox("Furnishing Status", get_encoder_labels("Status"))
    property_type = st.selectbox("Property Type", get_encoder_labels("property_type"))

with col2:
    bhk           = st.number_input("BHK", min_value=0, max_value=10, value=2, step=1)
    rooms_num     = st.number_input("Total Rooms", min_value=1, max_value=20, value=3, step=1)
    num_bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
    num_balconies = st.number_input("Balconies", min_value=0, max_value=10, value=1, step=1)

st.subheader("Additional Details")
col3, col4 = st.columns(2)

with col3:
    size_sqft        = st.number_input("Size (sq ft)", min_value=50, max_value=10000, value=700, step=10)
    security_deposit = st.number_input("Security Deposit (₹)", min_value=0, max_value=1000000, value=0, step=1000)

with col4:
    latitude          = st.number_input("Latitude", value=19.076, format="%.6f")
    longitude         = st.number_input("Longitude", value=72.877, format="%.6f")
    is_negotiable     = st.selectbox("Is Negotiable?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    verification_days = st.number_input("Verification Days", min_value=0.0, max_value=5000.0, value=30.0)

st.divider()

# ─── Predict button ──────────────────────────────────────────────────────────
if st.button("🔍 Predict Rent", use_container_width=True, type="primary"):

    # Build raw (un-encoded) input dict matching the dataset columns
    raw_input = {
        "location":          location,
        "city":              city,
        "latitude":          latitude,
        "longitude":         longitude,
        "numBathrooms":      num_bathrooms,
        "numBalconies":      num_balconies,
        "isNegotiable":      is_negotiable,
        "SecurityDeposit":   security_deposit,
        "Status":            status,
        "Size_ft²":          size_sqft,
        "BHK":               bhk,
        "rooms_num":         rooms_num,
        "property_type":     property_type,
        "verification_days": verification_days,
    }

    # Apply the same LabelEncoders used during training
    input_encoded = dict(raw_input)
    for col in categorical_cols:
        if col in input_encoded:
            le  = encoders[col]
            val = str(input_encoded[col])
            if val in le.classes_:
                input_encoded[col] = int(le.transform([val])[0])
            else:
                # Unseen label — fall back to median encoded index
                input_encoded[col] = int(len(le.classes_) // 2)

    # Assemble feature vector in the exact column order used at training time
    feature_vector = np.array([[input_encoded[col] for col in feature_cols]])

    # Run inference
    predicted_rent = model.predict(feature_vector)[0]

    st.success(f"💰 Estimated Monthly Rent: **₹ {predicted_rent:,.0f}**")
    st.caption(
        "Prediction from a Random Forest model optimised via "
        "Grid Search, Random Search, and Bayesian Optimisation (Optuna)."
    )
