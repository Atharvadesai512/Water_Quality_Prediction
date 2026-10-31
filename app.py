
import streamlit as st
from src.predict import predict_water_quality

# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

# Title
st.title("💧 Water Quality Prediction System")

st.write(
    "Machine Learning based water potability "
    "prediction using physicochemical features."
)

st.divider()

# Sidebar
st.sidebar.header("About This Project")

st.sidebar.write(
    "This application uses a tuned Random Forest "
    "classifier trained on water-quality data."
)

st.sidebar.info(
    "Model: Random Forest\n\n"
    "Features: 9\n\n"
    "Classification: Binary"
)

# Input section
st.subheader("Enter Water Quality Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH", min_value=0.0, max_value=14.0, value=7.0
    )

    hardness = st.number_input(
        "Hardness", min_value=0.0, value=200.0
    )

    solids = st.number_input(
        "Solids", min_value=0.0, value=20000.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines", min_value=0.0, value=7.0
    )

    sulfate = st.number_input(
        "Sulfate", min_value=0.0, value=330.0
    )

    conductivity = st.number_input(
        "Conductivity", min_value=0.0, value=420.0
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon", min_value=0.0, value=14.0
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes", min_value=0.0, value=65.0
    )

    turbidity = st.number_input(
        "Turbidity", min_value=0.0, value=4.0
    )

st.divider()

# Prediction button
if st.button("Predict Water Quality", type="primary"):

    sample = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    try:
        result = predict_water_quality(sample)

        st.subheader("Prediction Results")

        score = result["potable_probability"]
        threshold = result["threshold"]

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Model Potability Score",
                f"{score * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                "Classification Threshold",
                f"{threshold:.2f}"
            )

        st.progress(min(max(score, 0.0), 1.0))

        if result["prediction"] == 1:
            st.info(
                "Model Prediction: POTABLE\n\n"
                "The model classified this sample "
                "as potable. This does not confirm "
                "that the water is safe to drink."
            )
        else:
            st.warning(
                "Model Prediction: NON-POTABLE\n\n"
                "The model classified this sample "
                "as non-potable."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Educational ML demonstration only. "
    "Predictions are not laboratory-tested "
    "water safety certifications."
)
