
import json
from pathlib import Path

import joblib
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_random_forest.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

# Required input features
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# Load trained model and threshold
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    settings = json.load(file)

threshold = float(settings["threshold"])


def predict_water_quality(values):

    # Check required features
    missing = [f for f in FEATURES if f not in values]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Convert input to DataFrame
    data = pd.DataFrame(
        [{feature: values[feature] for feature in FEATURES}]
    )

    # Validate numerical inputs
    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="raise"
        )

    # Predict probability of potable water
    probability = float(
        model.predict_proba(data)[0][1]
    )

    # Apply selected threshold
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": (
            "Predicted Potable"
            if prediction == 1
            else "Predicted Non-Potable"
        ),
        "potable_probability": round(probability, 4),
        "threshold": threshold
    }


# Example prediction
if __name__ == "__main__":

    sample = {
        "ph": 7.0,
        "Hardness": 200,
        "Solids": 20000,
        "Chloramines": 7.0,
        "Sulfate": 330,
        "Conductivity": 420,
        "Organic_carbon": 14,
        "Trihalomethanes": 65,
        "Turbidity": 4.0
    }

    result = predict_water_quality(sample)

    print("\nWATER QUALITY PREDICTION")
    print("------------------------")
    print("Prediction:", result["label"])
    print("Potable Probability:", result["potable_probability"])
    print("Threshold:", result["threshold"])
    print("\nWarning: Prediction is not a drinking-water safety certification.")
