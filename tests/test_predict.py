
import pytest

from src.predict import predict_water_quality, FEATURES

# Sample water-quality measurements
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


# Test 1: Valid prediction
def test_prediction():

    result = predict_water_quality(sample)

    assert result["prediction"] in [0, 1]
    assert result["label"] in [
        "Predicted Potable",
        "Predicted Non-Potable"
    ]


# Test 2: Probability range
def test_probability():

    result = predict_water_quality(sample)

    assert 0 <= result["potable_probability"] <= 1


# Test 3: Missing feature
def test_missing_feature():

    invalid_sample = sample.copy()
    invalid_sample.pop("ph")

    with pytest.raises(ValueError):
        predict_water_quality(invalid_sample)


# Test 4: Invalid data type
def test_invalid_input():

    invalid_sample = sample.copy()
    invalid_sample["Hardness"] = "invalid"

    with pytest.raises(ValueError):
        predict_water_quality(invalid_sample)


# Test 5: Consistent predictions
def test_consistency():

    result1 = predict_water_quality(sample)
    result2 = predict_water_quality(sample)

    assert result1 == result2


# Test 6: Feature count
def test_feature_count():

    assert len(FEATURES) == 9
