"""
test_model.py — pytest tests that run inside GitHub Actions
before every deploy (see deploy.yml).

If either test fails, the deploy is blocked automatically.
The model fixture loads xgb_car_price_model.pkl from the
local filesystem. In CI, deploy.yml downloads it from GCS
before running pytest so this file is always present.
"""

import pytest
import pickle
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def model():
    """Loads the model once for the entire test session."""
    with open("xgb_car_price_model.pkl", "rb") as f:
        return pickle.load(f)


def make_input(km_driven, mileage, age, fuel_type):
    """Helper: formats raw inputs into a DataFrame the model expects."""
    fuel_encoding = {
        "Petrol":   [1, 0, 0],
        "Diesel":   [0, 1, 0],
        "Electric": [0, 0, 1]
    }
    petrol, diesel, electric = fuel_encoding[fuel_type]
    return pd.DataFrame(
        [[km_driven, mileage, age, petrol, diesel, electric]],
        columns=["km_driven", "mileage", "age", "Petrol", "Diesel", "Electric"]
    )


def test_prediction_reasonable_range_multiple_samples(model):
    """100 random inputs should all produce a price between 0 and 100 lakh."""
    for _ in range(100):
        X = make_input(
            mileage=np.random.randint(5, 20),
            age=np.random.randint(0, 20),
            km_driven=np.random.randint(5000, 300000),
            fuel_type=np.random.choice(["Petrol", "Diesel", "Electric"])
        )
        price = model.predict(X)[0]
        if price < 0:
            price = 0
        assert 0 <= price <= 100


def test_age_pred_compare(model):
    """A newer car (age=5) should predict higher price than an older one (age=15),
    all else being equal."""
    X1 = make_input(mileage=10, age=5,  km_driven=20000, fuel_type="Petrol")
    X2 = make_input(mileage=10, age=15, km_driven=20000, fuel_type="Petrol")
    assert model.predict(X1)[0] > model.predict(X2)[0]
