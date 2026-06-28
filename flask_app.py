import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
from google.cloud import storage
from gcs_logger import log_prediction

app = Flask(__name__)

# Model location is never hardcoded -- it's pinned per-deployment via
# env vars (set with `gcloud run deploy --set-env-vars` / `--update-env-vars`).
# This lets you roll out a retrained model without rebuilding the image,
# while every Cloud Run revision still records exactly which model it used.
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_PATH = os.environ["MODEL_PATH"]


def load_model():
    client = storage.Client()
    blob = client.bucket(MODEL_BUCKET).blob(MODEL_PATH)
    return pickle.loads(blob.download_as_bytes())


# Loaded once at container startup, not per-request.
model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    km_driven = data["km_driven"]
    mileage = data["mileage"]
    age = data["age"]
    fuel_type = data["fuel_type"]

    fuel_encoding = {
        "Petrol": [1, 0, 0],
        "Diesel": [0, 1, 0],
        "Electric": [0, 0, 1]
    }
    petrol, diesel, electric = fuel_encoding[fuel_type]

    input_df = pd.DataFrame(
        [[km_driven, mileage, age, petrol, diesel, electric]],
        columns=["km_driven", "mileage", "age", "Petrol", "Diesel", "Electric"]
    )

    prediction = model.predict(input_df)[0]

    # Log every request for drift monitoring. Wrapped so a logging
    # failure never breaks the actual prediction response.
    try:
        log_prediction({**data, "predicted_price": float(prediction)})
    except Exception as e:
        print(f"Logging failed (prediction still returned): {e}")

    return jsonify({"predicted_price": float(prediction)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
