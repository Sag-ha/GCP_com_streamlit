"""
train.py — retrains the XGBoost car price model from scratch.

Reads training data from GCS, trains, logs everything to MLflow,
saves a NEW versioned model file back to GCS, and prints the exact
MODEL_PATH to use in the next Cloud Run deploy command.

Run manually after drift is detected:
    python streamlit/train.py

The printed MODEL_PATH is then used to update Cloud Run:
    gcloud run deploy car-price-api \
      --update-env-vars MODEL_PATH=<printed path>
"""

import os
import io
import pickle
import datetime
import matplotlib
matplotlib.use("Agg")   # no display needed when running headless
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from xgboost import XGBRegressor

# ── Config (all overrideable via env vars) ─────────────────────────────────────
MODEL_BUCKET      = os.environ.get("MODEL_BUCKET",      "car-price-api-data")
REFERENCE_PATH    = os.environ.get("REFERENCE_PATH",    "data/cars24-car-price-cleaned-new.csv")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://YOUR_MLFLOW_VM_IP:5000")

# Hyperparameters (same as your notebook)
N_ESTIMATORS  = 500
LEARNING_RATE = 0.05
MAX_DEPTH     = 20


def load_training_data() -> pd.DataFrame:
    """Reads the training CSV from GCS."""
    client = storage.Client()
    blob = client.bucket(MODEL_BUCKET).blob(REFERENCE_PATH)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def save_model_to_gcs(model, version_tag: str) -> str:
    """Pickles the model and saves it to GCS as a new versioned file.
    Returns the GCS path (used to update Cloud Run env var MODEL_PATH)."""
    model_bytes = pickle.dumps(model)
    model_path = f"models/xgb_car_price_model_{version_tag}.pkl"

    client = storage.Client()
    client.bucket(MODEL_BUCKET).blob(model_path).upload_from_string(
        model_bytes,
        content_type="application/octet-stream"
    )
    return model_path


def main():
    cars_df = load_training_data()

    X = cars_df[["km_driven", "mileage", "age", "Petrol", "Diesel", "Electric"]]
    y = cars_df[["selling_price"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("xgb_car_price")

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f"xgb_car_price_{now}"):

        model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Log params (same as your notebook)
        mlflow.log_param("n_estimators",  N_ESTIMATORS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_depth",     MAX_DEPTH)

        # Log metrics (same as your notebook)
        mlflow.log_metric("mape",                mape)
        mlflow.log_metric("rmse",                rmse)
        mlflow.log_metric("training_data_rows",  len(X_train))
        mlflow.log_metric("number_of_features",  len(X_train.columns))
        mlflow.log_metric("mean_of_mileage",     X_train["mileage"].mean())

        # Actual vs predicted scatter plot (same as your notebook)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Selling Price")
        plt.ylabel("Predicted Selling Price")
        plt.title("Actual vs Predicted Price")
        plot_file = "actual_vs_predicted.png"
        plt.savefig(plot_file)
        plt.close()
        mlflow.log_artifact(plot_file)

        mlflow.sklearn.log_model(model, "xgb_model")

        # ── Save versioned model to GCS ────────────────────────────────────────
        model_path = save_model_to_gcs(model, version_tag=now)
        mlflow.log_param("gcs_model_path", model_path)

        print(f"\nTraining complete.")
        print(f"MAPE: {mape:.4f}  |  RMSE: {rmse:.4f}")
        print(f"\nNew model saved to GCS: {model_path}")
        print(f"\nTo deploy this model, run:")
        print(f"  gcloud run deploy car-price-api \\")
        print(f"    --update-env-vars MODEL_PATH={model_path} \\")
        print(f"    --region asia-south1")


if __name__ == "__main__":
    main()
