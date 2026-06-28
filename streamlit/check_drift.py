"""
Scheduled drift check for car-price-api.

Replaces mlflow_drift.ipynb. The notebook compared two slices of the
SAME static training CSV (simulated drift). This script compares the
original training distribution against REAL production traffic logged
by gcs_logger.py, and alerts Slack if drift is found.

Run manually:   python check_drift.py
Run on schedule: see .github/workflows/drift_check.yml
"""

import os
import io
import json
import datetime
import requests
import pandas as pd
import mlflow
from google.cloud import storage
from scipy.stats import ks_2samp, chi2_contingency

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "car-price-api-data")
REFERENCE_PATH = os.environ.get("REFERENCE_PATH", "data/cars24-car-price-cleaned-new.csv")
PREDICTIONS_PREFIX = os.environ.get("PREDICTIONS_PREFIX", "predictions/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://YOUR_MLFLOW_VM_IP:5000")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

NUMERIC_FEATURES = ["km_driven", "mileage", "age"]
DRIFT_PVALUE_THRESHOLD = 0.05


def load_reference_data() -> pd.DataFrame:
    """Loads the original training CSV from GCS -- this is the baseline
    distribution everything else gets compared against."""
    client = storage.Client()
    blob = client.bucket(MODEL_BUCKET).blob(REFERENCE_PATH)
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

    # The reference CSV stores fuel type as separate one-hot columns.
    # Production logs store it as a single string column ("Petrol" etc.).
    # Collapse the one-hot columns so both datasets compare like-for-like.
    fuel_cols = ["Petrol", "Diesel", "Electric", "LPG"]
    df["fuel_type"] = df[fuel_cols].idxmax(axis=1)

    # The live API only accepts Petrol/Diesel/Electric -- drop LPG rows
    # from the reference set so the comparison is fair.
    df = df[df["fuel_type"] != "LPG"]

    return df[NUMERIC_FEATURES + ["fuel_type"]]


def load_production_data() -> pd.DataFrame:
    """Reads every JSON file gcs_logger.py has written under predictions/
    and assembles them into one dataframe of real production traffic."""
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)

    rows = [
        json.loads(blob.download_as_text())
        for blob in bucket.list_blobs(prefix=PREDICTIONS_PREFIX)
        if blob.name.endswith(".json")
    ]

    if not rows:
        raise ValueError(
            "No production data found yet under predictions/. "
            "Drift can't be checked until the API has served real traffic."
        )

    return pd.DataFrame(rows)[NUMERIC_FEATURES + ["fuel_type"]]


def send_slack_alert(message: str):
    if not SLACK_WEBHOOK_URL:
        print("SLACK_WEBHOOK_URL not set -- skipping Slack alert. Message was:")
        print(message)
        return
    requests.post(SLACK_WEBHOOK_URL, json={"text": message})


def main():
    reference_df = load_reference_data()
    production_df = load_production_data()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("drift_detection")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drift_flags = {}

    with mlflow.start_run(run_name=f"drift_check_{now}"):
        mlflow.log_metric("reference_rows", len(reference_df))
        mlflow.log_metric("production_rows", len(production_df))

        # Numeric features -- KS-test (same test your notebook already used)
        for col in NUMERIC_FEATURES:
            ks_stat, p_value = ks_2samp(reference_df[col], production_df[col])
            mlflow.log_metric(f"ks_stat_{col}", ks_stat)
            mlflow.log_metric(f"ks_pvalue_{col}", p_value)
            drift_flags[col] = p_value < DRIFT_PVALUE_THRESHOLD

        # Categorical feature -- chi-square
        ref_counts = reference_df["fuel_type"].value_counts()
        prod_counts = production_df["fuel_type"].value_counts()
        contingency = pd.concat(
            [ref_counts, prod_counts], axis=1, keys=["reference", "production"]
        ).fillna(0)

        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
        mlflow.log_metric("chi2_stat_fuel_type", chi2_stat)
        mlflow.log_metric("chi2_pvalue_fuel_type", chi2_p)
        drift_flags["fuel_type"] = chi2_p < DRIFT_PVALUE_THRESHOLD

        drifted_columns = [col for col, drifted in drift_flags.items() if drifted]
        mlflow.log_metric("any_drift_detected", int(len(drifted_columns) > 0))

    if drifted_columns:
        message = (
            ":warning: Drift detected in car-price-api production data.\n"
            f"Columns affected: {', '.join(drifted_columns)}\n"
            "Check the 'drift_detection' experiment in MLflow for details."
        )
        print(message)
        send_slack_alert(message)
    else:
        print("No drift detected.")


if __name__ == "__main__":
    main()
