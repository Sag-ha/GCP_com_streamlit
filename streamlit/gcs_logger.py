import json
from datetime import datetime
from google.cloud import storage

BUCKET_NAME = "car-price-api-data"


def log_prediction(row: dict):
    """Writes one JSON file per request to GCS under predictions/.

    One file per request (instead of appending to one growing CSV)
    avoids read-modify-write race conditions when Cloud Run runs
    multiple instances at once.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    row["timestamp"] = datetime.utcnow().isoformat()
    blob_name = f"predictions/{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"

    bucket.blob(blob_name).upload_from_string(
        json.dumps(row),
        content_type="application/json"
    )
