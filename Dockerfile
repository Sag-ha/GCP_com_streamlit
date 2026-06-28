FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY flask_app.py .
COPY gcs_logger.py .

# Note: xgb_car_price_model.pkl is intentionally NOT copied in anymore.
# The model is now loaded from GCS at container startup using
# MODEL_BUCKET / MODEL_PATH env vars (see flask_app.py). This means a
# new model version no longer requires rebuilding this image.

EXPOSE 5000
CMD ["python", "flask_app.py"]

# Known open item (already flagged in your original Dockerfile):
# CMD still runs Flask's dev server, not Gunicorn. Fine for now, but
# swap to gunicorn before this sees real production traffic:
#   RUN pip install gunicorn  (add to requirements.txt)
#   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "flask_app:app"]
