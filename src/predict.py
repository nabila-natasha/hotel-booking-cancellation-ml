#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import xgboost as xgb


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_DIR / "models" / "xgboost_model.bin"



def load_artifact(model_file=MODEL_FILE):
    """Load the trained model and its preprocessing objects."""

    with open(MODEL_FILE, "rb") as f_in:
        artifact = pickle.load(f_in)

    return artifact


def predict_cancellation(booking, artifact):
    """"Predict the cancellation probability for one hotel booking."""

    # Work with a copy so we do not modify the caller's dictionary
    booking = booking.copy()

    # The training script treats agent as a categorical identifier
    booking['agent'] = str(booking.get('agent', ''))

    # The target must never be included in prediction features
    if 'is_canceled' in booking:
        del booking['is_canceled']


    dv = artifact["dv"]
    model = artifact["model"]
    threshold = artifact["threshold"]

    # Transform the booking data into a feature vector
    X = dv.transform([booking])

    # XGBoost expect a DMatrix object as input
    features = list(dv.get_feature_names_out())

    dmatrix = xgb.DMatrix(
        X,
        feature_names=features
    )

    # Get the predicted probability of cancellation
    y_pred = float(model.predict(dmatrix)[0])

    # Convert the predicted probability into a binary prediction
    is_canceled = y_pred >= threshold

    return {
        "cancellation_probability": float(y_pred),
        "is_canceled": bool(is_canceled),
        "threshold": float(threshold)
    }


if __name__ == "__main__":

    # Load the trained model and its preprocessing objects
    artifact = load_artifact()

    print("Model artifact loaded successfully.")
    print("Model file:", MODEL_FILE)