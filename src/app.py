#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify

from predict import load_artifact, predict_cancellation


# Initialise the Flask application
app = Flask("hotel-booking-cancellation-prediction")


# Load the model once when the application starts
artifact = load_artifact()


# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Predict the cancellation probability for hotel bookings."""

    # Get the JSON data from the request
    booking = request.get_json() # get data from request and parse JSON into Python dictionary

    # Make a prediction using the loaded model
    result = predict_cancellation(booking, artifact)

    # Return the prediction result as JSON
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Check whether the API is running."""

    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":
    # Run the Flask application
    app.run(
        debug=True,
        host="0.0.0.0",
        port=9696)
