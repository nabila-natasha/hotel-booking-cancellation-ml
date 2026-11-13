#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import date

from predict import load_artifact, predict_cancellation


# Initialise the Flask application
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# Load the model once when the application starts
artifact = load_artifact()


# Define a route for homepage
@app.route('/', methods=['GET'])
def home():
    """Display the hotel booking prediction form."""
    return render_template(
        'index.html',
        today=date.today().isoformat()
    )


@app.route('/booking-prediction', methods=['GET', 'POST'])
def booking_prediction():
    """Process the booking form and display the prediction."""

    if request.method == 'GET':
        return redirect(url_for('home'))

    try:
        booking_date = date.fromisoformat(
            request.form['booking_date']
        )

        arrival_date = date.fromisoformat(
            request.form['arrival_date']
        )

        lead_time = (arrival_date - booking_date).days

        if lead_time < 0:
            raise ValueError(
                "Arrival date cannot be earlier than booking entry date."
            )

        booking = {
            'hotel': request.form['hotel'],
            'lead_time': lead_time,

            'arrival_date_year': arrival_date.year,
            'arrival_date_month': arrival_date.strftime('%B').lower(),
            'arrival_date_week_number': arrival_date.isocalendar().week,
            'arrival_date_day_of_month': arrival_date.day,

            'stays_in_weekend_nights': int(
                request.form['stays_in_weekend_nights']
            ),
            'stays_in_week_nights': int(
                request.form['stays_in_week_nights']
            ),

            'adults': int(request.form['adults']),
            'children': float(request.form['children']),
            'babies': int(request.form['babies']),

            'meal': request.form['meal'],
            'country': request.form['country'],
            'market_segment': request.form['market_segment'],
            'distribution_channel': request.form['distribution_channel'],

            'is_repeated_guest': int(
                request.form['is_repeated_guest']
            ),
            'previous_cancellations': int(
                request.form['previous_cancellations']
            ),
            'previous_bookings_not_canceled': int(
                request.form['previous_bookings_not_canceled']
            ),

            'reserved_room_type': request.form['reserved_room_type'],
            'deposit_type': request.form['deposit_type'],

            'agent': request.form['agent'],

            'days_in_waiting_list': int(
                request.form['days_in_waiting_list']
            ),

            'customer_type': request.form['customer_type'],

            'adr': float(request.form['adr']),

            'total_of_special_requests': int(
                request.form['total_of_special_requests']
            ),

            'has_company': int(request.form['has_company'])
        }

        result = predict_cancellation(booking, artifact)

        return render_template(
            'index.html',
            today=date.today().isoformat(),
            prediction=result
        )

    except (ValueError, KeyError) as error:
        return render_template(
            'index.html',
            today=date.today().isoformat(),
            error=str(error)
        ), 400


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
        debug=False,
        use_reloader=False,
        host="0.0.0.0",
        port=9696)
