#!/usr/bin/env python
# coding: utf-8


from predict import load_artifact, predict_cancellation


booking = {
    "hotel": "city_hotel",
    "lead_time": 0,
    "arrival_date_year": 2026,
    "arrival_date_month": "november",
    "arrival_date_week_number": 1,
    "arrival_date_day_of_month": 3,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 1,
    "adults": 3,
    "children": 2.0,
    "babies": 0,
    "meal": "bb",
    "country": "prt",
    "market_segment": "online_ta",
    "distribution_channel": "ta/to",
    "is_repeated_guest": 1,
    "previous_cancellations": 1,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "a",
    "deposit_type": "no_deposit",
    "agent": 8,
    "days_in_waiting_list": 0,
    "customer_type": "transient",
    "adr": 220.0,
    "total_of_special_requests": 3,
    "has_company": 1
}


artifact = load_artifact()

result = predict_cancellation(booking, artifact)


print("\n=== Hotel Booking Cancellation Prediction ===\n")

print("Booking Details:", booking)
print("\nPrediction:", result)

if result['is_canceled']:
    print("\nThis booking is classified as likely to be cancelled.")
else:
    print("\nThis booking is classified as unlikely to be cancelled.")