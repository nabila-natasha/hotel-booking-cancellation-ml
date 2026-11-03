#!/usr/bin/env python
# coding: utf-8

import json
import sys
from pathlib import Path


# Make the src directory importable for this test
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from app import app


# Load a sample booking
booking_file = PROJECT_DIR / "test_data" / "booking_02.json"

with open(booking_file, "r") as f_in:
    booking = json.load(f_in)


# Create Flask's test client
client = app.test_client()


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_predict():

    response = client.post("/predict", json=booking)

    assert response.status_code == 200

    result = response.json

    assert "cancellation_probability" in result
    assert "is_canceled" in result
    assert "threshold" in result

    assert 0 <= result["cancellation_probability"] <= 1
    assert isinstance(result['is_canceled'], bool)
