# Hotel Booking Cancellation Prediction

## Project Overview

This project develops a machine learning classifier that estimates the probability that a hotel booking will be cancelled.

The goal is to demonstrate an end-to-end machine learning workflow, covering exploratory data analysis, data cleaning, model development, hyperparameter tuning, model training, prediction, API development and automated testing.

**Project status**

| Stage | Status |
|---|---|
| Data audit, cleaning and EDA | Completed |
| Model development and hyperparameter tuning | Completed |
| Model training script and saved model artifact | Completed |
| Prediction function | Completed |
| Flask prediction API | Completed |
| API automated tests | Completed (2 tests passing) |
| Docker containerization | Planned |
| Cloud deployment | Planned |

## Business Problem

Hotel cancellations affect room availability, revenue planning and operational decisions.

A cancellation-risk model could help a hotel's revenue-management team identify bookings that may need attention. Potential actions include sending booking reminders, reviewing deposit policies or planning room availability more carefully.

**Intended use:** Support booking-risk assessment and operational planning. Predictions are estimates, not guarantees, and should not be the sole basis for decisions affecting customers.

**Prediction point:** The model estimates cancellation risk using the booking information supplied at prediction time. In a real deployment, the hotel would need to ensure that each input feature is available when the prediction is made.

## Dataset

- **Dataset:** Hotel Booking Demand, covering a city hotel and a resort hotel in Portugal.
- **Source:** [TidyTuesday, 11 February 2020](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md)
- **Original research:** Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41–49.
- **Prediction target:** `is_canceled` (1 = cancelled, 0 = not cancelled).
- **Raw dataset:** 119,390 rows and 32 columns.
- **Cleaned dataset:** 84,392 rows and 27 columns.

| File | Description |
|---|---|
| `data/raw/hotels.csv` | Original dataset |
| `data/processed/hotels_clean.csv` | Cleaned dataset used for modelling |

If the raw dataset is unavailable, the first notebook downloads it from the TidyTuesday repository.

## Data Cleaning Decisions

The decisions below are documented with supporting analysis in `notebooks/01_eda_cleaning.ipynb`.

| Issue | Finding | Decision |
|---|---|---|
| Outcome leakage | `reservation_status` matches the target; `reservation_status_date` is recorded after the outcome | Dropped both columns |
| Information recorded after booking | `booking_changes`, `assigned_room_type` and `required_car_parking_spaces` may contain information unavailable at the intended prediction point | Dropped these columns |
| Duplicate rows | Approximately 34,000 exact duplicates after removing the leakage and post-booking columns | Removed duplicates, keeping the first occurrence |
| Missing `company` values | Approximately 94% missing | Created `has_company` as a binary feature and dropped the original column |
| Missing `agent` values | Approximately 14% missing | Filled with `0` and treated as a categorical feature |
| Missing `country` and `children` values | Small number of missing values | Filled with `"unknown"` and `0`, respectively |
| Invalid or extreme values | Records with invalid guest counts, zero-night stays, negative prices or extreme values | Removed affected records |

## Exploratory Data Analysis: Key Findings

- Approximately 27.9% of cleaned bookings were cancelled.
- A model that always predicts "not cancelled" would achieve approximately 72.1% accuracy without identifying any cancellations.
- Cancellation patterns vary with lead time, booking channel, country and the number of special requests.
- These patterns were investigated further during model development.

## Machine Learning Workflow

The project follows these steps:

1. Data audit, cleaning and exploratory data analysis.
2. Stratified train, validation and test split (60/20/20).
3. Feature preprocessing using `DictVectorizer`.
4. Baseline model development.
5. Tree-based model development and comparison.
6. XGBoost hyperparameter tuning using cross-validation.
7. Model evaluation using classification metrics, including ROC-AUC, precision and recall.
8. Saving the trained model and preprocessing object.
9. Loading the saved artifact for individual booking predictions.
10. Exposing predictions through a Flask REST API.
11. Automated API testing with pytest.
12. Docker containerization and cloud deployment (planned).

### Model and Training

The selected model uses XGBoost with a binary logistic objective.

The training script saves the model, fitted `DictVectorizer` and classification threshold together in:

`models/xgboost_model.bin`

This allows the prediction application to reuse the same fitted preprocessing and model without retraining.

The selected threshold is 0.5.

Detailed modelling experiments and evaluation are documented in `notebooks/02_modelling.ipynb`.

## Prediction API

A Flask application exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the API is running |
| `/predict` | POST | Accepts a booking as JSON and returns a cancellation prediction |

Example prediction response:

```json
{
  "cancellation_probability": 0.1132,
  "is_canceled": false,
  "threshold": 0.5
}
```

The probability represents the model's estimated cancellation risk. The Boolean prediction is determined by comparing that probability with the selected threshold.

## Repository Structure

```text
hotel-booking-cancellation-ml/
│
├── data/
│   ├── raw/
│   │   └── hotels.csv
│   └── processed/
│       └── hotels_clean.csv
│
├── models/
│   └── xgboost_model.bin
│
├── notebooks/
│   ├── 01_eda_cleaning.ipynb
│   └── 02_modelling.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── app.py
│   └── predict-test.py
│
├── tests/
│   └── test_app.py
│
├── test_data/
│   └── booking_01.json
│
├── README.md
├── requirements.txt
├── .gitignore
├── .dockerignore
└── Dockerfile
```

Docker-related files will be added when containerization is completed.

## Running the Project

The following instructions are intended for someone reviewing or reproducing the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/nabila-natasha/hotel-booking-cancellation-ml.git

cd hotel-booking-cancellation-ml
```

### 2. Create and activate a virtual environment

Python 3.10 or newer is recommended.

**Windows PowerShell**

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

**Linux / macOS / GitHub Codespaces**

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

The `.venv` directory is intentionally excluded from version control. Each user should create their own environment using the instructions above.

### 4. Run the notebooks

From the project root:

```bash
jupyter notebook
```

Run the notebooks in order:

1. `notebooks/01_eda_cleaning.ipynb` — performs data cleaning and EDA.
2. `notebooks/02_modelling.ipynb` — performs model development, evaluation and tuning.

The cleaned dataset is available under `data/processed/`, so the modelling notebook can also be run independently if that file is present.

### 5. Train the model

From the project root:

```bash
python src/train.py
```

This trains the model and saves the resulting artifact to:

```text
models/xgboost_model.bin
```

The prediction API uses this saved artifact and does not retrain the model when it starts.

### 6. Run the prediction API

Start the Flask application:

```bash
python src/app.py
```

The API runs locally on:

```text
http://localhost:9696
```

### 7. Check the API health endpoint

Open another terminal and run:

```bash
curl http://localhost:9696/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### 8. Submit a booking prediction

Using the example booking JSON file:

**Windows PowerShell**

```powershell
curl.exe -X POST http://localhost:9696/predict `
  -H "Content-Type: application/json" `
  --data-binary "@test_data/booking_01.json"
```

**Linux / macOS**

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  --data-binary "@test_data/booking_01.json"
```

The API returns the cancellation probability, predicted class and classification threshold.

### 9. Run automated tests

From the project root:

```bash
python -m pytest tests -v
```

The tests currently cover:

- The health endpoint response.
- The prediction endpoint response and basic output validation.

Current result: **2 tests passed**.

## Limitations

- The dataset contains historical bookings from two hotels in Portugal between 2015 and 2017. It may not reflect current demand or customer behaviour.
- Model performance on historical data does not guarantee performance on future bookings.
- The dataset does not contain a booking ID, so duplicates are identified using identical column values. Some identical records may represent separate real bookings.
- `total_of_special_requests` may be updated after the original booking is created. Its availability must be considered when defining the actual prediction point.
- Non-refundable bookings have a very high cancellation rate in this dataset. This relationship requires further investigation before operational interpretation.
- Model calibration, threshold selection and performance on new data would need further evaluation before real-world use.
- This is a portfolio and learning project, not a production-validated hotel decision system.

## Author

Developed as a machine learning portfolio project for the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), as part of the midterm project.
