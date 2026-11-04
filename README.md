# Hotel Booking Cancellation Prediction

## Project Overview

This project develops an end-to-end machine learning application that estimates the probability of a hotel booking being cancelled.

The project demonstrates a complete machine learning workflow, including exploratory data analysis (EDA), data cleaning, feature engineering, model development, hyperparameter tuning, model evaluation, prediction, API development, web application development, automated testing, Docker containerization and cloud deployment.

**Project status**

| Component | Status |
|---|---|
| Data audit, cleaning and EDA | Completed |
| Feature engineering and model development | Completed |
| Hyperparameter tuning and model evaluation | Completed |
| Saved XGBoost model artifact | Completed |
| Flask prediction API | Completed |
| Interactive web application | Completed |
| Automated API testing | Completed |
| Docker containerization | Completed |
| Cloud deployment using Render | Completed |

## Business Problem

Hotel cancellations affect room availability, revenue planning and operational decisions.

A cancellation-risk model could help a hotel's revenue-management team identify bookings that may require attention. Potential actions include sending booking reminders, reviewing deposit policies or planning room availability more carefully.

**Project objective:** Develop a machine learning model that estimates cancellation probability using hotel booking information available to the prediction application.

The application provides:
- An estimated cancellation probability.
- A predicted cancellation status.
- A configurable classification threshold stored with the model artifact.

Predictions are estimates, not guarantees, and should not be the sole basis for decisions affecting customers.

## Live Application

The application is deployed on Render and can be accessed publicly.

**Live web application:**  
https://hotel-cancellation-api-dmr5.onrender.com/

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive hotel booking prediction form |
| `/health` | GET | Application health check |
| `/predict` | POST | JSON-based cancellation prediction |

The web interface allows users to enter hotel booking information and submit it to the trained model. The application displays the estimated cancellation probability and predicted booking status.

The JSON API remains available for programmatic predictions and integration with other applications.

## Dataset

- **Dataset:** Hotel Booking Demand.
- **Source:** TidyTuesday, 11 February 2020.
- **Original research:** Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41–49.
- **Hotels:** One city hotel and one resort hotel in Portugal.
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
| Duplicate rows | Approximately 34,000 exact duplicates after removing leakage and post-booking columns | Removed duplicates, keeping the first occurrence |
| Missing `company` values | Approximately 94% missing | Created `has_company` as a binary feature and dropped the original column |
| Missing `agent` values | Approximately 14% missing | Filled with `0` and treated as a categorical feature |
| Missing `country` and `children` values | Small number of missing values | Filled with `"unknown"` and `0`, respectively |
| Invalid or extreme values | Invalid guest counts, zero-night stays, negative prices or extreme values | Removed affected records |

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
10. Developing a Flask REST API.
11. Building an interactive Flask web interface.
12. Automated API testing with pytest.
13. Docker containerization.
14. Cloud deployment using Render.

### Model and Training

The selected model uses XGBoost with a binary logistic objective.

The training script saves the following objects together:

- Trained XGBoost model.
- Fitted `DictVectorizer`.
- Classification threshold.

The saved artifact is located at:

`models/xgboost_model.bin`

The prediction application loads this artifact at startup. It does not retrain the model for each prediction or when the application starts.

The selected classification threshold is 0.5.

Detailed modelling experiments and evaluation are documented in `notebooks/02_modelling.ipynb`.

## Interactive Web Application

The Flask application provides a browser-based interface for entering hotel booking information.

The form is organised into sections:

- Hotel and arrival details.
- Guests and stay information.
- Booking information.
- Reservation history.
- Room and payment details.

### Lead Time Calculation

Lead time represents the number of days between the date a booking is entered into the property management system and the guest's arrival date.

The application accepts both dates and calculates lead time automatically:

`lead_time = arrival_date - booking_date`

The backend independently calculates this value before passing the booking information to the model.

### Average Daily Rate (ADR)

The application displays ADR as a price-per-night field.

ADR represents the average lodging revenue per occupied night. The dollar symbol in the interface is a display convention; it should not be interpreted as independently verified currency metadata from the source dataset.

### Prediction Result

After submitting the form, the application displays:

- Estimated cancellation probability.
- Predicted booking status.
- Classification threshold.

The predicted class is determined by comparing the estimated probability with the stored threshold.

## Prediction API

The Flask application exposes two API endpoints in addition to the interactive homepage.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the application is running |
| `/predict` | POST | Accepts booking information as JSON and returns a prediction |

Example request:

```json
{
  "hotel": "city_hotel",
  "lead_time": 30,
  "arrival_date_year": 2017,
  "arrival_date_month": "july",
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 15,
  "stays_in_weekend_nights": 2,
  "stays_in_week_nights": 3,
  "adults": 2,
  "children": 0,
  "babies": 0,
  "meal": "bb",
  "country": "prt",
  "market_segment": "online_ta",
  "distribution_channel": "ta_to",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "reserved_room_type": "a",
  "deposit_type": "no_deposit",
  "agent": "9",
  "days_in_waiting_list": 0,
  "customer_type": "transient",
  "adr": 100.0,
  "total_of_special_requests": 1,
  "has_company": 0
}
```

Example response:

```json
{
  "cancellation_probability": 0.5711,
  "is_canceled": true,
  "threshold": 0.5
}
```

The probability represents the model's estimated cancellation risk. The Boolean prediction is determined by comparing the probability with the selected threshold.

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
│   ├── predict-test.py
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── tests/
│   └── test_app.py
│
├── test_data/
│   └── booking_01.json
|   └── booking_02.json
|
├── screenshots/
│   ├── 01-homepage.png
│   ├── 02-render-deployment.PNG
│   ├── 03-postman-prediction.PNG
│   └── 04-health-check.PNG
│
├── README.md
├── requirements.txt
├── requirements-api.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Reproducibility: Running the Project Locally

The following instructions allow another developer or reviewer to reproduce the application locally.

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

For the full project, including notebooks and development dependencies:

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

For the prediction application only:

```bash
pip install -r requirements-api.txt
```

The `.venv` directory is excluded from version control. Each user should create their own environment.

### 4. Run the notebooks

From the project root:

```bash
jupyter notebook
```

Run the notebooks in order:

1. `notebooks/01_eda_cleaning.ipynb` — data cleaning and EDA.
2. `notebooks/02_modelling.ipynb` — model development, evaluation and tuning.

The cleaned dataset is available under `data/processed/`, so the modelling notebook can also be run independently if that file is present.

### 5. Train the model (optional)

If you want to reproduce model training:

```bash
python src/train.py
```

This trains the model and saves the resulting artifact to:

```text
models/xgboost_model.bin
```

The existing saved artifact can be used to run the application without retraining, provided it is present in the repository.

### 6. Run the Flask application

From the project root:

```bash
python src/app.py
```

Open the application in your browser:

http://127.0.0.1:9696/

The application provides the interactive booking form and prediction result.

### 7. Check the health endpoint

Open another terminal and run:

```bash
curl http://127.0.0.1:9696/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### 8. Submit a booking prediction through the API

Using the example booking JSON file:

**Windows PowerShell**

```powershell
curl.exe -X POST http://127.0.0.1:9696/predict `
  -H "Content-Type: application/json" `
  --data-binary "@test_data/booking_01.json"
```

**Linux / macOS / GitHub Codespaces**

```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  --data-binary "@test_data/booking_01.json"
```

The API returns the cancellation probability, predicted class and classification threshold.

### 9. Run automated tests

From the project root:

```bash
python -m pytest tests -v
```

The tests cover:

- Health endpoint response.
- Prediction endpoint response and basic output validation.

## Docker Containerization

The application is packaged using Docker to provide a reproducible runtime environment.

The Docker image includes:

- Python runtime.
- Application dependencies.
- Flask application.
- HTML templates and CSS.
- Saved XGBoost model artifact.

### 1. Build the Docker image

From the project root:

```bash
docker build -t hotel-cancellation-app:latest .
```

### 2. Run the Docker container

```bash
docker run --rm -p 9696:9696 hotel-cancellation-app:latest
```

The application runs inside the container using Gunicorn as its production WSGI server.

### 3. Access the application

Open:

http://localhost:9696/

For GitHub Codespaces, open the forwarded port `9696` from the Ports tab and use the generated browser URL instead of `127.0.0.1`.

### 4. Test the container

Check:

- Homepage loads successfully.
- CSS and HTML are served correctly.
- `/health` returns a successful response.
- Booking form submits successfully.
- Prediction result is displayed.

## Cloud Deployment: Render

The Dockerized Flask application is deployed as a Render Web Service.

**Deployment platform:** Render  
**Service name:** `hotel-cancellation-api`  
**Public application URL:** https://hotel-cancellation-api-dmr5.onrender.com/

### Deployment workflow

1. Commit and push application changes to the GitHub `main` branch.
2. Render retrieves the latest repository version.
3. Render builds the application using the repository Dockerfile.
4. Gunicorn starts the Flask application inside the container.
5. The public application becomes accessible through the Render URL.

### Verify the deployed application

**Homepage**

https://hotel-cancellation-api-dmr5.onrender.com/

**Health check**

https://hotel-cancellation-api-dmr5.onrender.com/health

The health endpoint should return:

```json
{
  "status": "ok"
}
```

**Prediction**

Submit a booking through the public web interface and verify that the estimated cancellation probability and predicted status are displayed.

Render free services may spin down after periods of inactivity, so the first request after inactivity can take longer to respond.

## Screenshots and Deployment Evidence

The following screenshots demonstrate the deployed application, successful model prediction and application health check.

### 1. Interactive Web Application

The homepage provides an interactive form for entering hotel booking information.

![Hotel Booking Prediction Homepage](screenshots/01-homepage.png)

### 2. Render Cloud Deployment

The Render dashboard confirms that the application has been successfully deployed.

![Render Deployment](screenshots/02-render-deployment.PNG)

### 3. Prediction API

A successful Postman request demonstrates that the deployed API accepts booking information and returns a cancellation probability and predicted status.

![Postman Prediction Result](screenshots/03-postman-prediction.PNG)

### 4. Application Health Check

The health endpoint confirms that the deployed application is responding successfully.

![Application Health Check](screenshots/04-health-check.PNG)

**Live application:** https://hotel-cancellation-api-dmr5.onrender.com/


## Limitations

- The dataset contains historical bookings from two hotels in Portugal between 2015 and 2017. It may not reflect current demand or customer behaviour.
- Model performance on historical data does not guarantee performance on future bookings.
- The dataset does not contain a booking ID, so duplicates are identified using identical column values. Some identical records may represent separate real bookings.
- `total_of_special_requests` may be updated after the original booking is created. Its availability must be considered when defining the actual prediction point.
- Non-refundable bookings have a very high cancellation rate in this dataset. This relationship requires further investigation before operational interpretation.
- Model calibration, threshold selection and performance on new data would need further evaluation before real-world use.
- The application is a portfolio and learning project, not a production-validated hotel decision system.

## Author

Developed as a machine learning portfolio project for the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), as part of the midterm project.
