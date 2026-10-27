# Hotel Booking Cancellation Prediction

## Project Overview

This project develops a machine learning classification system to estimate the probability that a hotel booking will be cancelled.

The goal is to demonstrate an end-to-end machine learning workflow, from exploratory data analysis and data preparation to model evaluation, a prediction API, containerization, and deployment.

## Business Problem

Hotel cancellations can affect room availability, revenue planning, and operational decisions. A cancellation-risk model could help hotel revenue-management teams identify bookings that may require further attention.

**Intended use:** Support booking-risk assessment and operational planning. Predictions are estimates, not guarantees, and should not be used as the sole basis for decisions affecting customers.

## Dataset

- **Dataset:** Hotel Booking Demand
- **Source:** [TidyTuesday — 11 February 2020](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md)
- **Original research:** Antonio, Almeida, and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41–49.
- **Prediction target:** `is_canceled` — 1 indicates a cancelled booking; 0 indicates a booking that was not cancelled.

The dataset contains historical booking records for city and resort hotels. Its source documentation describes the variables and their meanings.

Dataset redistribution and licensing terms will be reviewed before the raw data is committed to this repository.

## Prediction Design

The intended prediction point is booking creation. Features will be reviewed to ensure they would be available at that time.

Outcome-related or later-known information, including `reservation_status` and `reservation_status_date`, will be excluded to prevent target leakage. Other fields will be reviewed for timing-related leakage before model training.

## Machine Learning Workflow

1. Exploratory data analysis and data-quality checks.
2. Target distribution and feature analysis.
3. Train, validation, and test split.
4. Preprocessing using a reproducible pipeline.
5. Baseline classification with a dummy model.
6. Comparison of logistic regression and tree-based models.
7. Model selection and hyperparameter tuning.
8. Evaluation using ROC-AUC, precision, recall, and a confusion matrix.
9. Saving the fitted preprocessing-and-model pipeline.
10. Serving predictions through an API.
11. Testing and packaging the application with Docker.
12. Deployment attempt and reproducibility checks.

Final model results will be added after experiments are completed.

## Repository Structure

```text
hotel-booking-cancellation-ml/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── models/
├── tests/
└── Dockerfile
```

Some directories and files will be added as the project develops.

## Running the Project

Setup, training, testing, API, and Docker instructions will be added once the implementation is tested.

## Limitations

- The dataset represents historical bookings and may not reflect current hotel demand or customer behavior.
- Predictive performance on historical data does not guarantee performance on future bookings.
- Feature availability at booking creation must be verified to avoid leakage.
- Model quality, calibration, and the appropriate decision threshold will be evaluated before proposing operational use.

## Author

Developed as a machine learning portfolio and learning project.
