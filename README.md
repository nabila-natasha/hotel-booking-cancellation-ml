# Hotel Booking Cancellation Prediction

## Project Overview

This project builds a machine learning classifier that estimates the probability that a hotel booking will be cancelled.

The goal is to demonstrate an end-to-end machine learning workflow: exploratory data analysis, data cleaning, model comparison and tuning, a prediction API, containerization, and deployment.

**Project status**

| Stage | Status |
|---|---|
| Data audit, cleaning and EDA (`01_eda_cleaning.ipynb`) | Done |
| Modelling and tuning (`02_modelling.ipynb`) | In progress |
| Training script, prediction API | Not started |
| Docker and cloud deployment | Not started |

Sections marked "to be added" below will be completed as each stage is finished.

## Business Problem

Hotel cancellations affect room availability, revenue planning and operational decisions. A cancellation-risk model could help a hotel's revenue-management team identify bookings that may need attention, for example by requiring a deposit, sending a reminder, or planning overbooking more carefully.

**Intended use:** support booking-risk assessment and planning. Predictions are estimates, not guarantees, and should not be the only basis for decisions that affect customers.

**Prediction point:** the model predicts cancellation risk **at the moment a booking is created**, so it only uses information the hotel would know at that time.

## Dataset

- **Dataset:** Hotel Booking Demand (city and resort hotel in Portugal, bookings arriving 2015-2017)
- **Source:** [TidyTuesday, 11 February 2020](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md)
- **Original research:** Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41-49.
- **Prediction target:** `is_canceled` (1 = cancelled, 0 = not cancelled)
- **Size:** 119,390 rows and 32 columns in the raw file; 84,392 rows and 27 columns after cleaning.

| File | Description |
|---|---|
| `data/raw/hotels.csv` | Original file, unchanged |
| `data/processed/hotels_clean.csv` | Output of `01_eda_cleaning.ipynb`, the input to `02_modelling.ipynb` |

If `data/raw/hotels.csv` is missing, the first notebook downloads it from the TidyTuesday repository.

## Data Cleaning Decisions

Each decision is explained with evidence in `01_eda_cleaning.ipynb` (sections 5 to 7).

| Issue | What I found | Decision |
|---|---|---|
| Outcome leakage | `reservation_status` matches the target exactly; `reservation_status_date` is recorded after the outcome | Dropped both |
| Information known only after booking | `booking_changes` and `assigned_room_type` are set later; `required_car_parking_spaces` had 0 cancellations in 7,416 raw bookings that had parking | Dropped all three |
| Duplicate rows | About 34,000 exact duplicates (about 28%) after removing the columns above | Dropped, keeping the first copy, so identical rows cannot end up in both train and test |
| Missing `company` (94%) | Missing most likely means "no company" | Created `has_company` (0/1), dropped the original column |
| Missing `agent` (14%) | Mostly direct and corporate bookings, likely "no agent" | Filled with `0`, treated as a category, not a number |
| Missing `country` (0.5%) and `children` (4 rows) | Small | Filled with `"unknown"` and `0` |
| Invalid or extreme values | About 760 rows with zero guests, zero nights, negative or extreme price (above 1000), or extreme guest counts | Dropped |

## Exploratory Data Analysis: Takeaways

- 27.9% of cleaned bookings were cancelled, so a model that always predicts "not cancelled" would be right 72% of the time without catching any cancellation. Models are therefore judged with ROC-AUC, precision and recall, not accuracy alone.
- Cancellation varies most with lead time (days between booking and arrival), booking channel (online travel agencies have the highest rate), country, and the number of special requests.
- Whether these patterns help predict cancellations is tested in the modelling notebook.

## Machine Learning Workflow

1. Data audit, cleaning and EDA (done).
2. Train, validation and test split (60/20/20, fixed random seed).
3. Preprocessing in a reproducible scikit-learn pipeline.
4. Baseline with a dummy classifier.
5. Comparison of logistic regression, decision tree and random forest.
6. Hyperparameter tuning and model selection.
7. Final evaluation on the test set using ROC-AUC, precision, recall and a confusion matrix.
8. Saving the fitted pipeline.
9. Serving predictions through a web API.
10. Containerizing the service with Docker.
11. Cloud deployment.

**Model results: to be added after the experiments are finished.**

## Repository Structure

```text
hotel-booking-cancellation-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── hotels.csv
│   └── processed/
│       └── hotels_clean.csv
├── notebooks/
│   ├── 01_eda_cleaning.ipynb
│   └── 02_modelling.ipynb        (in progress)
├── src/                          (to be added)
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── models/                       (to be added)
├── tests/                        (to be added)
└── Dockerfile                    (to be added)
```

## Running the Project

### 1. Set up the environment

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/<your-username>/hotel-booking-cancellation-ml.git
cd hotel-booking-cancellation-ml

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS / GitHub Codespaces
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the notebooks

```bash
cd notebooks
jupyter notebook
```

Run the notebooks **in order**, from the `notebooks/` folder, with *Restart and Run All*:

1. `01_eda_cleaning.ipynb` reads `data/raw/hotels.csv` and writes `data/processed/hotels_clean.csv`.
2. `02_modelling.ipynb` reads `data/processed/hotels_clean.csv`. The cleaned file is committed, so this notebook also runs on its own.

### 3. Training, API and Docker

To be added once the implementation is tested.

## Limitations

- The data are historical bookings from two hotels in Portugal (2015-2017) and may not reflect current demand or customer behaviour. Seasonal patterns come from about two years of data.
- Performance on historical data does not guarantee performance on future bookings.
- There is no booking ID, so duplicates are identified by identical column values. Some may be separate real bookings, for example several identical rooms booked together.
- `total_of_special_requests` may be updated after booking, so it may not be fully available at booking creation.
- Non-refundable bookings are cancelled 95% of the time, a pattern this data does not explain.
- Model quality, calibration and the decision threshold will be evaluated before suggesting any operational use.

## Author

Developed as a machine learning portfolio and learning project for the DataTalks.Club ML Zoomcamp (midterm project).
