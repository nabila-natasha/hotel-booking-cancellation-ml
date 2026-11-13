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

## Quick Start for Reviewers

| I want to... | Go to |
|---|---|
| Try the live app | https://hotel-cancellation-api-dmr5.onrender.com/ (free tier: the first load after inactivity can take about 1 minute) |
| Read the data analysis | `notebooks/01_eda_cleaning.ipynb` |
| Read the model comparison and tuning | `notebooks/02_modeling.ipynb` |
| Run the notebooks, training script, app and tests | [Reproducibility](#reproducibility-running-the-project) |
| Run the Docker container | [Docker Containerization](#docker-containerization) |

Fastest check, with no git or local install needed: open the repository in **GitHub Codespaces** (**Code > Codespaces > Create codespace on main**), then run this in the terminal (the install takes a few minutes):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -v          # expected: 2 passed
python src/app.py                  # then click "Open in Browser" for port 9696
```

Running on your own computer or in Google Colab, and running the notebooks, is explained in [Reproducibility](#reproducibility-running-the-project).

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
| `/booking-prediction` | POST | Receives the web form and shows the result. Opening it directly in the browser redirects to the form. |

The web interface allows users to enter hotel booking information and submit it to the trained model. The application displays the estimated cancellation probability and predicted booking status.

The JSON API remains available for programmatic predictions and integration with other applications.

## Dataset

- **Dataset:** Hotel Booking Demand.
- **Source:** [TidyTuesday, 11 February 2020](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md)
- **Original research:** Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41–49.
- **Hotels:** One city hotel and one resort hotel in Portugal.
- **Prediction target:** `is_canceled` (1 = cancelled, 0 = not cancelled).
- **Raw dataset:** 119,390 rows and 32 columns.
- **Cleaned dataset:** 84,392 rows and 27 columns.

| File | Description |
|---|---|
| `data/raw/hotels.csv` | Original dataset |
| `data/processed/hotels_clean.csv` | Cleaned dataset used for modelling |

Both files are committed to the repository, so no download is needed. If `data/raw/hotels.csv` is missing, the first notebook downloads it from the [TidyTuesday repository](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/hotels.csv); if your network blocks the download, save the file manually as `data/raw/hotels.csv`.

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

- 27.9% of the cleaned bookings were cancelled. Always predicting "not cancelled" gives 72.1% accuracy but finds no cancellations.
- **Hotel:** the city hotel has a higher cancellation rate (30.4%) than the resort hotel (23.9%).
- **Lead time:** cancellations rise steadily with lead time, from 8.6% for bookings made 0-7 days ahead to 40.9% for bookings made more than 180 days ahead.
- **Special requests:** bookings with no special requests are cancelled more often (34.0%) than bookings with one (22.7%) or two or more (20.4%).
- **Booking channel:** online travel agent bookings are cancelled most often (35.5%); direct (14.8%) and corporate (12.2%) bookings least.
- **Guest history:** repeat guests rarely cancel (8.3% vs 28.7%), but guests who cancelled before cancel again often (67.5% vs 27.1%).
- **Deposit type:** non-refundable bookings are cancelled 95.0% of the time (991 bookings). This is a surprising pattern and is discussed in the limitations.
- Leakage columns (for example `reservation_status`) and columns recorded after the booking were removed before modelling.

(See the EDA sections of `notebooks/01_eda_cleaning.ipynb` for the charts and tables behind these findings.)

## Machine Learning Workflow

The project follows these steps:

1. Data audit, cleaning and exploratory data analysis.
2. Stratified train, validation and test split (60/20/20).
3. Feature preprocessing using `DictVectorizer`.
4. Training and tuning four model types: logistic regression, decision tree, random forest and XGBoost.
5. Comparing the models on the validation set using ROC-AUC and selecting XGBoost.
6. XGBoost hyperparameter tuning using 5-fold cross-validation.
7. Final evaluation on the held-out test set using ROC-AUC, precision and recall.
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

### Model Comparison

Four model types were trained on the same training data and compared on the validation set. Each model was tuned over a small set of hyperparameters.

| Model | Tuned hyperparameters | Best setting | Validation ROC-AUC |
|---|---|---|---|
| **XGBoost** | `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree` (5-fold CV) | eta 0.1, max_depth 8, min_child_weight 1, subsample 0.8, colsample_bytree 0.8, 223 rounds | **0.8979** |
| Random forest | `n_estimators`, `max_depth` | 300 trees, max_depth 25 | 0.8768 |
| Decision tree | `max_depth`, `min_samples_leaf` | max_depth 10, min_samples_leaf 20 | 0.8728 |
| Logistic regression | `C` | C = 10 | 0.8061 |

XGBoost has the highest validation ROC-AUC, so it was selected as the final model. It was then retrained on the training and validation data and evaluated once on the test set.

### Results

| Measure | Value |
|---|---|
| Test ROC-AUC (notebook) | 0.9039 |
| Test ROC-AUC (`src/train.py`) | 0.9045 |
| Test PR-AUC | 0.7898 |
| Precision at threshold 0.5 | 0.743 |
| Recall at threshold 0.5 | 0.643 |
| F1 at threshold 0.5 | 0.689 |
| Accuracy at threshold 0.5 | 0.838 (always predicting "not cancelled" gives 0.721) |

All values in the first table come from `notebooks/02_modeling.ipynb` (final model trained on the training and validation data, evaluated once on the test set), except the `src/train.py` row.

The notebook (0.9039) and `src/train.py` (0.9045) differ slightly because the notebook uses `agent` as a number, while `train.py` treats it as a text category (it is an ID). The deployed model is the one trained by `src/train.py`. Retraining also gives slightly different values because XGBoost uses random subsampling.

### Threshold

The selected classification threshold is 0.5. At this threshold about 74% of the bookings flagged as "likely to cancel" are really cancelled (precision), and about 64% of all cancellations are found (recall). A lower threshold finds more cancellations but raises more false alarms (for example at 0.3: recall about 0.83, precision about 0.63). A hotel can change the threshold stored in the model artifact to match its own costs.

Detailed modelling experiments and evaluation are documented in `notebooks/02_modeling.ipynb`.

## Interactive Web Application

The Flask application provides a browser-based interface for entering hotel booking information.

The form follows a hotel reservation layout with four sections and a live reservation summary:

1. **Stay:** hotel, check-in, check-out and booking entry date.
2. **Guests:** adults, children, babies, country, customer type and guest history.
3. **Room and rate:** room type, meal plan, rate per night, deposit and special requests.
4. **Booking source:** market segment, agent, company booking and waiting-list days.

Some values are calculated automatically, so users do not enter them:

| Calculated value | How it is calculated |
|---|---|
| Number of nights, weekend nights and week nights | From check-in and check-out (a weekend night is a Saturday or Sunday night) |
| Lead time | Check-in date minus booking entry date |
| Repeat guest | Yes if the guest has any previous bookings or cancellations |
| Distribution channel | The usual channel for the chosen market segment in the training data |
| Estimated room total | Rate per night multiplied by the number of nights |
| Arrival year, month, week number and day | From the check-in date (done by the backend) |

The form keeps its values after a prediction and checks the dates before submitting.

### Lead Time Calculation

Lead time represents the number of days between the date a booking is entered into the property management system and the guest's arrival date.

The application accepts both dates and calculates lead time automatically:

`lead_time = arrival_date - booking_date`

The backend calculates this value again before passing the booking information to the model.

### Average Daily Rate (ADR)

The application shows ADR as a rate-per-night field.

ADR represents the average lodging revenue per occupied night. The dataset does not state the currency, so the interface shows no currency symbol.

### Prediction Result

After submitting the form, the application displays:

- Estimated cancellation probability, with a bar showing the decision threshold.
- Risk level (low or high).
- Classification threshold.

The predicted class is determined by comparing the estimated probability with the stored threshold.

## Prediction API

The Flask application exposes two API endpoints in addition to the interactive homepage.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the application is running |
| `/predict` | POST | Accepts booking information as JSON and returns a prediction |

Example request (the same booking as `test_data/booking_02.json`):

```json
{
    "hotel": "resort_hotel",
    "lead_time": 120,
    "arrival_date_year": 2025,
    "arrival_date_month": "july",
    "arrival_date_week_number": 28,
    "arrival_date_day_of_month": 12,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 6,
    "children": 2.0,
    "babies": 1,
    "meal": "bb",
    "country": "gbr",
    "market_segment": "online_ta",
    "distribution_channel": "ta/to",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "a",
    "deposit_type": "no_deposit",
    "agent": 9,
    "days_in_waiting_list": 0,
    "customer_type": "transient",
    "adr": 500.0,
    "total_of_special_requests": 0,
    "has_company": 1
}
```

Example response for this request (the exact probability can differ slightly if the model is retrained):

```json
{
    "cancellation_probability": 0.5058664679527283,
    "is_canceled": true,
    "threshold": 0.5
}
```

The probability represents the model's estimated cancellation risk. The Boolean prediction is determined by comparing the probability with the selected threshold.

## Project Files

| File | Purpose |
|---|---|
| `notebooks/01_eda_cleaning.ipynb` | Data audit, cleaning decisions, EDA; writes `data/processed/hotels_clean.csv` |
| `notebooks/02_modeling.ipynb` | Train/validation/test split, model comparison, tuning, final evaluation |
| `src/train.py` | The final training steps from notebook 02 as a script: same split (`random_state=42`), same preprocessing and XGBoost parameters; saves `models/xgboost_model.bin` |
| `src/predict.py` | Loads the saved artifact and predicts for one booking |
| `src/predict-test.py` | Sends an example booking through the prediction code |
| `src/app.py` | Flask app: web form, `/predict` API and `/health` |
| `tests/test_app.py` | pytest tests for `/health` and `/predict` |
| `requirements.txt` | All packages for notebooks, training, app and tests (exact versions) |
| `requirements-api.txt` | Only the packages needed to run the app; used by the Dockerfile to keep the image small |
| `Dockerfile` | Builds the container image used locally and on Render |

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
│   └── 02_modeling.ipynb
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
│   ├── booking_01.json
│   └── booking_02.json
│
├── screenshots/
│   ├── 01-homepage.png
│   ├── 02-render-deployment.PNG
│   ├── 03-postman-prediction.PNG
│   ├── 04-health-check.PNG
│   └── 05-docker-postman-prediction.PNG
│
├── README.md
├── requirements.txt
├── requirements-api.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Reproducibility: Running the Project

This section explains how to run the notebooks, the training script, the web application, the tests and Docker. **You do not need git or a local Python installation:** the easiest way is GitHub Codespaces, which runs everything in your browser.

The cleaned and raw datasets are **committed to the repository** (`data/raw/hotels.csv`, 16.9 MB, and `data/processed/hotels_clean.csv`, 9.6 MB), so nothing has to be downloaded before running the notebooks.

### Where to run it

| Option | Best when | What you need |
|---|---|---|
| **A. GitHub Codespaces (recommended)** | You cannot install git, Python or Docker on your computer | A free GitHub account and a browser. Python and Docker are already available there |
| **B. Your own computer** | You can install Python 3.12 or newer | Python 3.12+; git is optional (you can download a ZIP file instead) |
| **C. Google Colab** | You only want to run the two notebooks | A Google account |

**Codespaces usage limit:** GitHub Free personal accounts include 120 compute hours and 15 GB-month of storage per month ([GitHub documentation](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces)). A 2-core codespace uses 2 of those hours for every hour it runs, so you get roughly 60 hours. Running the whole project takes well under 1 hour. Stop or delete the codespace when you finish.

**Python version:** the pinned packages in `requirements.txt` (`numpy`, `scipy` and `xgboost`) need **Python 3.12 or newer**. This project was developed and tested with Python 3.14.2 in GitHub Codespaces (the Docker image also uses Python 3.14) and the tests also pass on Python 3.13.

---

### Option A: GitHub Codespaces (no git needed)

1. Sign in to GitHub and open the repository page: https://github.com/nabila-natasha/hotel-booking-cancellation-ml
2. Click the green **Code** button, open the **Codespaces** tab and click **Create codespace on main**. After 1 to 2 minutes, VS Code opens in your browser with all the project files already loaded. (No `git clone` is needed.)
3. Open a terminal: **Terminal > New Terminal**. Check the Python version:

   ```bash
   python --version
   ```

   It must show 3.12 or newer.
4. Create a virtual environment and install the packages (this takes a few minutes):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If VS Code asks whether to use the new environment for this workspace, click **Yes**.
5. Continue with [Step 1: Run the notebooks](#step-1-run-the-notebooks).

> The other steps (notebooks, training, app, tests and Docker) all work in the Codespaces terminal. When the Flask app starts, Codespaces shows a pop-up for port 9696: click **Open in Browser**, or use the **Ports** tab.

### Option B: Your own computer (VS Code or a terminal)

1. **Get the code.**
   - *With git:* `git clone https://github.com/nabila-natasha/hotel-booking-cancellation-ml.git` and then `cd hotel-booking-cancellation-ml`.
   - *Without git:* open the repository page, click **Code > Download ZIP**, unzip it, and open the unzipped folder (named `hotel-booking-cancellation-ml-main`) in VS Code with **File > Open Folder**.
2. **Check Python 3.12 or newer** is installed: `python --version` (on Windows you can also try `py --version`). If it is missing, install it from https://www.python.org/downloads/.
3. **Create a virtual environment** in a terminal opened in the project folder (in VS Code: **Terminal > New Terminal**):

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell says "running scripts is disabled", either use Command Prompt (`.\.venv\Scripts\activate.bat`) or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then activate again.

   **Linux / macOS**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. **Install the packages:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For the prediction application only (without notebooks) you can use `pip install -r requirements-api.txt` instead. `requirements-api.txt` is also the file used by the Dockerfile.
5. If `pip` cannot reach the internet (for example behind a company proxy), use Option A instead.
6. The `.venv` folder is excluded from version control. Each user creates their own.

### Option C: Google Colab (notebooks only)

1. Open the notebook directly from GitHub in Colab:
   - [01_eda_cleaning.ipynb](https://colab.research.google.com/github/nabila-natasha/hotel-booking-cancellation-ml/blob/main/notebooks/01_eda_cleaning.ipynb)
   - [02_modeling.ipynb](https://colab.research.google.com/github/nabila-natasha/hotel-booking-cancellation-ml/blob/main/notebooks/02_modeling.ipynb)
2. In the first code cell, remove the `#` in front of these two lines and run the cell:

   ```python
   !git clone https://github.com/nabila-natasha/hotel-booking-cancellation-ml.git
   %cd hotel-booking-cancellation-ml/notebooks
   ```

   Do this only in Colab. In Codespaces or on your own computer the project files are already there.
3. Colab already has pandas, scikit-learn, XGBoost and seaborn. If a package is missing, run `%pip install xgboost` in a new cell.
4. Colab shows text columns with the dtype `object`, while the pinned pandas version shows `str`. The notebooks check for non-numeric columns instead of a specific dtype, so they work with both. Running notebook 01 with pandas 2.2 (which shows `object`) and with the pinned pandas 3.1 produced an identical `hotels_clean.csv`.

---

### Step 1: Run the notebooks

Run the notebooks in order. The notebooks use relative paths, so Jupyter must start in the project folder or in the `notebooks/` folder (this is the default in VS Code and Jupyter).

1. Open `notebooks/01_eda_cleaning.ipynb`, click **Select Kernel** (top right), choose **Python Environments > .venv**, and click **Run All**.
   - Takes about 15 to 30 seconds.
   - It writes `data/processed/hotels_clean.csv`. The final checks in the notebook confirm the cleaned data has 84,392 rows and 27 columns, and the last cell prints `Saved shape: (84392, 27)`.
   - If `data/raw/hotels.csv` is missing, the notebook downloads it from the [TidyTuesday repository](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md). If your network blocks that download, save the file manually as `data/raw/hotels.csv`.
2. Open `notebooks/02_modeling.ipynb`, select the same `.venv` kernel and click **Run All**.
   - Takes about 10 to 25 minutes depending on the machine (cross-validation and the random forest are the slow parts).
   - It ends with the model comparison table and the final results. You should see a validation ROC-AUC of about 0.90 for XGBoost (0.8979 in the committed run) and a test ROC-AUC of about 0.90 (0.9039 in the committed run). Small differences in the third or fourth decimal are normal.

**Run both notebooks from the terminal instead (optional).** With the virtual environment activated:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_eda_cleaning.ipynb --output-dir ../_check
jupyter nbconvert --to notebook --execute 02_modeling.ipynb --output-dir ../_check
cd ..
```

This runs every cell and stops with an error if any cell fails. The executed copies are saved in `_check/` and your original notebooks are not changed. Delete the `_check/` folder when you are done.

### Step 2: (Optional) Train the model

The repository includes the previously trained model at `models/xgboost_model.bin`, so retraining is not required to run the application.

To retrain the model, run:

```bash
python src/train.py
```

This script reads `data/processed/hotels_clean.csv`, trains the XGBoost model, prints the test AUC, and saves the updated model to:

```text
models/xgboost_model.bin
```

**Note:** Retraining overwrites the existing model `models/xgboost_model.bin`. XGBoost uses random subsampling, so a retrained model can give slightly different probabilities (in the later decimals) from the numbers shown in this README, and a booking close to the 0.5 threshold could change class.

To restore the original model tracked in Git, run:

```bash
git restore --source=HEAD --staged --worktree models/xgboost_model.bin
```

(If you downloaded the repository as a ZIP file instead of using git, download a fresh copy to restore the original model.)

### Step 3: Run the Flask application

From the project root:

```bash
python src/app.py
```

Open the application in your browser:

http://127.0.0.1:9696/

In Codespaces, click **Open in Browser** in the port 9696 pop-up instead. The application provides the interactive booking form and prediction result.

### Step 4: Check the health endpoint

Open another terminal (in Codespaces: **Terminal > New Terminal**; activate the environment again if needed) and run:

```bash
curl http://127.0.0.1:9696/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Step 5: Submit a booking prediction through the API

Using the example booking JSON file:

**Windows PowerShell**

```powershell
curl.exe -X POST http://127.0.0.1:9696/predict `
  -H "Content-Type: application/json" `
  --data-binary "@test_data/booking_02.json"
```

**Linux / macOS / GitHub Codespaces**

```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  --data-binary "@test_data/booking_02.json"
```

The API returns the cancellation probability, predicted class and classification threshold. With the saved model in this repository, `booking_02.json` returns:

```json
{
  "cancellation_probability": 0.5058664679527283,
  "is_canceled": true,
  "threshold": 0.5
}
```

### Step 6: Run automated tests

From the project root:

```bash
python -m pytest tests -v
```

Expected result: `2 passed`.

The tests cover:

- Health endpoint response.
- Prediction endpoint response and basic output validation.

## Docker Containerization

The application is packaged using Docker to provide a reproducible runtime environment. Docker is already installed in GitHub Codespaces; on your own computer you need Docker Desktop or Docker Engine (run `docker --version` to check). The notebooks, training script, tests and app do not need Docker.

The `Dockerfile` does the following:

1. Starts from the `python:3.14-slim` base image.
2. Installs the packages in `requirements-api.txt`.
3. Copies `src/` (the app) and `models/` (the trained model) into the image.
4. Starts the app with Gunicorn on the port given by the `PORT` environment variable (Render sets it), or 9696 when it is not set.

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

### 4. Test the Dockerized API using Postman

The Dockerized application was built and tested in GitHub Codespaces (Docker 29.9.2). The running container was accessed through the forwarded port `9696` and tested externally using Postman.

**Codespaces port visibility**

By default, a forwarded Codespaces port may have Private visibility. To allow Postman or another external client to access the Docker API:

1. Open the **Ports** tab in GitHub Codespaces.
2. Find port `9696`.
3. Right-click the port and select **Port Visibility > Public**.
4. Copy the forwarded address shown for port `9696`.

**Note:** Public visibility allows anyone with the forwarded URL to access the application while the Codespace and container are running. Use this setting only when external access is needed for testing.

**Postman setup note**

If you are using Postman Web and the **Send** button is disabled, check the selected Postman Agent. If the Desktop Agent is selected but not installed, download and install the Postman Desktop Agent, then open it before sending the request.

**Request configuration:**

| Setting | Value |
|---|---|
| Method | POST |
| Endpoint | `https://<your-codespaces-forwarded-url>/predict` |
| Header | `Content-Type: application/json` |
| Body | Raw JSON booking information |

Replace `<your-codespaces-forwarded-url>` with the current forwarded address for port `9696` shown in the Codespaces Ports tab. The generated address may change if the Codespace is recreated.

Before submitting the prediction request, you can check the application health using:

`https://<your-codespaces-forwarded-url>/health`

The expected response is:

```json
{
  "status": "ok"
}
```

The same booking payload was also tested using the local Python prediction script:

```bash
python src/predict-test.py
```

The Docker API returned the following response:

```json
{
  "cancellation_probability": 0.5058664679527283,
  "is_canceled": true,
  "threshold": 0.5
}
```

The prediction probability, classification result and threshold matched the local Python prediction exactly. This confirmed consistent inference between the local Python environment and the Docker container.

## Cloud Deployment: Render

The Dockerized Flask application is deployed as a Render Web Service.

**Deployment platform:** Render  
**Service name:** `hotel-cancellation-api`  
**Public application URL:** https://hotel-cancellation-api-dmr5.onrender.com/

| Render setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Docker (uses the repository `Dockerfile`) |
| Branch | `main` |
| Instance type | Free |
| Start command | From the Dockerfile (`gunicorn`) |

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

## Model Prediction Consistency Validation

The application was tested with local Python execution, Docker containerization and the Render deployment to verify that the same booking information produces
consistent predictions across environments.

### Test Scenario

The same 26-feature hotel booking payload (`test_data/booking_02.json`) was used for all three tests.

The test booking produced the following result:

| Metric | Result |
|---|---|
| Cancellation probability | 0.5058664679527283 |
| Predicted cancellation status | True |
| Classification threshold | 0.5 |

### Validation Environments

| Environment | Test Method | Result |
|---|---|---|
| Local Python | `python src/predict-test.py` | Passed |
| Docker | Postman POST request to `/predict` | Passed |
| Render | Postman POST request to `/predict` (screenshot 3) | Passed |

The local Python, Docker and Render predictions returned identical cancellation probabilities, classification results and thresholds. All three use the same saved model file, so identical results are expected.


## Screenshots and Deployment Evidence

The following screenshots demonstrate the interactive web application,
Docker API testing, successful cloud deployment, prediction results
and application health check.

### 1. Interactive Web Application

The homepage provides an interactive form for entering hotel booking information.

![Hotel Booking Prediction Homepage](screenshots/01-homepage.png)

### 2. Render Cloud Deployment

The Render dashboard confirms that the application has been successfully deployed.

![Render Deployment](screenshots/02-render-deployment.PNG)

### 3. Prediction API

A successful Postman request to the live Render URL (`/predict`) demonstrates that the deployed API accepts booking information and returns a cancellation probability (0.5058664679527283) and predicted status.

![Postman Prediction Result](screenshots/03-postman-prediction.PNG)

### 4. Application Health Check

The health endpoint confirms that the deployed application is responding successfully.

![Application Health Check](screenshots/04-health-check.PNG)

### 5. Docker API Prediction Test

A successful Postman POST request to the Dockerized application, running through the forwarded GitHub Codespaces port, demonstrates that the container can process booking information and return a prediction.

The prediction was also compared with the local Python execution result. Both environments returned the same cancellation probability and predicted classification.

![Docker Postman Prediction](screenshots/05-docker-postman-prediction.PNG)


**Live application:** https://hotel-cancellation-api-dmr5.onrender.com/


## Limitations

- The dataset contains historical bookings from two hotels in Portugal between 2015 and 2017. It may not reflect current demand or customer behaviour.
- The model uses the arrival year as a feature. It only saw the years 2015-2017, so predictions for arrival dates in later years are extrapolations and should be treated as indicative only.
- The country list in the web form only includes countries with at least 100 bookings in the training data. Other countries can be entered as "Other / not listed".
- Model performance on historical data does not guarantee performance on future bookings.
- The dataset does not contain a booking ID, so duplicates are identified using identical column values. Some identical records may represent separate real bookings.
- `total_of_special_requests` may be updated after the original booking is created. Its availability must be considered when defining the actual prediction point.
- Non-refundable bookings have a very high cancellation rate in this dataset. This relationship requires further investigation before operational interpretation.
- Model calibration, threshold selection and performance on new data would need further evaluation before real-world use.
- The application is a portfolio and learning project, not a production-validated hotel decision system.

## Author

Developed as a machine learning portfolio project for the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), as part of the midterm project.
