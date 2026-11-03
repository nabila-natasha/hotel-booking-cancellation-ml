#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# Parameters

PROJECT_DIR = Path(__file__).resolve().parents[1]

input_file = PROJECT_DIR / "data" / "processed" / "hotels_clean.csv"

model_dir = PROJECT_DIR / "models"
model_dir.mkdir(exist_ok=True)

output_file = model_dir/ "xgboost_model.bin"

threshold = 0.5
num_boost_round = 223

xgb_params = {
    "eta": 0.1,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 42,
    "verbosity": 1
}


# Data Preparation

print("\n=== Loading Hotel Booking Dataset ===\n")

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# Agent is an identifier, so treat it as a categorical
df['agent'] = df['agent'].astype(str)

# Separate training and test datasets

df_full_train, df_test = train_test_split(
    df, 
    test_size=0.2,
    random_state=42, 
    stratify=df['is_canceled']
)

X_full_train = df_full_train.drop(columns=['is_canceled'])
y_full_train = df_full_train['is_canceled'].values

X_test = df_test.drop(columns=['is_canceled'])
y_test = df_test['is_canceled'].values


# Training

def train(df_train, y_train):

    dicts = df_train.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    features = list(dv.get_feature_names_out())

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=features
    )

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round
    )

    return dv, model


def predict(df, dv, model):

    dicts = df.to_dict(orient="records")    

    X = dv.transform(dicts)

    features = list(dv.get_feature_names_out())

    dmatrix = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dmatrix)

    return y_pred


# Training the Final Model

print("\n=== Training the Final Model ===\n")

dv, model = train(X_full_train, y_full_train)


# Model Evaluation

print("\n=== Model Evaluation on Test Set ===\n")

y_pred = predict(X_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f"Test AUC: {auc:.4f}")


# Save the Model

artifact = {
    "dv": dv,
    "model": model,
    "threshold": threshold
}

with open(output_file, "wb") as f_out:
    pickle.dump(artifact, f_out)

print(f"\nThe model is saved to: {output_file}")