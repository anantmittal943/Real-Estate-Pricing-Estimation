from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REQUIRED_TRAIN_COLUMNS,
    TEST_SIZE,
)


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_TRAIN_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Dataset schema mismatch. Missing columns: " + ", ".join(missing_columns)
        )
    return dataframe


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    transformed["date"] = pd.to_datetime(transformed["date"], errors="coerce")

    if transformed["date"].isna().any():
        raise ValueError("The input data contains invalid date values.")

    transformed["sale_year"] = transformed["date"].dt.year
    transformed["age_at_sale"] = transformed["sale_year"] - transformed["yr_built"]
    transformed["age_since_renovation"] = np.where(
        transformed["yr_renovated"] == 0,
        transformed["age_at_sale"],
        transformed["sale_year"] - transformed["yr_renovated"],
    )

    transformed = transformed.drop(columns=["date", "id"], errors="ignore")
    return transformed


def prepare_training_data(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    transformed = engineer_features(dataframe)
    if "price" not in transformed.columns:
        raise ValueError("Training data must include the 'price' column.")

    features = transformed.drop(columns=["price"])
    target = transformed["price"]
    return features, target


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_linear_regression(dataframe: pd.DataFrame):
    features, target = prepare_training_data(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = calculate_metrics(y_test, y_pred)

    return model, list(features.columns), metrics


def save_model_artifacts(model: LinearRegression, feature_columns, metrics: Dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_columns": feature_columns,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "LinearRegression",
    }

    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def load_model_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts were not found. Run train_model.py before launching dashboard."
        )

    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return model, metadata


def build_inference_frame(raw_input: Dict[str, float], feature_columns) -> pd.DataFrame:
    inference_df = pd.DataFrame([raw_input])
    transformed = engineer_features(inference_df)

    if "price" in transformed.columns:
        transformed = transformed.drop(columns=["price"])

    for column in feature_columns:
        if column not in transformed.columns:
            transformed[column] = 0

    return transformed[feature_columns]


def get_test_predictions(model: LinearRegression, dataframe: pd.DataFrame):
    features, target = prepare_training_data(dataframe)

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    y_pred = model.predict(x_test)
    metrics = calculate_metrics(y_test, y_pred)

    predictions = pd.DataFrame({"actual_price": y_test.values, "predicted_price": y_pred})
    return predictions, metrics


def train_and_save_model(csv_path: Path) -> Dict[str, float]:
    dataframe = load_raw_data(csv_path)
    model, feature_columns, metrics = train_linear_regression(dataframe)
    save_model_artifacts(model, feature_columns, metrics)
    return metrics
