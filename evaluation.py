"""Shared scoring and labeled-upload validation for the diagnostic dataset."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

TARGET = "malignant"
CLASSIFIERS = {
    "Logistic Regression": "logistic.joblib",
    "Decision Tree": "tree.joblib",
    "K-Nearest Neighbors": "neighbors.joblib",
    "Gaussian Naive Bayes": "bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def validate_upload(
    frame: pd.DataFrame, expected_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    if frame.empty:
        raise ValueError("The CSV must contain data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    missing = set(expected_columns + [TARGET]) - set(frame.columns)
    extra = set(frame.columns) - set(expected_columns + [TARGET])
    if missing or extra:
        raise ValueError(
            f"CSV schema mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}. Use the supplied test_data.csv schema."
        )
    features = frame.loc[:, expected_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be numeric, non-missing and finite.")
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError(
            "The malignant column must contain both classes: "
            "0 = benign and 1 = malignant. Both are needed for ROC AUC."
        )
    return features, target.astype(int)


def score_pipeline(
    pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series
) -> tuple[dict[str, float], np.ndarray]:
    predictions = pipeline.predict(features)
    if hasattr(pipeline, "predict_proba"):
        positive_index = list(pipeline.classes_).index(1)
        scores = pipeline.predict_proba(features)[:, positive_index]
    else:
        scores = pipeline.decision_function(features)
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "AUC": float(roc_auc_score(labels, scores)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }, predictions
