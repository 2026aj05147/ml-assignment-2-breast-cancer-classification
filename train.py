"""Fit five diagnostic classifiers using one reproducible stratified holdout."""

import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import CLASSIFIERS, TARGET, score_pipeline

ROOT = Path(__file__).resolve().parent
SEED = 17


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.rename(columns=lambda name: name.replace(" ", "_"))
    labels = (1 - dataset.target).rename(TARGET)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    estimators = (
        LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=SEED),
        KNeighborsClassifier(n_neighbors=7),
        GaussianNB(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    )
    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows = []
    for (name, filename), estimator in zip(CLASSIFIERS.items(), estimators, strict=True):
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", estimator)])
        pipeline.fit(x_train, y_train)
        metrics, _ = score_pipeline(pipeline, x_test, y_test)
        rows.append({"ML Model Name": name, **metrics})
        joblib.dump(pipeline, ROOT / "model" / filename)
        print(name, {key: round(value, 4) for key, value in metrics.items()})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "comparison.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)
    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic), sklearn bundled copy",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "rows": len(features),
        "features": features.shape[1],
        "class_counts": {"benign": int((labels == 0).sum()), "malignant": int(labels.sum())},
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "seed": SEED,
        "positive_class": "1 = malignant",
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    (ROOT / "results" / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
