"""Regression checks for upload validation, persisted models and app evaluation."""

from io import BytesIO
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit.testing.v1 import AppTest

from evaluation import CLASSIFIERS, TARGET, score_pipeline, validate_upload

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,filename", CLASSIFIERS.items())
def test_saved_pipeline_and_app(monkeypatch, name, filename):
    frame = pd.read_csv(ROOT / "test_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(ROOT / "model" / filename)
        features, labels = validate_upload(frame, list(pipeline.feature_names_in_))
        metrics, predictions = score_pipeline(pipeline, features, labels)
    expected = pd.read_csv(ROOT / "results" / "comparison.csv").set_index("ML Model Name")
    assert len(predictions) == len(frame) == 114
    assert len(features.columns) == 30
    for key, value in metrics.items():
        assert value == pytest.approx(expected.loc[name, key])
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(frame.to_csv(index=False).encode()))
    monkeypatch.setattr(st, "selectbox", lambda *args, **kwargs: name)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert "Model evaluation" in [heading.value for heading in app.subheader]
    assert len(app.metric) == 6
    assert [tab.label for tab in app.tabs] == [
        "Confusion matrix", "Classification report", "Data preview",
        "Model comparison",
    ]
    for metric in app.metric:
        assert float(metric.value) == pytest.approx(metrics[metric.label], abs=0.000051)


@pytest.mark.parametrize("fault", ["empty", "missing", "extra", "one_class", "nan", "infinity", "text", "duplicate"])
def test_invalid_upload_is_rejected(fault):
    frame = pd.read_csv(ROOT / "test_data.csv")
    expected = frame.drop(columns=TARGET).columns.tolist()
    if fault == "empty":
        frame = frame.iloc[:0]
    elif fault == "missing":
        frame = frame.drop(columns=expected[0])
    elif fault == "extra":
        frame["patient_id"] = 1
    elif fault == "one_class":
        frame[TARGET] = 0
    elif fault == "nan":
        frame.loc[0, expected[0]] = np.nan
    elif fault == "infinity":
        frame.loc[0, expected[0]] = np.inf
    elif fault == "text":
        frame[expected[0]] = "invalid"
    else:
        frame = pd.concat([frame, frame[[expected[0]]]], axis=1)
    with pytest.raises(ValueError):
        validate_upload(frame, expected)


def test_app_without_upload():
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 5
    assert len(app.dataframe) == 1
    assert len(app.warning) == 0
    assert all("—" not in caption.value for caption in app.caption)


def test_app_shows_invalid_csv_error(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(b"bad\nvalue\n"))
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert len(app.error) == 1
    assert len(app.metric) == 0


def test_compare_all_models_on_uploaded_data(monkeypatch):
    frame = pd.read_csv(ROOT / "test_data.csv")
    monkeypatch.setattr(
        st, "file_uploader", lambda *args, **kwargs: BytesIO(frame.to_csv(index=False).encode())
    )
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    app.checkbox[0].set_value(True).run()
    assert not app.exception
    assert "Results on uploaded CSV" in [heading.value for heading in app.subheader]
    expected = pd.read_csv(ROOT / "results" / "comparison.csv").round(4)
    comparison = app.dataframe[-2].value
    pd.testing.assert_frame_equal(comparison, expected)


def test_holdout_and_scaler_use_the_documented_split():
    data = load_breast_cancer(as_frame=True)
    features = data.data.rename(columns=lambda name: name.replace(" ", "_"))
    labels = (1 - data.target).rename(TARGET)
    train_features, test_features, _, test_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=17
    )
    assert features.shape == (569, 30)
    assert len(train_features) == 455
    assert set(train_features.index).isdisjoint(test_features.index)
    expected = pd.concat([test_features, test_labels], axis=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(ROOT / "test_data.csv"), expected)
    for filename in CLASSIFIERS.values():
        pipeline = joblib.load(ROOT / "model" / filename)
        np.testing.assert_allclose(
            pipeline.named_steps["scale"].mean_, train_features.mean().to_numpy()
        )


def test_readme_metric_table_matches_generated_results():
    results = pd.read_csv(ROOT / "results" / "comparison.csv").set_index("ML Model Name")
    count = 0
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in CLASSIFIERS:
            assert [float(value) for value in cells[1:]] == [
                round(value, 4) for value in results.loc[cells[0]]
            ]
            count += 1
    assert count == 5
