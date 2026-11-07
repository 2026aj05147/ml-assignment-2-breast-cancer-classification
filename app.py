"""Evaluate saved classifiers on labeled breast-cancer diagnostic measurements."""

from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import streamlit as st

from evaluation import CLASSIFIERS, score_pipeline, validate_upload

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def saved_pipeline(filename: str) -> Pipeline:
    return joblib.load(ROOT / "model" / filename)


def main() -> None:
    st.set_page_config(page_title="Diagnostic Classifier Studio", layout="wide")
    st.title("Diagnostic Classifier Studio")
    st.caption("UCI Wisconsin Diagnostic data | 30 measurements | 5 saved classifiers")
    st.warning(
        "Educational model comparison only. Not validated for clinical use; "
        "do not use these predictions for diagnosis or treatment."
    )
    with st.sidebar:
        st.header("Evaluation controls")
        choice = st.selectbox("Classifier", list(CLASSIFIERS))
        upload = st.file_uploader("Labeled diagnostic CSV", type=["csv"])
        st.download_button(
            "Download held-out example CSV",
            (ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.info(
            "Upload 30 numeric feature columns plus malignant: "
            "0 = benign, 1 = malignant. Include both classes to calculate AUC."
        )
    if upload is None:
        st.info("Download the example, then upload it to evaluate a saved pipeline.")
        st.subheader("Recorded holdout comparison")
        st.dataframe(pd.read_csv(ROOT / "results" / "comparison.csv"), hide_index=True)
        return
    pipeline = saved_pipeline(CLASSIFIERS[choice])
    try:
        frame = pd.read_csv(BytesIO(upload.getvalue()))
        features, labels = validate_upload(frame, list(pipeline.feature_names_in_))
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        return
    metrics, predictions = score_pipeline(pipeline, features, labels)
    st.subheader(f"{choice}: {len(features)} labeled cases")
    st.caption("Positive class = malignant. Saved training scaler and classifier; no retraining.")
    for column, (name, value) in zip(st.columns(6), metrics.items(), strict=True):
        column.metric(name, f"{value:.4f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Where predictions disagree")
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(labels, predictions, labels=[0, 1]),
            display_labels=["Benign (0)", "Malignant (1)"],
        ).plot(ax=axis, cmap="Purples", colorbar=False)
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Class-level report")
        report = classification_report(
            labels, predictions, labels=[0, 1], target_names=["Benign", "Malignant"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4))
    st.subheader("Uploaded measurements")
    st.dataframe(frame.head(20))


if __name__ == "__main__":
    main()
