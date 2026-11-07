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
    st.set_page_config(
        page_title="Diagnostic Classifier Studio", page_icon="🔬", layout="wide"
    )
    st.title("🔬 Diagnostic Classifier Studio")
    st.write(
        "Compare five machine learning models using the Wisconsin Breast Cancer "
        "Diagnostic dataset."
    )
    st.caption(
        "Educational benchmark only — not validated for clinical diagnosis or treatment."
    )
    with st.sidebar:
        st.header("Get started")
        st.markdown("**1. Download the example data**")
        st.download_button(
            "Download test CSV",
            (ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("**2. Upload labeled test data**")
        upload = st.file_uploader("Test data CSV", type=["csv"])
        st.markdown("**3. Select a model**")
        choice = st.selectbox("Classifier", list(CLASSIFIERS))
        st.divider()
        with st.expander("CSV requirements"):
            st.write(
                "Include the 30 numeric feature columns and the `malignant` "
                "target column: 0 = benign, 1 = malignant. Both classes "
                "must be present to calculate AUC. Use the example as a template."
            )
        st.caption(
            "Models are pre-trained; the app evaluates them without retraining."
        )
    if upload is None:
        st.subheader("Explore the benchmark")
        st.write(
            "**569** samples · **30** features · **5** classifiers · "
            "**114** held-out test cases"
        )
        st.info(
            "Start by downloading the example CSV, uploading it in the sidebar, "
            "and choosing a classifier."
        )
        st.subheader("Recorded holdout results")
        st.caption(
            "All models were evaluated on the same held-out split. "
            "Precision, recall and F1 use malignant as the positive class."
        )
        st.dataframe(
            pd.read_csv(ROOT / "results" / "comparison.csv").round(4),
            hide_index=True,
            use_container_width=True,
        )
        return
    pipeline = saved_pipeline(CLASSIFIERS[choice])
    try:
        frame = pd.read_csv(BytesIO(upload.getvalue()))
        features, labels = validate_upload(frame, list(pipeline.feature_names_in_))
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        return
    metrics, predictions = score_pipeline(pipeline, features, labels)
    st.subheader("Evaluation results")
    st.write(f"**{choice}** · {len(features):,} uploaded cases")
    st.caption(
        "Positive class: malignant (1). The saved preprocessing and model "
        "are applied without retraining."
    )
    help_text = {
        "Accuracy": "Fraction of all cases classified correctly.",
        "AUC": "ROC AUC measures how well the model ranks malignant cases above benign cases.",
        "Precision": "Fraction of predicted malignant cases that are actually malignant.",
        "Recall": "Fraction of actual malignant cases correctly identified.",
        "F1": "Harmonic mean of malignant-class precision and recall.",
        "MCC": "Matthews correlation coefficient; ranges from -1 to 1.",
    }
    for row in (list(metrics.items())[:3], list(metrics.items())[3:]):
        for column, (name, value) in zip(st.columns(3), row, strict=True):
            column.metric(name, f"{value:.4f}", help=help_text[name])

    matrix_tab, report_tab, data_tab, comparison_tab = st.tabs(
        ["Confusion matrix", "Classification report", "Data preview", "Holdout comparison"]
    )
    with matrix_tab:
        st.caption("Rows are actual classes; columns are predicted classes.")
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(labels, predictions, labels=[0, 1]),
            display_labels=["Benign (0)", "Malignant (1)"],
        ).plot(ax=axis, cmap="Purples", colorbar=False)
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
    with report_tab:
        report = classification_report(
            labels, predictions, labels=[0, 1], target_names=["Benign", "Malignant"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    with data_tab:
        st.caption(f"Showing the first {min(len(frame), 20)} of {len(frame):,} uploaded rows.")
        st.dataframe(frame.head(20), hide_index=True, use_container_width=True)
    with comparison_tab:
        st.caption(
            "These are the recorded results on the provided holdout, not "
            "necessarily the uploaded data."
        )
        st.dataframe(
            pd.read_csv(ROOT / "results" / "comparison.csv").round(4),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
