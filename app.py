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


def comparison_table() -> None:
    st.caption(
        "Recorded on the provided holdout split; these values do not change "
        "when a different CSV is uploaded."
    )
    st.dataframe(
        pd.read_csv(ROOT / "results" / "comparison.csv").round(4),
        hide_index=True,
        use_container_width=True,
    )


@st.cache_resource
def saved_pipeline(filename: str) -> Pipeline:
    return joblib.load(ROOT / "model" / filename)


def main() -> None:
    st.set_page_config(
        page_title="Diagnostic Classifier Studio", page_icon="🔬", layout="wide"
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 5rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e8e4f2;}
        [data-testid="stMetric"] {
            background: #f7f5ff;
            border: 1px solid #eae5fb;
            border-radius: 12px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("MACHINE LEARNING ASSIGNMENT 2")
    st.title("🔬 Diagnostic Classifier Studio")
    st.write(
        "Explore and compare five classifiers on the Wisconsin Breast Cancer "
        "Diagnostic dataset."
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
        compare_uploaded = st.checkbox(
            "Compare all models on this CSV",
            help="Evaluate each saved pipeline on the uploaded labeled data.",
        )
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
        st.divider()
        st.subheader("Dataset at a glance")
        for column, label, value in zip(
            st.columns(4),
            ("Samples", "Features", "Classifiers", "Holdout cases"),
            ("569", "30", "5", "114"),
            strict=True,
        ):
            column.metric(label, value)
        st.caption(
            "Two classes: benign (0) and malignant (1). "
            "The same stratified 80/20 split is used for all five models."
        )
        st.info(
            "To evaluate a model: download the example CSV, upload it in the "
            "sidebar, and select a classifier. Choose the comparison option "
            "to evaluate all five models on the uploaded data."
        )
        st.subheader("Recorded holdout results")
        st.write("Performance metrics for each trained classifier.")
        comparison_table()
        return
    pipeline = saved_pipeline(CLASSIFIERS[choice])
    try:
        frame = pd.read_csv(BytesIO(upload.getvalue()))
        features, labels = validate_upload(frame, list(pipeline.feature_names_in_))
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        return
    metrics, predictions = score_pipeline(pipeline, features, labels)
    st.divider()
    st.subheader("Model evaluation")
    st.write(
        f"**{choice}** | {len(features):,} cases | "
        f"{int((labels == 0).sum()):,} benign | "
        f"{int((labels == 1).sum()):,} malignant"
    )
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
        ["Confusion matrix", "Classification report", "Data preview", "Model comparison"]
    )
    with matrix_tab:
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        st.caption("Rows are actual classes; columns are predicted classes.")
        st.write(
            f"**Correct benign:** {matrix[0, 0]:,}  |  "
            f"**Correct malignant:** {matrix[1, 1]:,}  |  "
            f"**False positives:** {matrix[0, 1]:,}  |  "
            f"**False negatives:** {matrix[1, 0]:,}"
        )
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            matrix,
            display_labels=["Benign (0)", "Malignant (1)"],
        ).plot(ax=axis, cmap="Purples", colorbar=False)
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
    with report_tab:
        report = classification_report(
            labels, predictions, labels=[0, 1], target_names=["Benign", "Malignant"],
            output_dict=True, zero_division=0,
        )
        del report["accuracy"]
        report_frame = pd.DataFrame.from_dict(report, orient="index").round(4)
        report_frame.index.name = "Class or average"
        report_frame["support"] = report_frame["support"].astype(int)
        st.dataframe(report_frame, use_container_width=True)
        st.caption("Overall accuracy is shown in the model evaluation above.")
    with data_tab:
        st.caption(f"Showing the first {min(len(frame), 20)} of {len(frame):,} uploaded rows.")
        st.dataframe(frame.head(20), hide_index=True, use_container_width=True)
    with comparison_tab:
        if compare_uploaded:
            with st.spinner("Evaluating all five models on the uploaded CSV..."):
                rows = []
                for name, filename in CLASSIFIERS.items():
                    model = saved_pipeline(filename)
                    model_features, model_labels = validate_upload(
                        frame, list(model.feature_names_in_)
                    )
                    model_metrics, _ = score_pipeline(model, model_features, model_labels)
                    rows.append({"ML Model Name": name, **model_metrics})
            st.subheader("Results on uploaded CSV")
            st.caption(
                f"All five models evaluated on the same {len(frame):,} uploaded cases."
            )
            st.dataframe(
                pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True
            )
        else:
            st.info("Enable 'Compare all models on this CSV' in the sidebar for a custom comparison.")
        st.subheader("Original holdout benchmark")
        comparison_table()


if __name__ == "__main__":
    main()
