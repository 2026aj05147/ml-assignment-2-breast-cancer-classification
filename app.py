"""Evaluate saved classifiers on labeled breast-cancer diagnostic measurements."""

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
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
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e8e4f2;}
        [data-testid="stMetric"] {
            background: #f7f5ff;
            border: 1px solid #eae5fb;
            border-radius: 8px;
            padding: 0.65rem;
        }
        .matrix-table {width: 100%; border-collapse: collapse; font-size: 0.9rem;}
        .matrix-table th, .matrix-table td {
            padding: 0.45rem 0.6rem; border: 1px solid #ddd7ee; text-align: left;
        }
        .matrix-table thead th {background: #e9e3f7; color: #372b53;}
        .matrix-table tbody th {background: #f4f1fa; color: #372b53;}
        .matrix-table td {background: #fff;}
        .matrix-table td strong {display: block; font-size: 1.15rem; color: #31234c;}
        .matrix-table td span {display: block; font-size: 0.78rem; color: #584773;}
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
        st.markdown("**1. Download the test data**")
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
                "must be present to calculate AUC. Use the test CSV as a template."
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
            "To evaluate a model: download the test CSV, upload it in the "
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
    for column, (name, value) in zip(st.columns(6), metrics.items(), strict=True):
        column.metric(name, f"{value:.4f}", help=help_text[name])

    matrix_column, report_column = st.columns(2, gap="large")
    with matrix_column:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        st.markdown(
            '<table class="matrix-table" aria-label="Confusion matrix">'
            '<thead><tr><th scope="col">Actual / predicted</th>'
            '<th scope="col">Benign (0)</th><th scope="col">Malignant (1)</th>'
            '</tr></thead><tbody>'
            f'<tr><th scope="row">Benign (0)</th>'
            f'<td><strong>{matrix[0, 0]:,}</strong><span>True negative (TN)</span></td>'
            f'<td><strong>{matrix[0, 1]:,}</strong><span>False positive (FP)</span></td></tr>'
            f'<tr><th scope="row">Malignant (1)</th>'
            f'<td><strong>{matrix[1, 0]:,}</strong><span>False negative (FN)</span></td>'
            f'<td><strong>{matrix[1, 1]:,}</strong><span>True positive (TP)</span></td></tr>'
            '</tbody></table>',
            unsafe_allow_html=True,
        )
        st.caption("Rows are actual diagnoses; columns are model predictions.")
    with report_column:
        st.subheader("Classification report")
        report = classification_report(
            labels, predictions, labels=[0, 1], target_names=["Benign", "Malignant"],
            output_dict=True, zero_division=0,
        )
        del report["accuracy"]
        report_frame = pd.DataFrame.from_dict(report, orient="index").round(4)
        report_frame.index.name = "Class or average"
        report_frame["support"] = report_frame["support"].astype(int)
        st.dataframe(report_frame, use_container_width=True)
        st.caption("Support is the number of cases in each actual class.")
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
        with st.expander("Compare models on uploaded CSV", expanded=True):
            st.subheader("Results on uploaded CSV")
            st.caption(
                f"All five models evaluated on the same {len(frame):,} uploaded cases."
            )
            st.dataframe(
                pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True
            )
    else:
        st.info("To compare all models on this CSV, select the comparison option in the sidebar.")
    with st.expander("Recorded holdout benchmark"):
        comparison_table()
    with st.expander("Data preview"):
        st.caption(f"Showing the first {min(len(frame), 20)} of {len(frame):,} uploaded rows.")
        st.dataframe(frame.head(20), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
