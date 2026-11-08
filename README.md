# Diagnostic Classifier Studio

## a. Problem statement

Compare five classification methods for distinguishing benign and malignant
breast-mass samples using numerical measurements extracted from digitized
fine-needle aspirate images. The experiment focuses on malignant-class recall,
precision, F1 and MCC as well as accuracy and ROC AUC. An interactive Streamlit
app evaluates the saved classifiers on a user-uploaded labeled CSV, without
retraining.

This is an educational benchmark, not a clinical diagnostic system. No
performance claim here establishes safety for diagnosis or treatment.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository.
- **Public source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Local access:** `sklearn.datasets.load_breast_cancer(as_frame=True)` loads
  scikit-learn's bundled copy of the UCI dataset; no network download is required.
- **Instances:** 569, satisfying the minimum of 500.
- **Predictors:** 30 real-valued features, satisfying the minimum of 12.
- **Measurements:** radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry and fractal dimension, each summarized
  by mean, standard error and worst value.
- **Classes:** 357 benign samples (62.7%) and 212 malignant samples (37.3%).
- **Missing values:** none in the bundled dataset.
- **Target used here:** `malignant`, with **0 = benign, 1 = malignant**.
  The training script explicitly reverses scikit-learn's original target
  encoding so the malignant class is the positive class for all metrics.
- **Input naming:** spaces in scikit-learn feature names are replaced with
  underscores, for example `mean_radius`. The patient ID is not a predictor.

### Experiment design

A single stratified 80/20 split with seed 17 is shared by all five models:
455 training rows and 114 holdout rows (72 benign, 42 malignant).
Standardization is fitted exclusively on the training data inside each
pipeline. The exact holdout is exported as `test_data.csv`, containing all
30 predictors and the labeled target.

Hyperparameters were specified before evaluation; there is no holdout-driven
hyperparameter or threshold search. All models use their default prediction
thresholds. This small holdout is used to compare models, not to establish
unbiased performance of a subsequently selected winner. External validation
would be necessary for stronger generalization claims.

## c. GitHub repository

**GitHub repository:** https://github.com/2026aj05147/ml-assignment-2-breast-cancer-classification

**Live Streamlit app:** https://2026aj05147-ml-assignment.streamlit.app/

The repository includes the application, source code, pinned requirements,
test CSV and five saved models. Each saved pipeline contains its fitted scaler.

## d. Models used and comparison

| Classifier | Configuration |
|---|---|
| Logistic Regression | C=1.0, max_iter=2000 |
| Decision Tree | max_depth=4, min_samples_leaf=4, random_state=17 |
| K-Nearest Neighbors | 7 neighbors, uniform distance voting |
| Gaussian Naive Bayes | GaussianNB defaults |
| Support Vector Machine | RBF kernel, C=1.0, gamma=scale |

### Comparison table

All values below are from the same 114-row labeled holdout. Precision, recall
and F1 refer to malignant samples; AUC uses malignant-class probabilities or,
for SVM, continuous decision scores. MCC uses the predicted binary labels.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9974 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9474 | 0.9519 | 0.9737 | 0.8810 | 0.9250 | 0.8872 |
| K-Nearest Neighbors | 0.9737 | 0.9797 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Gaussian Naive Bayes | 0.9474 | 0.9927 | 0.9500 | 0.9048 | 0.9268 | 0.8864 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.9974), suggesting strong ranking on this holdout. Its default predictions miss 3 malignant samples and incorrectly flag 1 benign sample. Accuracy and F1 are slightly below KNN and SVM despite the stronger AUC. |
| Decision Tree | The depth-limited tree misses 5 malignant samples and falsely flags 1 benign sample. It has the lowest recall (0.8810) and AUC (0.9519); these results do not prove overfitting because training performance was not compared here. |
| K-Nearest Neighbors | Ties SVM on accuracy, precision, recall, F1 and MCC: 3 false negatives and no false positives. AUC is lower than SVM's, showing that identical hard-prediction metrics need not imply identical ranking performance. Scaling prevents large-valued features such as area from dominating distances. |
| Gaussian Naive Bayes | AUC is strong (0.9927), but its default decisions miss 4 malignant samples and falsely flag 2 benign samples. Its lowest MCC (0.8864) and lower precision distinguish it from the other models. Correlated measurement summaries make conditional independence a simplifying assumption, not an established property of this dataset. |
| Support Vector Machine | The RBF SVM shares the best hard-prediction metrics with KNN and has a higher AUC (0.9947 versus 0.9797). It misses 3 malignant samples with no false positives on this holdout; that does not guarantee perfect precision on new cases. |
| **Overall, Winner for your dataset?** | **SVM, with KNN explicitly tied on all five hard-prediction metrics.** SVM is selected using AUC as the tie-breaker. Logistic Regression is the AUC-only winner. None of these models eliminates false negatives, and the small holdout does not justify a clinical-use recommendation. |

## Run the application

Use **Python 3.12**. Saved scikit-learn pipelines are not supported across
arbitrary library versions; the pinned dependencies match the training runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Download the provided holdout from the sidebar and upload it for evaluation.
Select a classifier to view its six metrics, confusion matrix and
classification report, or use the comparison option to evaluate all five.
The matrix uses actual classes as rows and predicted classes as columns;
malignant (1) is positive. The app loads saved pipelines without retraining.

Other test CSVs must use the same 30 feature names and include the `malignant`
target with both classes; a single-class file cannot produce ROC AUC.
To reproduce training and regenerate the artifacts, run `python train.py`.
