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

**GitHub repository:** https://github.com/2026aj05147/breast-cancer-classifier-lab

**Live Streamlit app:** pending deployment; insert the verified app URL here.

### Repository layout

```text
app.py                         Streamlit upload/evaluation interface
.streamlit/config.toml         Streamlit color theme
evaluation.py                  Shared metrics and strict CSV validation
train.py                       Reproducible training on the bundled UCI data
requirements.txt               Versions matching the saved model environment
README.md                      Experiment report and submission instructions
test_data.csv                  114 labeled holdout rows
model/
    logistic.joblib            StandardScaler + LogisticRegression
    tree.joblib                StandardScaler + DecisionTreeClassifier
    neighbors.joblib           StandardScaler + KNeighborsClassifier
    bayes.joblib               StandardScaler + GaussianNB
    svm.joblib                 StandardScaler + SVC
results/
    comparison.csv             Six measured metrics for each model
    training_metadata.json     Dataset counts, split seed and runtime versions
tests/
    test_project.py            Artifact, upload-validation and app tests
```

The preprocessing artifacts are embedded in each saved pipeline; do not scale
uploaded data separately or fit a new scaler in the app.

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

## Reproduce the experiment

Use **Python 3.12**. Saved scikit-learn pipelines are not supported across
arbitrary library versions; the pinned dependencies match the training runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m pytest -q
streamlit run app.py
```

Training regenerates all five pipelines, the comparison CSV, metadata and
labeled holdout. The app only loads saved pipelines.

## App walkthrough

1. Download the held-out example using the sidebar button.
2. Upload that CSV using the labeled-data uploader.
3. Choose each of the five classifiers from the dropdown.
4. Inspect all six metrics, the confusion matrix, classification report, data
   preview and recorded holdout comparison in the results tabs. The recorded
   comparison uses the supplied holdout, even if a different CSV is uploaded.
5. For another CSV, keep the exact feature schema and include both target
   classes. Missing columns, unexpected columns, nonnumeric/nonfinite
   measurements and invalid labels produce visible errors.

Single-class uploads cannot produce ROC AUC, so the app rejects them rather
than presenting a misleading score.

## Deploy on Streamlit Community Cloud

Push the complete project, including `model/` and `test_data.csv`, to the
student's own repository. Select its branch and `app.py` entrypoint; under
**Advanced settings**, choose **Python 3.12** before deployment.

Do not use the default Python 3.14 runtime with these saved models and pinned
dependencies. For an existing app on a different Python version, record its
settings, then delete and redeploy it with Python 3.12. A reboot does not change
the Python runtime.

After deployment, verify the CSV upload workflow with all five models, not just
the initial page, and replace the pending links above.

## Submission and authorship checklist

- Add the actual GitHub and deployed-app URLs.
- Run the assignment in BITS Virtual Lab and capture the required screenshot.
- Assemble a single PDF in the required order: GitHub URL, live app URL,
  Virtual Lab screenshot, then the full README.
- Check the course rules on AI assistance and collaboration. This starter was
  created with AI assistance; acknowledge that assistance wherever required.
- The submitting student should review the code, reproduce the measurements
  and write their own interpretation based on what they understand.
- This is not a renamed copy of the bank-marketing project, but a different
  dataset or layout cannot guarantee originality-check acceptance. Do not
  fabricate authorship or commit history.
