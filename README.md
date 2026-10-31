
# 💧 Water Quality Potability Prediction Using Machine Learning

An end-to-end machine learning project that predicts water potability using physicochemical water-quality measurements.

The project includes exploratory data analysis, data preprocessing, model comparison, hyperparameter tuning, evaluation, feature importance, threshold analysis, automated testing, and a Streamlit web application.

## 1. Project Objective

The objective is to develop a machine learning classification system that predicts whether a water sample belongs to the potable or non-potable category based on nine water-quality measurements.

**Target variable:**
- `0` — Non-Potable
- `1` — Potable

**Important:** This project is an educational machine learning demonstration. Predictions cannot establish whether water is safe for human consumption. Laboratory testing and applicable water-quality standards are necessary.

## 2. Dataset

The project uses the Water Potability dataset.

| Property | Description |
|---|---|
| Total records | 3,276 |
| Input features | 9 |
| Target variable | Potability |
| Problem type | Binary classification |
| Training split | 80% |
| Testing split | 20% |

### Input Features

1. pH
2. Hardness
3. Solids
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic Carbon
8. Trihalomethanes
9. Turbidity

The dataset contains missing values in pH, Sulfate, and Trihalomethanes.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Pytest

## 4. Machine Learning Workflow

1. Load and explore the dataset.
2. Analyze missing values and feature distributions.
3. Split the dataset into training and testing sets.
4. Apply median imputation for missing values.
5. Establish a baseline classifier.
6. Compare multiple classification algorithms using cross-validation.
7. Tune the selected Random Forest model.
8. Evaluate the trained model.
9. Analyze feature importance.
10. Explore classification thresholds.
11. Save the trained model.
12. Build an interactive prediction application.
13. Perform automated software testing.

Preprocessing is implemented using scikit-learn pipelines to reduce data leakage.

## 5. Algorithms Compared

The following supervised machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Cross-Validation Results

| Algorithm | Accuracy | Balanced Accuracy | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 60.99% | 50.00% | 0.0000 |
| Decision Tree | 63.70% | 55.82% | 0.2913 |
| Random Forest | 66.91% | 59.23% | 0.3627 |
| Gradient Boosting | 63.74% | 56.43% | 0.3316 |

Random Forest achieved the highest balanced accuracy among the initially compared algorithms.

These are cross-validation results from the training portion of the dataset, not final test-set results.

## 6. Hyperparameter Tuning

RandomizedSearchCV was used to optimize the Random Forest classifier.

**Selected hyperparameters:**

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | None |
| min_samples_split | 10 |
| min_samples_leaf | 2 |
| class_weight | balanced |

Best cross-validation balanced accuracy: **64.55%**

## 7. Model Evaluation

The tuned Random Forest model was evaluated on the held-out test split using the default classification threshold of 0.50.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 65.09% |
| Balanced Accuracy | 61.46% |
| ROC-AUC | 65.93% |
| Potable Precision | 57% |
| Potable Recall | 45% |
| Potable F1 Score | 0.50 |

### Confusion Matrix

| Actual / Predicted | Non-Potable | Potable |
|---|---:|---:|
| Non-Potable | 312 | 88 |
| Potable | 141 | 115 |

At the default threshold, 88 of 400 non-potable test samples were incorrectly classified as potable. This is an important limitation, especially for a water-quality application.

## 8. Feature Importance

Permutation importance was used to examine how strongly individual features affected model performance on the training data.

The three highest-ranked features were:

1. pH
2. Sulfate
3. Hardness

Feature importance does not establish causation or prove that these measurements alone determine drinking-water safety.

## 9. Classification Threshold Analysis

The default classification threshold of 0.50 was compared with stricter alternatives using out-of-fold predictions on the training data.

A threshold of **0.65** was selected to reduce the number of non-potable samples classified as potable.

At this threshold, the training out-of-fold results included:

- Non-potable recall: 96.75%
- Potable recall: 15.85%
- False-potable predictions: 52

This threshold improves the model's ability to reject non-potable samples in the reported training validation results, but substantially reduces its ability to identify potable samples.

These results do not establish safety certification or guarantee equivalent performance on new water samples.

## 10. Streamlit Web Application

The application allows users to:

- Enter nine water-quality measurements.
- Generate a classification using the trained Random Forest model.
- View the model-estimated potability score.
- View the classification threshold.
- Read the model's classification and safety disclaimer.

The model score is not a calibrated probability of drinking-water safety.

## 11. Running the Project Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd water-quality-prediction-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
python -m streamlit run app.py
```

The application should open in a local browser.

The saved model and threshold configuration must be available in the `models/` directory. Otherwise, the model must be trained before launching the application.

## 12. Automated Testing

Pytest is used to verify the prediction system.

The six tests check:

1. Valid prediction output.
2. Prediction score range.
3. Missing-feature handling.
4. Invalid-input handling.
5. Consistent predictions.
6. Expected feature count.

Run the tests:

```bash
python -m pytest tests/ -v
```

**Latest local test result: 6 passed.**

## 13. Project Limitations

- The dataset is relatively small.
- The dataset contains missing values.
- Predictive performance is moderate.
- The model produces false-potable predictions.
- The application uses a stricter threshold that reduces potable-class recall.
- Model probabilities have not been calibrated.
- Repeated experimentation and evaluation on the same test split may make the reported performance optimistic.
- The model has not been validated against independent real-world laboratory measurements.

The system must not be used as a substitute for certified drinking-water testing.

## 14. Future Improvements

- Validate on independent water-quality datasets.
- Improve probability calibration.
- Explore additional algorithms and feature engineering.
- Evaluate the selected threshold on untouched external data.
- Add data-drift monitoring.
- Improve application usability.
- Deploy the application online.

## 15. Project Status

- [x] Exploratory Data Analysis
- [x] Model Comparison
- [x] Hyperparameter Tuning
- [x] Model Evaluation
- [x] Feature Importance
- [x] Threshold Analysis
- [x] Prediction Pipeline
- [x] Streamlit Application Code
- [x] Automated Testing
- [ ] Dependency and Git Configuration
- [ ] GitHub Publishing
- [ ] Online Deployment

---

**Project Type:** End-to-End Machine Learning Classification

**Purpose:** Academic learning, portfolio development, and demonstration of machine learning engineering practices.
