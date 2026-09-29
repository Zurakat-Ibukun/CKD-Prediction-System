# 🩺 Chronic Kidney Disease Prediction System

### A Comparative Analysis of Machine Learning Algorithms for the Early Detection of Chronic Kidney Disease (NHANES 2021–2023)

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)

> ⚠️ **Disclaimer:** This application is a research and educational prototype. It is **not** a diagnostic tool and must not be used as a substitute for professional medical assessment.

---

## 📌 Project Overview

Chronic Kidney Disease (CKD) is often asymptomatic in its early stages, so early detection is essential for timely monitoring and intervention.

This project is a B.Sc. Computer Science final-year project (University of Lagos, 2026). It compares **seven machine learning algorithms** for CKD classification using **NHANES 2021–2023** data, selects predictors with the **Boruta** algorithm, and deploys the best-performing model, **XGBoost**, as an interactive **Streamlit** application with **SHAP** explanations.

---

## 🎯 Project Objective

The main objective is to compare machine learning models for the early detection of CKD using demographic, clinical and laboratory data, interpret the selected model with Explainable AI, and deploy it as a research prototype.

The project specifically aims to:

- Preprocess and prepare NHANES data for machine learning.
- Identify relevant predictors of CKD using Boruta feature selection.
- Train seven classifiers under the same experimental conditions.
- Evaluate models using multiple performance metrics.
- Select the best-performing model for deployment.
- Apply SHAP for model explainability.
- Develop an interactive web-based CKD prediction prototype.

---

## 📊 Dataset

| Characteristic | Description |
|---|---|
| Source | CKD dataset on Kaggle, derived from NHANES 2021–2023 (`CKD_NHANES_2021_2023.csv`) |
| Records | 11,933 |
| Variables | 29 original variables (23 candidate predictors after preprocessing) |
| Target | `ckd_present` (CKD / No CKD), a supervised binary classification task |
| Class distribution | 8,341 CKD (69.90%) and 3,592 No CKD (30.10%) |
| Train/test split | 80:20 stratified split: 9,546 training and 2,387 test observations (random seed 42) |

The data are secondary, publicly available NHANES data from the National Center for Health Statistics.

### Preprocessing

The train-test split was performed **before** any fitted preprocessing, so no information from the test set leaked into training. Median imputation (numeric) and mode imputation (categorical), IQR-based outlier capping and z-score standardization were all learned from the training data only, then applied unchanged to the test data and to the deployed application.

---

## 🧪 Feature Selection

Feature selection was performed with the **Boruta** algorithm (Random Forest estimator, 200 trees, balanced class weights) on the training set only.

The preprocessing pipeline produced **23 candidate features**. Boruta retained **16** and rejected 7.

### Selected Features (16)

- Age
- BMI
- Weight
- Height
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Serum Creatinine
- Blood Urea Nitrogen
- Phosphorus
- Bicarbonate
- Calcium
- Uric Acid
- Urine Creatinine
- Urine Albumin
- Albumin Creatinine Ratio
- eGFR

### Rejected Features (7)

Gender, Poverty-Income Ratio, Education Level, Ethnicity, Diabetes Status, Smoking History, Serum Albumin

---

## 🤖 Machine Learning Models

The following algorithms were tuned (5-fold cross-validated grid search on the training set, scored by ROC-AUC; Naïve Bayes was fitted directly) and evaluated on the same held-out test set:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Naïve Bayes
6. Support Vector Machine (SVM)
7. XGBoost

---

## 📈 Model Performance Comparison

Results on the held-out test set (n = 2,387):

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **98.07%** | 98.33% | **98.92%** | 96.11% | **98.63%** | **99.85%** |
| Random Forest | 97.91% | 98.15% | 98.86% | 95.69% | 98.51% | **99.85%** |
| Decision Tree | 97.78% | **98.44%** | 98.38% | **96.38%** | 98.41% | 99.57% |
| SVM | 95.81% | 95.85% | 98.26% | 90.13% | 97.04% | 98.35% |
| KNN | 93.93% | 94.56% | 96.88% | 87.07% | 95.71% | 98.11% |
| Logistic Regression | 91.50% | 91.45% | 96.88% | 79.00% | 94.09% | 92.99% |
| Naïve Bayes | 88.27% | 95.72% | 87.11% | 90.96% | 91.21% | 96.03% |

---

## 🏆 Model Selection

**XGBoost was selected as the deployed model.** It achieved the highest accuracy, recall and F1-score, and tied with Random Forest for the highest ROC-AUC.

| Metric | XGBoost |
|---|---:|
| Accuracy | 98.07% |
| Precision | 98.33% |
| Recall | 98.92% |
| Specificity | 96.11% |
| F1-Score | 98.63% |
| ROC-AUC | 99.85% |

The Decision Tree scored marginally higher on precision (98.44%) and specificity (96.38%), but XGBoost had the strongest performance overall.

### Tuned XGBoost Configuration

100 trees, max depth 5, learning rate 0.1, subsample 0.8, column sampling 0.8 (5-fold CV ROC-AUC: 0.9982).

### Validation

- **Confusion matrix (test set):** 691 TN, 28 FP, 18 FN, 1,650 TP
- **10-fold stratified cross-validation:** mean accuracy 98.23% (SD 0.52%)

---

## 🔬 Feature Importance and Ablation

The most influential XGBoost predictors were **eGFR (0.2994)**, **blood urea nitrogen (0.2134)** and **serum creatinine (0.1861)**, followed by uric acid (0.0817) and albumin-creatinine ratio (0.0575).

Because CKD staging is defined around eGFR, part of the headline performance reflects the definition of the label. To test this, XGBoost was re-tuned and retrained **without eGFR and serum creatinine** (14 predictors):

| Configuration | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| With eGFR & serum creatinine | 98.07% | 98.33% | 98.92% | 96.11% | 98.63% | 99.85% |
| Without eGFR & serum creatinine | 89.95% | 92.70% | 92.93% | 83.03% | 92.81% | 96.35% |

The model still discriminates well from the remaining demographic, anthropometric and laboratory predictors, but the full-model results should be read alongside this more conservative estimate.

---

## 🧠 Explainable AI with SHAP

The system uses **SHAP (SHapley Additive exPlanations)** to explain the XGBoost model:

- Which features influenced a prediction.
- Whether a feature pushed the prediction towards or away from CKD.
- The relative contribution of individual features, globally and per prediction.

SHAP explains model behaviour; it does not establish medical causation.

---

## 💻 Application Features

The deployed Streamlit application (hosted on Streamlit Cloud from this GitHub repository) provides:

### 📊 Prediction

Users enter demographic details, medical/lifestyle history, vital signs and laboratory values, then click **Predict CKD** to obtain:

- CKD / No CKD classification
- Prediction probability

The app applies the same preprocessing pipeline used in training (encoding, IQR capping, scaling and feature selection) before predicting.

### 📋 Patient Summary

Displays the submitted patient information and laboratory measurements.

### 🧠 SHAP Explainability

Visual explanations of the prediction, showing the features that contributed to the result.

### ℹ️ Model Information

Information about the dataset, feature selection, machine learning models, model performance, the selected XGBoost model and the research methodology, along with a notice that the app is not a substitute for professional medical diagnosis.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.13 | Programming language |
| JupyterLab (Anaconda) | Development and experimentation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing, machine learning and evaluation |
| Boruta | Feature selection |
| XGBoost | Final prediction model |
| SHAP | Model explainability |
| Streamlit | Web application |
| Matplotlib / Seaborn | Data visualization |
| Joblib | Model serialization |

---

## 📁 Project Structure

```text
CKD-Prediction-System/
│
├── app.py
├── style.css
├── requirements.txt
│
├── best_xgboost_model.pkl
├── scaler.pkl
├── selected_features.pkl
│
├── kidney.png
│
└── README.md
```

---

## ⚠️ Limitations

- The data are a single Kaggle-held NHANES extract with no external validation set, so clinical viability is unproven.
- The task is binary (CKD vs. No CKD); CKD stage classification is left for future work.
- The model relies heavily on eGFR and serum creatinine, which are part of the clinical definition of CKD (see the ablation above).
- IQR capping compresses some variables into narrow ranges (for example, serum creatinine was capped at roughly 0.765–0.885 mg/dL and the albumin-creatinine ratio at 17.745 mg/g), so abnormal input values can be masked. Outputs should **not** be interpreted for individual patients.
- SHAP shows model behaviour, not clinical causation.

---

## 🎓 Academic Context

Submitted in partial fulfilment of the requirements for the B.Sc. degree in Computer Science, Department of Computer Science, University of Lagos, under the supervision of Dr. Chika Ojiako (September 2026).
