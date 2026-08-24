# 🩺 Chronic Kidney Disease Prediction System

### Machine Learning-Based Early Detection Using the NHANES Dataset

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)

---

## 📌 Project Overview

Chronic Kidney Disease (CKD) is a major health condition that
can progress gradually without obvious symptoms during its early
stages.

This project presents a machine learning-based system for the
early detection of Chronic Kidney Disease using data from the
**National Health and Nutrition Examination Survey (NHANES)**.

The system compares multiple machine learning algorithms and
selects the best-performing model for deployment. The final
application was developed using **Streamlit** and incorporates
**SHAP explainability** to provide insight into the factors
contributing to individual predictions.

---

## 🎯 Project Objective

The main objective of this project is to develop and evaluate
machine learning models for the early detection of Chronic Kidney
Disease and deploy the best-performing model as an interactive
prediction system.

The project specifically aims to:

- Preprocess and prepare NHANES data for machine learning.
- Identify relevant predictors of CKD.
- Apply feature selection using the Boruta algorithm.
- Compare different machine learning algorithms.
- Evaluate models using multiple performance metrics.
- Select the best-performing model for deployment.
- Apply SHAP for model explainability.
- Develop an interactive web-based CKD prediction system.

---

## 📊 Dataset

The project uses data derived from the:

**National Health and Nutrition Examination Survey (NHANES)**

The dataset contains demographic, clinical, lifestyle, and
laboratory variables that can be used to investigate factors
associated with Chronic Kidney Disease.

---

## 🧪 Feature Selection

Feature selection was performed using the **Boruta feature
selection algorithm**.

The original preprocessing pipeline contained **23 features**.

After feature selection, **17 features** were retained for the
final machine learning model.

### Selected Features

- Age
- Gender
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

---

## 🤖 Machine Learning Models

The following machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Naive Bayes
6. Support Vector Machine (SVM)
7. XGBoost

---

## 📈 Model Performance Comparison

The models were evaluated using:

- Accuracy
- Precision
- Recall
- Specificity
- F1-Score
- ROC-AUC

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **97.86%** | **98.15%** | 98.80% | **95.69%** | **98.48%** | **99.77%** |
| Random Forest | 97.78% | 97.92% | **98.92%** | 95.13% | 98.42% | 99.75% |
| Decision Tree | 96.86% | 97.67% | 97.84% | 94.58% | 97.75% | 96.21% |
| SVM | 95.85% | 95.74% | 98.44% | 89.85% | 97.07% | 98.60% |
| KNN | 93.42% | 94.73% | 95.92% | 87.62% | 95.32% | 97.08% |
| Logistic Regression | 91.33% | 91.34% | 96.76% | 78.72% | 93.97% | 92.89% |
| Naive Bayes | 88.69% | 96.05% | 87.41% | 91.66% | 91.53% | 95.98% |

---

## 🏆 Model Selection

**XGBoost was selected as the deployed model** because it achieved
the strongest overall performance among the evaluated algorithms.

XGBoost achieved:

- **Accuracy:** 97.86%
- **Precision:** 98.15%
- **Recall:** 98.80%
- **Specificity:** 95.69%
- **F1-Score:** 98.48%
- **ROC-AUC:** 99.77%

Random Forest achieved a slightly higher recall of **98.92%**
compared with XGBoost's **98.80%**.

However, XGBoost demonstrated stronger overall performance across
the other major evaluation metrics and was therefore selected for
deployment.

---

## 🧠 Explainable AI with SHAP

The system incorporates **SHAP (SHapley Additive exPlanations)**
to improve the interpretability of the machine learning model.

SHAP helps explain:

- Which features influenced a prediction.
- Whether a feature increased or decreased the predicted CKD risk.
- The relative contribution of individual features.
- How the model arrived at a particular prediction.

This provides greater transparency and makes the machine learning
system easier to interpret.

---

## 💻 Application Features

The deployed Streamlit application provides:

### 📊 Prediction

Users can enter patient information and obtain:

- CKD prediction
- No CKD prediction
- Prediction probabilities
- Model confidence

### 📋 Patient Summary

The application displays the entered:

- Demographic information
- Medical history
- Vital signs
- Laboratory measurements

### 🧠 SHAP Explainability

The application provides visual explanations of the model's
prediction and identifies the features contributing to the result.

### ℹ️ Model Information

The application provides information about:

- The dataset
- Feature selection
- Machine learning models
- Model performance
- Selected XGBoost model
- Research methodology

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| XGBoost | Final prediction model |
| SHAP | Model explainability |
| Streamlit | Web application |
| Matplotlib | Data visualization |
| Joblib | Model serialization |

---

## 📁 Project Structure

```text
ckd-prediction-system-nhanes/
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
