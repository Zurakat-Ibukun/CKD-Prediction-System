import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()


# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================

model = joblib.load("best_xgboost_model.pkl")

scaler = joblib.load("scaler.pkl")

selected_features = joblib.load("selected_features.pkl")

iqr_bounds = joblib.load("iqr_bounds.pkl")


# ============================================================
# MAIN PAGE TITLE
# ============================================================

st.markdown(
"""
# 🩺 Chronic Kidney Disease Prediction System

### Machine Learning-Based Early Detection Using the NHANES Dataset

This application uses machine learning to estimate the likelihood
of Chronic Kidney Disease (CKD) based on selected demographic,
clinical, and laboratory features.

---
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "kidney.png",
    width=150
)

st.sidebar.title("Patient Information")


# ============================================================
# DEMOGRAPHICS
# ============================================================

st.sidebar.subheader("👤 Demographics")


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)


gender = st.sidebar.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ],
    key="gender_select"
)


ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other/Multiracial"
    ],
    key="ethnicity_select"
)

education_level = st.sidebar.number_input(
    "Education Level",
    min_value=1.0,
    max_value=5.0,
    value=3.0
)


poverty_income_ratio = st.sidebar.number_input(
    "Poverty Income Ratio",
    min_value=0.0,
    max_value=20.0,
    value=2.0
)


weight_kg = st.sidebar.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=250.0,
    value=70.0
)


height_cm = st.sidebar.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=220.0,
    value=170.0
)


bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=22.0
)


# ============================================================
# MEDICAL HISTORY
# ============================================================

st.sidebar.subheader("🩺 Medical History")


diabetes_diagnosed = st.sidebar.selectbox(
    "Diabetes Diagnosed",
    [
        "No",
        "Yes",
        "Borderline"
    ],
    key="diabetes_diagnosed_select"
)

diabetes_mapping = {
    "Yes": 1,
    "No": 2,
    "Borderline": 3
}

diabetes_diagnosed = diabetes_mapping[diabetes_diagnosed]


ever_smoked = st.sidebar.selectbox(
    "Ever Smoked",
    [
        "No",
        "Yes"
    ],
    key="ever_smoked_select"
)

smoking_mapping = {
    "Yes": 1,
    "No": 2
}

ever_smoked = smoking_mapping[ever_smoked]


# ============================================================
# VITAL SIGNS
# ============================================================

st.sidebar.subheader("❤️ Vital Signs")


bp_systolic = st.sidebar.number_input(
    "Systolic Blood Pressure",
    min_value=60.0,
    max_value=250.0,
    value=120.0
)


bp_diastolic = st.sidebar.number_input(
    "Diastolic Blood Pressure",
    min_value=30.0,
    max_value=150.0,
    value=80.0
)


# ============================================================
# LABORATORY TESTS
# ============================================================

st.sidebar.subheader("🧪 Laboratory Tests")


egfr = st.sidebar.number_input(
    "eGFR",
    min_value=1.0,
    max_value=200.0,
    value=95.0
)


serum_creatinine = st.sidebar.number_input(
    "Serum Creatinine",
    min_value=0.1,
    max_value=20.0,
    value=1.0
)


blood_urea_nitrogen = st.sidebar.number_input(
    "Blood Urea Nitrogen",
    min_value=1.0,
    max_value=150.0,
    value=15.0
)


albumin_serum = st.sidebar.number_input(
    "Serum Albumin",
    min_value=0.0,
    max_value=10.0,
    value=4.0
)


phosphorus = st.sidebar.number_input(
    "Phosphorus",
    min_value=1.0,
    max_value=10.0,
    value=3.5
)


bicarbonate = st.sidebar.number_input(
    "Bicarbonate",
    min_value=5.0,
    max_value=50.0,
    value=24.0
)


calcium = st.sidebar.number_input(
    "Calcium",
    min_value=5.0,
    max_value=15.0,
    value=9.5
)


uric_acid = st.sidebar.number_input(
    "Uric Acid",
    min_value=1.0,
    max_value=20.0,
    value=5.0
)


urine_creatinine = st.sidebar.number_input(
    "Urine Creatinine",
    min_value=1.0,
    max_value=500.0,
    value=100.0
)


urine_albumin = st.sidebar.number_input(
    "Urine Albumin",
    min_value=0.0,
    max_value=1000.0,
    value=20.0
)


albumin_creatinine_ratio = st.sidebar.number_input(
    "Albumin Creatinine Ratio",
    min_value=0.0,
    max_value=1000.0,
    value=30.0
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔍 Predict CKD",
    use_container_width=True
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Prediction",
        "📋 Patient Summary",
        "🧠 SHAP Explainability",
        "ℹ️ Model Information"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # 1. CREATE ALL 23 FEATURES
    # ========================================================

    patient_data = {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "poverty_income_ratio": poverty_income_ratio,
        "bmi": bmi,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "serum_creatinine": serum_creatinine,
        "blood_urea_nitrogen": blood_urea_nitrogen,
        "albumin_serum": albumin_serum,
        "phosphorus": phosphorus,
        "bicarbonate": bicarbonate,
        "calcium": calcium,
        "uric_acid": uric_acid,
        "urine_creatinine": urine_creatinine,
        "urine_albumin": urine_albumin,
        "albumin_creatinine_ratio": albumin_creatinine_ratio,
        "diabetes_diagnosed": diabetes_diagnosed,
        "ever_smoked": ever_smoked,
        "egfr": egfr
    }

    input_data = pd.DataFrame([patient_data])


    # ========================================================
    # 2. ENCODE CATEGORICAL VARIABLES
    # ========================================================

    input_data["gender"] = input_data["gender"].map({
        "Male": 1,
        "Female": 0
    })

    ethnicity_mapping = {
    "Mexican American": 0,
    "Non-Hispanic Asian": 1,
    "Non-Hispanic Black": 2,
    "Non-Hispanic White": 3,
    "Other Hispanic": 4,
    "Other/Multiracial": 5
    }

    input_data["ethnicity"] = input_data[
        "ethnicity"
    ].map(ethnicity_mapping)


    # ========================================================
    # 3. PREPARE DATA FOR SCALER
    # ========================================================

    scaler_features = list(scaler.feature_names_in_)

    input_data = input_data[scaler_features]

    # Apply the same IQR bounds used during training
    for column, bounds in iqr_bounds.items():
        if column in input_data.columns:
            input_data[column] = input_data[column].clip(
            lower=bounds["lower"],
            upper=bounds["upper"]
            )


    # ========================================================
    # 4. SCALE THE 23 FEATURES
    # ========================================================

    # Scale using the scaler fitted on X_capped
    input_scaled = scaler.transform(input_data)


    # ========================================================
    # 5. CONVERT SCALED DATA TO DATAFRAME
    # ========================================================

    input_scaled = pd.DataFrame(
        input_scaled,
        columns=scaler_features
    )


    # ========================================================
    # 6. SELECT THE 17 MODEL FEATURES
    # ========================================================

    model_input = input_scaled[
        selected_features
    ]


    # ========================================================
    # 7. MAKE PREDICTION
    # ========================================================

    prediction = model.predict(
        model_input
    )[0]


    # ========================================================
    # 8. GET PROBABILITIES
    # ========================================================

    probabilities = model.predict_proba(
        model_input
    )[0]

    confidence = (
        probabilities[int(prediction)] * 100
    )


    # ========================================================
    # 📊 TAB 1 — PREDICTION
    # ========================================================

    with tab1:

        st.subheader(
            "🩺 CKD Prediction Result"
        )

    if prediction == 1:

        st.error(
            f"""

            ## ⚠️ Model Prediction: CKD
             
            **Predicted Probability: {confidence:.2f}%**
            """
        )

    else:
        st.success(
        f"""
        ## ✅ Model Prediction: No CKD

        **Predicted Probability: {confidence:.2f}%**
        """
        )

        
        st.subheader(
            "Prediction Probability"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "No CKD",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "CKD",
                f"{probabilities[1] * 100:.2f}%"
            )


        st.progress(
            float(confidence / 100)
        )

        st.markdown("### 📌 Prediction Interpretation")

    if prediction == 1:
        st.info(
            f"""
            Based on the information provided, the model predicts the
            **CKD class** with a probability of
            **{probabilities[1] * 100:.2f}%**.
            """
        )
    else:
        st.info(
            f"""
            Based on the information provided, the model predicts the
            **No CKD class** with a probability of
            **{probabilities[0] * 100:.2f}%**.
            """
        )

        st.warning(
            """
            ⚠️ **Research Disclaimer**

            This system provides a machine learning-based prediction for
            research and educational purposes only. The result should not
            be interpreted as a medical diagnosis or used as a substitute
            for professional medical advice.
            """
        )


    # ========================================================
    # 📋 TAB 2 — PATIENT SUMMARY
    # ========================================================

    with tab2:

        st.subheader(
            "📋 Patient Information Summary"
        )

        st.markdown(
            "### 👤 Demographics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Age", age)

        with col2:
            st.metric("Gender", gender)

        with col3:
            st.metric("BMI", f"{bmi:.1f}")


        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Weight", f"{weight_kg:.1f} kg")

        with col2:
            st.metric("Height", f"{height_cm:.1f} cm")

        with col3:
            st.metric("Ethnicity", ethnicity)


        st.markdown(
            "### 🩺 Medical History"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Diabetes",
                diabetes_diagnosed
            )

        with col2:
            st.metric(
                "Ever Smoked",
                ever_smoked
            )


        st.markdown(
            "### ❤️ Vital Signs"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Systolic BP",
                f"{bp_systolic:.0f} mmHg"
            )

        with col2:
            st.metric(
                "Diastolic BP",
                f"{bp_diastolic:.0f} mmHg"
            )


        st.markdown(
            "### 🧪 Laboratory Results"
        )

        lab_data = pd.DataFrame({
            "Laboratory Test": [
                "eGFR",
                "Serum Creatinine",
                "Blood Urea Nitrogen",
                "Serum Albumin",
                "Phosphorus",
                "Bicarbonate",
                "Calcium",
                "Uric Acid",
                "Urine Creatinine",
                "Urine Albumin",
                "Albumin Creatinine Ratio"
            ],

            "Value": [
                egfr,
                serum_creatinine,
                blood_urea_nitrogen,
                albumin_serum,
                phosphorus,
                bicarbonate,
                calcium,
                uric_acid,
                urine_creatinine,
                urine_albumin,
                albumin_creatinine_ratio
            ]
        })


        st.dataframe(
            lab_data,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # 🧠 TAB 3 — SHAP EXPLAINABILITY
    # ========================================================

    with tab3:

        st.subheader(
            "🧠 SHAP Model Explainability"
        )

        st.write(
            """
            SHAP (SHapley Additive exPlanations) helps explain
            how each feature contributed to the model's
            prediction.
            """
        )


        try:

            # ------------------------------------------------
            # Create SHAP explainer
            # ------------------------------------------------

            explainer = shap.TreeExplainer(
                model
            )


            # ------------------------------------------------
            # Calculate SHAP values
            # ------------------------------------------------

            shap_values = explainer.shap_values(
                model_input
            )


            # ------------------------------------------------
            # Handle different SHAP output formats
            # ------------------------------------------------

            if isinstance(shap_values, list):

                shap_for_prediction = shap_values[
                    int(prediction)
                ][0]

            else:

                shap_for_prediction = shap_values[0]


            # ------------------------------------------------
            # Create feature explanation table
            # ------------------------------------------------

            shap_df = pd.DataFrame({
                "Feature": selected_features,
                "Value": model_input.iloc[0].values,
                "SHAP Value": shap_for_prediction
            })


            # Absolute importance

            shap_df["Importance"] = (
                shap_df["SHAP Value"].abs()
            )


            shap_df = shap_df.sort_values(
                "Importance",
                ascending=False
            )


            # ------------------------------------------------
            # Display top features
            # ------------------------------------------------

            st.markdown(
                "### 🔍 Features Influencing the Prediction"
            )


            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "Value",
                        "SHAP Value"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            st.markdown(
                "### 📊 Feature Contribution"
            )


            chart_data = shap_df.set_index(
                "Feature"
            )["SHAP Value"]


            st.bar_chart(
                chart_data
            )


            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.markdown(

                "### 💡 Interpretation"

            )

            top_feature = shap_df.iloc[0]

            if top_feature["SHAP Value"] > 0:
                st.info(
                    f"""
                    **{top_feature['Feature']}** had the strongest
                    positive contribution toward the model's predicted
                    class for this patient.
                    """
                )

            else:
                st.info(
                    f"""
                    **{top_feature['Feature']}** had the strongest
                    negative contribution toward the model's predicted
                    class for this patient.
                    """
                )

        except Exception as e:
            st.error(
                "SHAP explanation could not be generated."
            )

            st.code(
                str(e)
            )


    # ========================================================
    # ℹ️ TAB 4 — MODEL INFORMATION
    # ========================================================

    with tab4:
        st.subheader("ℹ️ Model Information")

        st.markdown("""
        ### 🧠 Machine Learning Model
        
        **Dataset:** CKD_NHANES_2021_2023  
    
        **Dataset Size:** 11,933 records  
    
        **Original Variables:** 29  

        **Predictors After Preprocessing:** 23  
    
        **Feature Selection:** Boruta  
    
        **Selected Features:** 17  
    
        **Best Performing Model:** XGBoost  
    
        **Explainability Method:** SHAP

        """)

        st.markdown("### 🔬 Selected Features")

        feature_df = pd.DataFrame({
        "Feature": selected_features
        })

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### ⚙️ Prediction Pipeline"
        )


        st.code(
            """
Patient Data
     ↓
Data Preprocessing
     ↓
Categorical Encoding
     ↓
IQR Outlier Capping
     ↓
Feature Scaling
     ↓
Feature Selection
     ↓
XGBoost Classifier
     ↓
CKD Prediction
            """
        )

        st.markdown("### 📊 Deployed Model Performance")

        xgb_metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score",
                "ROC-AUC"
            ],
            "XGBoost": [
                0.9786,
                0.9815,
                0.9880,
                0.9569,
                0.9848,
                0.9977
            ]
        })

        xgb_metrics["XGBoost"] = xgb_metrics["XGBoost"].apply(
            lambda x: f"{x:.2%}"
        )

        st.dataframe(
            xgb_metrics,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 🔬 Model Comparison")

        comparison = pd.DataFrame({
            "Model": [
                "XGBoost",
                "Random Forest",
                "Decision Tree",
                "SVM",
                "KNN",
                "Logistic Regression",
                "Naive Bayes"
            ],
            "Accuracy": [
                0.9786, 0.9778, 0.9686,
                0.9585, 0.9342, 0.9133, 0.8869
            ],
            "Precision": [
                0.9815, 0.9792, 0.9767,
                0.9574, 0.9473, 0.9134, 0.9605
            ],
            "Recall": [
                0.9880, 0.9892, 0.9784,
                0.9844, 0.9592, 0.9676, 0.8741
            ],
            "Specificity": [
                0.9569, 0.9513, 0.9458,
                0.8985, 0.8762, 0.7872, 0.9166
            ],
            "F1-Score": [
               0.9848, 0.9842, 0.9775,
               0.9707, 0.9532, 0.9397, 0.9153
            ],
            "ROC-AUC": [
               0.9977, 0.9975, 0.9621,
               0.9860, 0.9708, 0.9289, 0.9598
            ]
        })

        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "Specificity": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 🏆 Model Selection")

        st.info(
            """
            **XGBoost was selected as the deployed model** because it achieved
            the highest overall performance among the evaluated models, including
            the highest accuracy (97.86%), precision (98.15%), F1-score (98.48%),
            specificity (95.69%), and ROC-AUC (99.77%).

            Although Random Forest achieved a slightly higher recall (98.92%)
            compared with XGBoost (98.80%), XGBoost demonstrated the strongest
            overall performance and was therefore selected for deployment.
            """
        )

        st.markdown("### 📚 About This Project")

        st.write(
        """
        This application was developed as part of a research project
        on the comparative analysis of machine learning algorithms
        for the early detection of Chronic Kidney Disease.

        The system uses data from the National Health and Nutrition
        Examination Survey (NHANES), applies preprocessing and feature
        selection, and uses the selected machine learning model to
        generate predictions.

        SHAP explainability is incorporated to improve the
        interpretability of the model's predictions.
        """
        )

        st.markdown(
            "### ⚠️ Disclaimer"
        )


        st.warning(
            """
            This application is intended for academic,
            research, and educational purposes. The prediction
            produced by the system should not be used as a
            substitute for professional medical evaluation
            or diagnosis.
            """
        )