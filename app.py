import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from datetime import datetime


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
# LOAD TRAINED MODEL FILES (cached so this runs once, not per click)
# ============================================================

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_xgboost_model.pkl")
        scaler = joblib.load("scaler.pkl")
        selected_features = joblib.load("selected_features.pkl")
        iqr_bounds = joblib.load("iqr_bounds.pkl")
        return model, scaler, selected_features, iqr_bounds
    except FileNotFoundError as e:
        st.error(
            f"""
            ⚠️ **Required model file not found:** `{e.filename}`

            Make sure `best_xgboost_model.pkl`, `scaler.pkl`,
            `selected_features.pkl`, and `iqr_bounds.pkl` are in the
            same directory as this app before running it.
            """
        )
        st.stop()


model, scaler, selected_features, iqr_bounds = load_artifacts()


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


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


diabetes_diagnosed_label = st.sidebar.selectbox(
    "Diabetes Diagnosed",
    [
        "No",
        "Yes",
        "Borderline"
    ],
    key="diabetes_diagnosed_select",
    help="Has a doctor ever told you that you have diabetes?"
)

diabetes_mapping = {
    "Yes": 1,
    "No": 2,
    "Borderline": 3
}

# Keep the readable label for display, and a separate encoded
# value for the model — overwriting the label lost this
# distinction in the previous version.
diabetes_diagnosed = diabetes_mapping[diabetes_diagnosed_label]


ever_smoked_label = st.sidebar.selectbox(
    "Ever Smoked",
    [
        "No",
        "Yes"
    ],
    key="ever_smoked_select",
    help="Have you smoked at least 100 cigarettes in your life?"
)

smoking_mapping = {
    "Yes": 1,
    "No": 2
}

ever_smoked = smoking_mapping[ever_smoked_label]


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
    value=95.0,
    help="Estimated Glomerular Filtration Rate (mL/min/1.73m²), "
         "from a routine blood test. Normal is roughly 90 and above."
)


serum_creatinine = st.sidebar.number_input(
    "Serum Creatinine",
    min_value=0.1,
    max_value=20.0,
    value=1.0,
    help="From a basic metabolic panel blood test (mg/dL). "
         "Typical range is about 0.6–1.3 mg/dL."
)


blood_urea_nitrogen = st.sidebar.number_input(
    "Blood Urea Nitrogen",
    min_value=1.0,
    max_value=150.0,
    value=15.0,
    help="BUN, from a blood test (mg/dL). Typical range is "
         "about 7–20 mg/dL."
)


albumin_serum = st.sidebar.number_input(
    "Serum Albumin",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    help="From a blood test (g/dL). Typical range is about 3.4–5.4 g/dL."
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
    value=30.0,
    help="Urine ACR (mg/g), from a urine test. Below 30 mg/g is "
         "considered normal; above suggests kidney damage."
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
    # 6. SELECT THE 16 MODEL FEATURES
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

        # ----------------------------------------------------
        # Headline result — shown regardless of outcome
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Probability breakdown — shown for BOTH outcomes
        # (previously only appeared for the "No CKD" branch)
        # ----------------------------------------------------

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("No CKD", f"{probabilities[0] * 100:.2f}%")

        with col2:
            st.metric("CKD", f"{probabilities[1] * 100:.2f}%")

        st.progress(float(confidence / 100))

        # ----------------------------------------------------
        # Risk band — coarser, easier-to-read framing of the
        # raw CKD probability than a single percentage alone
        # ----------------------------------------------------

        ckd_probability = probabilities[1] * 100

        if ckd_probability < 20:
            risk_label, risk_color = "Low estimated risk", "green"
        elif ckd_probability < 50:
            risk_label, risk_color = "Moderate estimated risk", "orange"
        else:
            risk_label, risk_color = "High estimated risk", "red"

        st.markdown(
            f"**Risk band:** :{risk_color}[{risk_label}] "
            f"({ckd_probability:.1f}% estimated CKD probability)"
        )

        # ----------------------------------------------------
        # Interpretation — shown for BOTH outcomes
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Downloadable report
        # ----------------------------------------------------

        report_lines = [
            "CKD PREDICTION SYSTEM — RESULT REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "-" * 50,
            f"Prediction: {'CKD' if prediction == 1 else 'No CKD'}",
            f"CKD probability: {probabilities[1] * 100:.2f}%",
            f"No CKD probability: {probabilities[0] * 100:.2f}%",
            f"Risk band: {risk_label}",
            "-" * 50,
            "Patient Inputs",
            f"Age: {age}",
            f"Gender: {gender}",
            f"Ethnicity: {ethnicity}",
            f"BMI: {bmi}",
            f"Diabetes diagnosed: {diabetes_diagnosed_label}",
            f"Ever smoked: {ever_smoked_label}",
            f"Systolic / Diastolic BP: {bp_systolic} / {bp_diastolic} mmHg",
            f"eGFR: {egfr}",
            f"Serum creatinine: {serum_creatinine}",
            f"Blood urea nitrogen: {blood_urea_nitrogen}",
            f"Albumin (serum): {albumin_serum}",
            f"Albumin-creatinine ratio: {albumin_creatinine_ratio}",
            "-" * 50,
            "This report is for academic/research purposes only and is",
            "not a substitute for professional medical evaluation.",
        ]

        st.download_button(
            label="⬇️ Download Result Report (.txt)",
            data="\n".join(report_lines),
            file_name=f"ckd_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
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
                diabetes_diagnosed_label
            )

        with col2:
            st.metric(
                "Ever Smoked",
                ever_smoked_label
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

            explainer = get_shap_explainer(model)


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
    
        **Selected Features:** 16  
    
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
                0.9807,
                0.9833,
                0.9892,
                0.9611,
                0.9863,
                0.9985
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
                "Naive Bayes",
                "Logistic Regression"
            ],
            "Accuracy": [
                0.9807, 0.9791, 0.9778,
                0.9581, 0.9393, 0.8827, 0.9150
            ],
            "Precision": [
                0.9833, 0.9815, 0.9844,
                0.9585, 0.9456, 0.9572, 0.9145
            ],
            "Recall": [
                0.9892, 0.9886, 0.9838,
                0.9826, 0.9688, 0.8711, 0.9688
            ],
            "Specificity": [
                0.9611, 0.9569, 0.9638,
                0.9013, 0.8707, 0.9096, 0.7900
            ],
            "F1-Score": [
               0.9863, 0.9851, 0.9841,
               0.9704, 0.9571, 0.9121, 0.9409
            ],
            "ROC-AUC": [
               0.9985, 0.9985, 0.9957,
               0.9835, 0.9811, 0.9603, 0.9299
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
            the highest performance across every metric evaluated, including
            accuracy (98.07%), precision (98.33%), recall (98.92%),
            specificity (96.11%), and F1-score (98.63%).

            Random Forest was the closest competitor, matching XGBoost's
            ROC-AUC (99.85%) but trailing slightly on every other metric.
            XGBoost was therefore selected for deployment.
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