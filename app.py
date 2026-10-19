"""
Step 3: Streamlit web app for cardiovascular disease risk prediction.
Run locally:  streamlit run app/app.py
Deploy free:  Streamlit Community Cloud (see README.md) - works in any phone / laptop browser.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


bundle = load_model()
model, features = bundle["model"], bundle["features"]

st.title("❤️ Cardiovascular Disease Risk Predictor")
st.caption(f"Model: **{bundle['name']}** · trained on 70,000 patient records. "
           "Educational project - not a medical diagnosis.")

tab_pred, tab_perf, tab_data = st.tabs(["Predict", "Model performance", "Data insights"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 18, 100, 50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", 130, 210, 165)
        weight = st.number_input("Weight (kg)", 35.0, 200.0, 70.0)
        ap_hi = st.number_input("Systolic BP (ap_hi)", 70, 250, 120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 200, 80)
    with col2:
        chol = st.selectbox("Cholesterol", [1, 2, 3],
                            format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
        gluc = st.selectbox("Glucose", [1, 2, 3],
                            format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Drinks alcohol")
        active = st.checkbox("Physically active", value=True)

    if st.button("Predict risk", type="primary", use_container_width=True):
        if ap_hi <= ap_lo:
            st.error("Systolic pressure must be higher than diastolic pressure.")
        else:
            row = pd.DataFrame([{
                "age_years": age, "gender": 1 if gender == "Female" else 2,
                "height": height, "weight": weight,
                "bmi": weight / (height / 100) ** 2,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                "smoke": int(smoke), "alco": int(alco), "active": int(active)}])[features]
            p = float(model.predict_proba(row)[0, 1])
            st.metric("Estimated probability of cardiovascular disease", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            if p >= 0.6:
                st.error("High risk - please consult a doctor.")
            elif p >= 0.4:
                st.warning("Moderate risk - keep monitoring your health.")
            else:
                st.success("Low risk.")
            st.caption(f"BMI used by the model: {row['bmi'][0]:.1f}")

# ------------------------------------------------------------------ Performance
with tab_perf:
    res = ROOT / "outputs" / "model_comparison.csv"
    if res.exists():
        st.subheader("Model comparison (test set)")
        st.dataframe(pd.read_csv(res), use_container_width=True)
    for img, cap in [("model_comparison.png", "Metric comparison"), ("roc_curves.png", "ROC curves"),
                     ("confusion_matrix.png", "Confusion matrix - best model"),
                     ("feature_importance.png", "Feature importance")]:
        p = ROOT / "outputs" / img
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)

# ------------------------------------------------------------------ Data insights
with tab_data:
    for img, cap in [("eda.png", "Age, blood pressure and BMI vs disease"),
                     ("correlation.png", "Correlation matrix")]:
        p = ROOT / "outputs" / img
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)
    rep = ROOT / "outputs" / "cleaning_report.csv"
    if rep.exists():
        st.subheader("Data cleaning summary")
        st.dataframe(pd.read_csv(rep, index_col=0), use_container_width=True)
