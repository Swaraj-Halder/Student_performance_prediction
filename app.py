"""Streamlit dashboard for Student Performance Prediction and Academic Risk Analysis."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_models.joblib"
DATA_PATH = ROOT / "data" / "student_performance.csv"

st.set_page_config(page_title="Student Performance | SEC C", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction & Academic Risk Analysis")
st.caption("SEC C • Data Science and Machine Learning mini-project")

if not MODEL_PATH.exists():
    st.warning("The trained models are not available yet. Run the training command below in your VS Code terminal.")
    st.code("python train.py", language="bash")
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
features = bundle["features"]
metrics = bundle["metrics"]

with st.sidebar:
    st.header("Project information")
    st.write("**Models:** Linear Regression, Logistic Regression, KNN and K-Means")
    st.write("**Pass mark:** 40 / 100")
    st.caption("Use approved student data only. This demo dataset is synthetic.")

tab_predict, tab_batch, tab_analysis, tab_about = st.tabs(
    ["Individual prediction", "Batch analysis", "Dataset analysis", "About & metrics"]
)

with tab_predict:
    st.subheader("Enter student academic details")
    c1, c2 = st.columns(2)
    with c1:
        student_id = st.text_input("Student ID", value="SEC-C-NEW")
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)
        previous_marks = st.slider("Previous marks (%)", 0, 100, 65)
    with c2:
        assignment_marks = st.slider("Assignment marks (%)", 0, 100, 70)
        internal_marks = st.slider("Internal assessment marks (%)", 0, 100, 68)

    if st.button("Predict academic performance", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_marks": previous_marks,
            "assignment_marks": assignment_marks,
            "internal_marks": internal_marks,
        }], columns=features)

        predicted_marks = float(np.clip(bundle["regression"].predict(row)[0], 0, 100))
        pass_probability = float(bundle["logistic"].predict_proba(row)[0, 1])
        pass_prediction = int(bundle["logistic"].predict(row)[0])
        cluster_id = int(bundle["kmeans"].predict(bundle["scaler"].transform(row))[0])
        segment = bundle["cluster_labels"][cluster_id]

        # Risk level combines predicted result and model pass probability.
        if predicted_marks < 40 or pass_probability < 0.40:
            risk = "High"
        elif predicted_marks < 55 or pass_probability < 0.70:
            risk = "Medium"
        else:
            risk = "Low"

        st.divider()
        st.subheader(f"Prediction results — {student_id}")
        a, b, c, d = st.columns(4)
        a.metric("Predicted final marks", f"{predicted_marks:.1f}/100")
        b.metric("Pass probability", f"{pass_probability:.1%}")
        c.metric("Pass / Fail", "PASS" if pass_prediction else "FAIL")
        d.metric("Academic risk", risk)
        st.progress(min(max(pass_probability, 0.0), 1.0), text="Estimated probability of passing")
        st.info(f"Performance segment: **{segment}**")

        distances, indices = bundle["knn"].kneighbors(bundle["scaler"].transform(row))
        similar = bundle["training_data"].iloc[indices[0]].copy()
        similar.insert(0, "Student ID", bundle["training_student_ids"].iloc[indices[0]].values)
        similar["Actual final marks"] = bundle["training_marks"].iloc[indices[0]].values
        st.markdown("**Five students with similar academic profiles (KNN)**")
        st.dataframe(similar.round(1), use_container_width=True, hide_index=True)

        if risk == "High":
            st.warning("Suggested action: arrange an academic check-in, review attendance, and provide targeted study support.")
        elif risk == "Medium":
            st.info("Suggested action: monitor progress and encourage regular practice and assignment completion.")
        else:
            st.success("Suggested action: maintain current study habits and continue monitoring progress.")

with tab_batch:
    st.subheader("Upload a CSV for batch predictions")
    st.write("Required columns: " + ", ".join(features))
    uploaded = st.file_uploader("Choose student CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [col for col in features if col not in batch.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            clean = batch[features].apply(pd.to_numeric, errors="coerce")
            clean = clean.fillna(bundle["training_data"][features].median())
            predicted = np.clip(bundle["regression"].predict(clean), 0, 100)
            probabilities = bundle["logistic"].predict_proba(clean)[:, 1]
            passes = bundle["logistic"].predict(clean)
            clusters = bundle["kmeans"].predict(bundle["scaler"].transform(clean))
            result = batch.copy()
            result["Predicted final marks"] = predicted.round(1)
            result["Pass probability (%)"] = (probabilities * 100).round(1)
            result["Pass/Fail"] = np.where(passes == 1, "Pass", "Fail")
            result["Performance segment"] = [bundle["cluster_labels"][int(x)] for x in clusters]
            result["Risk level"] = [
                "High" if (m < 40 or p < .40) else "Medium" if (m < 55 or p < .70) else "Low"
                for m, p in zip(predicted, probabilities)
            ]
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download prediction results",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="student_prediction_results.csv",
                mime="text/csv",
            )

with tab_analysis:
    st.subheader("Explore the demonstration dataset")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.caption("The included sample records are synthetic and are not real student records.")
        x, y, z = st.columns(3)
        x.metric("Students in sample", len(data))
        y.metric("Average final marks", f"{data['final_exam_marks'].mean():.1f}")
        z.metric("Average attendance", f"{data['attendance'].mean():.1f}%")
        st.markdown("**Academic data preview**")
        st.dataframe(data.head(20), use_container_width=True, hide_index=True)
        st.markdown("**Feature relationships with final examination marks**")
        corr = data[features + ["final_exam_marks"]].corr(numeric_only=True)["final_exam_marks"].drop("final_exam_marks").sort_values()
        st.bar_chart(corr)
        st.markdown("**Final marks distribution**")
        st.bar_chart(data["final_exam_marks"].round(0).value_counts().sort_index())
    else:
        st.info("Run python train.py to generate the sample dataset.")

with tab_about:
    st.subheader("Model evaluation")
    st.markdown("**Linear Regression — final examination marks**")
    st.json(metrics["regression"])
    st.markdown("**Logistic Regression — pass/fail**")
    st.json(metrics["classification"])
    st.markdown("""
    **How the models are used**
    - Linear Regression estimates final examination marks.
    - Logistic Regression estimates pass/fail and pass probability.
    - KNN retrieves students with similar academic feature values.
    - K-Means groups students into At-Risk, Average and High Performing segments.

    **Important:** Academic risk and performance groups are model-generated indicators,
    not definitive judgments about a student. Use them only to guide supportive review.
    """)
