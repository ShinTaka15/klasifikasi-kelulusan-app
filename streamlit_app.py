import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# LOAD MODEL
# ===========================
artefacts = joblib.load("model_kelulusan.pkl")

svm_model = artefacts["svm_model"]
scaler = artefacts["scaler"]
selector = artefacts["selector"]
ordinal_mapping = artefacts["ordinal_mapping"]
numerical_cols = artefacts["numerical_cols"]
selected_features = artefacts["selected_features"]

# ===========================
# TITLE
# ===========================
st.title("📘 Prediksi Status Remidial Siswa")
st.write("Masukkan data siswa berikut untuk memprediksi apakah siswa **Remidial** atau **Tidak Remidial**.")

# ===========================
# DEFINISI INPUT USER
# ===========================
col1, col2 = st.columns(2)

with col1:
    Hours_Studied = st.number_input("Hours_Studied", 0, 44, 5)
    Attendance = st.number_input("Attendance (%)", 0, 100, 85)
    Sleep_Hours = st.number_input("Sleep_Hours", 0, 12, 7)
    Previous_Scores = st.number_input("Previous_Scores", 0, 100, 70)
    Motivation_Level = st.selectbox("Motivation_Level", ["Low", "Medium", "High"])
    Access_to_Resources = st.selectbox("Access_to_Resources", ["Low", "Medium", "High"])
    Family_Income = st.selectbox("Family_Income", ["Low", "Medium", "High"])
    School_Type = st.selectbox("School_Type", ["Private", "Public"])
    Peer_Influence = st.selectbox("Peer_Influence", ["Negative", "Neutral", "Positive"])
    Physical_Activity = st.number_input("Physical_Activity (hours/week)", 0, 20, 3)

with col2:
    Parental_Involvement = st.selectbox("Parental_Involvement", ["Low", "Medium", "High"])
    Extracurricular_Activities = st.selectbox("Extracurricular_Activities", ["No", "Yes"])
    Internet_Access = st.selectbox("Internet_Access", ["No", "Yes"])
    Tutoring_Sessions = st.number_input("Tutoring_Sessions", 0, 10, 1)
    Teacher_Quality = st.selectbox("Teacher_Quality", ["Low", "Medium", "High"])
    Learning_Disabilities = st.selectbox("Learning_Disabilities", ["No", "Yes"])
    Parental_Education_Level = st.selectbox(
        "Parental_Education_Level", ["High School", "College", "Postgraduate"]
    )
    Distance_from_Home = st.selectbox("Distance_from_Home", ["Near", "Moderate", "Far"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

# ===========================
# BENTUKKAN DATAFRAME
# ===========================
input_data = pd.DataFrame([{
    "Hours_Studied": Hours_Studied,
    "Attendance": Attendance,
    "Parental_Involvement": Parental_Involvement,
    "Access_to_Resources": Access_to_Resources,
    "Extracurricular_Activities": Extracurricular_Activities,
    "Sleep_Hours": Sleep_Hours,
    "Previous_Scores": Previous_Scores,
    "Motivation_Level": Motivation_Level,
    "Internet_Access": Internet_Access,
    "Tutoring_Sessions": Tutoring_Sessions,
    "Family_Income": Family_Income,
    "Teacher_Quality": Teacher_Quality,
    "School_Type": School_Type,
    "Peer_Influence": Peer_Influence,
    "Physical_Activity": Physical_Activity,
    "Learning_Disabilities": Learning_Disabilities,
    "Parental_Education_Level": Parental_Education_Level,
    "Distance_from_Home": Distance_from_Home,
    "Gender": Gender
}])

# ===========================
# PREPROCESSING SESUAI MODEL
# ===========================

# 1️⃣ Map ordinal
for col in input_data.columns:
    if input_data[col].dtype == "object":
        input_data[col] = input_data[col].map(ordinal_mapping).fillna(0)

# 2️⃣ Pastikan hanya kolom numerik yang di-scale
data_scaled = input_data.copy()
data_scaled[numerical_cols] = scaler.transform(data_scaled[numerical_cols])

# 3️⃣ SelectKBest untuk memilih fitur yang digunakan model
data_final = selector.transform(data_scaled)

# ===========================
# PREDIKSI
# ===========================
if st.button("🔍 Prediksi Status Remidial"):
    prediction = svm_model.predict(data_final)[0]
    prob = svm_model.predict_proba(data_final)[0]

    status = "Tidak Remidial" if prediction == "Lulus" else "Remidial" 

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Status:** {status}")

    st.subheader("📊 Probabilitas")
    st.write(f"Probabilitas Tidak Remidial: **{prob[0]:.2f}**")
    st.write(f"Probabilitas Remidial: **{prob[1]:.2f}**")

    st.success("Prediksi berhasil diproses!")
