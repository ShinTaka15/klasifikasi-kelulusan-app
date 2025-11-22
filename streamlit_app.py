import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# LOAD MODEL DAN PREPROCESSING
# ================================
artefacts = joblib.load("model_kelulusan.pkl")

model = artefacts["svm_model"]
scaler = artefacts["scaler"]
selector = artefacts["selector"]
ordinal_mapping = artefacts["ordinal_mapping"]
numerical_cols = artefacts["numerical_cols"]

# ================================
# FUNGSI PREPROCESSING
# ================================
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Mapping ordinal
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(ordinal_mapping).fillna(0)

    # Isi NaN (jika ada)
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Scaling
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # SelectKBest transform
    df_fs = selector.transform(df)

    return df_fs

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Prediksi Kelulusan Ujian", layout="wide")
st.title("🎓 Prediksi Kelulusan Ujian Siswa")

st.write("Masukkan data siswa berikut:")

# ================================
# MASUKAN FITUR
# ================================
col1, col2 = st.columns(2)

with col1:
    Study_Time = st.selectbox("Study Time", ["Low", "Medium", "High"])
    Attendance = st.selectbox("Attendance", ["Low", "Medium", "High"])
    School_Type = st.selectbox("School Type", ["Private", "Public"])
    Parental_Education = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"])

with col2:
    Distance_To_School = st.selectbox("Distance to School", ["Near", "Moderate", "Far"])
    Health_Status = st.selectbox("Health Status", ["Negative", "Neutral", "Positive"])
    Participation = st.selectbox("Participation", ["No", "Yes"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

# Jika Anda punya fitur numerik lain, tambahkan di sini:
# Contoh: 
# age = st.number_input("Age", 10, 25)

# ================================
# PREDIKSI
# ================================
if st.button("Prediksi"):
    input_data = {
        "Study_Time": Study_Time,
        "Attendance": Attendance,
        "School_Type": School_Type,
        "Parental_Education": Parental_Education,
        "Distance_To_School": Distance_To_School,
        "Health_Status": Health_Status,
        "Participation": Participation,
        "Gender": Gender,
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0]

    st.subheader("📌 Hasil Prediksi")
    if prediction == 1:
        st.success(f"💡 Siswa **BERPOTENSI REMIDIAL** (Probabilitas: {proba[1]:.2f})")
    else:
        st.info(f"✅ Siswa DIPREDIKSI **LULUS TANPA REMIDIAL** (Probabilitas: {proba[0]:.2f})")
