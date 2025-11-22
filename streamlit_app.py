import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ===========================
# CUSTOM CSS (UI MODERN)
# ===========================
st.markdown("""
<style>
    .main {
        background-color: #000000;
    }
    .card {
        padding: 20px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 30px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
st.write("Masukkan data siswa berikut untuk mengetahui apakah siswa **Remidial** atau **Tidak Remidial**.")

# ===========================
# INPUT FORM
# ===========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

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
    Parental_Education_Level = st.selectbox("Parental_Education_Level", ["High School", "College", "Postgraduate"])
    Distance_from_Home = st.selectbox("Distance_from_Home", ["Near", "Moderate", "Far"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# DATAFRAME INPUT
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
# PREPROCESSING
# ===========================
for col in input_data.columns:
    if input_data[col].dtype == "object":
        input_data[col] = input_data[col].map(ordinal_mapping).fillna(0)

data_scaled = input_data.copy()
data_scaled[numerical_cols] = scaler.transform(data_scaled[numerical_cols])

data_final = selector.transform(data_scaled)

# ===========================
# PREDIKSI UTAMA + VISUALISASI
# ===========================
if st.button("🔍 Prediksi Status Remidial"):

    prediction = svm_model.predict(data_final)[0]
    prob = svm_model.predict_proba(data_final)[0]

    status = "Tidak Remidial" if prediction == "Lulus" else "Remidial"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Status:** {status}")
    st.write(f"Probabilitas Tidak Remidial: **{prob[0]:.2f}**")
    st.write(f"Probabilitas Remidial: **{prob[1]:.2f}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # =======================
    # VISUALISASI PROBABILITAS
    # =======================
    st.subheader("📊 Probability Chart")

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(["Tidak Remidial", "Remidial"], prob)
    plt.title("Probability Distribution")
    st.pyplot(fig)

    # =======================
    # CONFUSION MATRIX (TRAIN DATA)
    # =======================
    st.subheader("📉 Confusion Matrix (Data Training)")

    try:
        y_true = artefacts["y_true"]
        y_pred_train = artefacts["y_pred_train"]

        cm = confusion_matrix(y_true, y_pred_train)

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig_cm)
    except:
        st.info("Confusion Matrix tidak tersedia karena data y_true tidak disimpan.")

    # =======================
    # ROC CURVE
    # =======================
    st.subheader("📈 ROC Curve")

    try:
        fpr, tpr, _ = roc_curve(y_true, artefacts["y_prob_train"])
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig_roc)
    except:
        st.info("ROC Curve tidak tersedia karena probabilitas training tidak disimpan.")

    # =======================
    # FEATURE IMPORTANCE (SelectKBest)
    # =======================
    st.subheader("📌 Feature Importance (SelectKBest Scores)")

    try:
        scores = selector.scores_
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        ax_imp.barh(selected_features, scores[:len(selected_features)])
        plt.title("Feature Scores")
        st.pyplot(fig_imp)
    except:
        st.info("Feature importance tidak tersedia.")

    # DEBUG LOG
    with st.expander("🔧 Debug Log"):
        st.write("Input awal:")
        st.write(input_data)
        st.write("Scaled:")
        st.write(data_scaled)
        st.write("Final feature set:")
        st.write(data_final)

