
# ===========================
# 📌 IMPORT LIBRARY
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# ===========================
# 1️⃣ LOAD DATASET
# ===========================
df = pd.read_csv("StudentPerformanceFactors.csv")

# ===========================
# 2️⃣ DATA UNDERSTANDING
# ===========================
print("Beberapa baris pertama dataset:")
print(df.head())
print("\nInformasi dataset:")
print(df.info())
print("\nDistribusi target:")
print(df['Status_Remidial'].value_counts())

# Visualisasi distribusi target
plt.figure(figsize=(6,4))
sns.countplot(x='Status_Remidial', data=df)
plt.title("Distribusi Status Remidial")
plt.show()

# ===========================
# 3️⃣ PREPROCESSING
# ===========================
X = df.drop(['Exam_Score', 'Status_Remidial'], axis=1)
y = df['Status_Remidial']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Mapping ordinal
ordinal_mapping = {
    'Low':0, 'Medium':1, 'High':2,
    'No':0, 'Yes':1,
    'Private':0, 'Public':1,
    'Negative':0, 'Neutral':1, 'Positive':2,
    'Near':0, 'Moderate':1, 'Far':2,
    'Male':0, 'Female':1,
    'High School':0, 'College':1, 'Postgraduate':2
}

ordinal_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

for col in ordinal_features:
    X_train[col] = X_train[col].map(ordinal_mapping).fillna(0)
    X_test[col] = X_test[col].map(ordinal_mapping).fillna(0)

# Isi NaN
for col in X_train.select_dtypes(include=['int64','float64']).columns:
    X_train[col].fillna(X_train[col].median(), inplace=True)
    X_test[col].fillna(X_train[col].median(), inplace=True)

# Scaling numerik
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ===========================
# 4️⃣ BALANCING DATA DENGAN SMOTE
# ===========================
print("Distribusi kelas sebelum SMOTE:\n", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Distribusi kelas setelah SMOTE:\n", pd.Series(y_train_bal).value_counts())

# ===========================
# 🔹 PCA SETELAH SMOTE (UNTUK VISUALISASI)
# ===========================
pca_viz = PCA(n_components=2)
X_train_bal_pca = pca_viz.fit_transform(X_train_bal)

plt.figure(figsize=(7,6))
sns.scatterplot(x=X_train_bal_pca[:,0], y=X_train_bal_pca[:,1],
                hue=y_train_bal, palette="Set2", alpha=0.7)
plt.title("Visualisasi PCA 2D Setelah SMOTE (Sebelum Feature Selection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# ===========================
# 5️⃣ FEATURE SELECTION (SelectKBest)
# ===========================
selector = SelectKBest(score_func=f_classif, k=8)
X_train_bal_fs = selector.fit_transform(X_train_bal, y_train_bal)
X_test_fs = selector.transform(X_test)

selected_features = X_train_bal.columns[selector.get_support()]
print("\nFitur Terpilih oleh SelectKBest:")
print(selected_features)

# ===========================
# 6️⃣ MODELING (SVM)
# ===========================
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_bal_fs, y_train_bal)

y_pred_train = svm_model.predict(X_train_bal_fs)
y_pred_test = svm_model.predict(X_test_fs)

# ===========================
# 📌 EVALUASI MODEL
# ===========================
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrices
cm_train = confusion_matrix(y_train_bal, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

train_acc = accuracy_score(y_train_bal, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("=== Confusion Matrix (Training) ===")
ConfusionMatrixDisplay(cm_train).plot(cmap="Blues")
plt.title("SVM - Training Set")
plt.show()

print("=== Confusion Matrix (Test) ===")
ConfusionMatrixDisplay(cm_test).plot(cmap="Blues")
plt.title("SVM - Test Set")
plt.show()

print(f"\nAkurasi Training Set : {train_acc:.4f}")
print(f"Akurasi Test Set     : {test_acc:.4f}")
print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_pred_test))

# 🔹 Perbandingan Akurasi (Visual)
plt.figure(figsize=(5,4))
sns.barplot(x=["Training", "Test"], y=[train_acc, test_acc], palette="Set2")
plt.title("Perbandingan Akurasi SVM")
plt.ylabel("Akurasi")
plt.ylim(0,1)
plt.show()

# ===========================
# 9️⃣ 5-FOLD CROSS VALIDATION (Evaluasi Kestabilan Model)
# ===========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    SVC(kernel='rbf', random_state=42),
    X_train_bal_fs, y_train_bal,
    cv=cv, scoring='accuracy', n_jobs=-1
)

# Ringkasan hasil
print("\n=== 5-Fold Cross Validation (Train Set) ===")
print(f"Akurasi per Fold   : {np.round(cv_scores, 4)}")
print(f"Rata-rata Akurasi  : {np.mean(cv_scores):.4f}")
print(f"Standar Deviasi    : {np.std(cv_scores):.4f}")

# Interpretasi singkat
if np.std(cv_scores) < 0.02:
    print("📊 Model sangat stabil antar fold (std < 0.02).")
elif np.std(cv_scores) < 0.05:
    print("📈 Model cukup stabil antar fold (std moderat).")
else:
    print("⚠ Model kurang stabil antar fold (perlu tuning parameter).")

# Visualisasi
plt.figure(figsize=(6,4))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores, palette="Set2")
plt.axhline(np.mean(cv_scores), color='red', linestyle='--', label='Rata-rata')
plt.title("Akurasi Tiap Fold (5-Fold Cross Validation)")
plt.ylabel("Akurasi")
plt.ylim(0,1)
plt.legend()
plt.show()

# ===========================
# 8️⃣ PCA SETELAH MODEL (VISUALISASI PREDIKSI)
# ===========================
pca_after_model = PCA(n_components=2)
X_test_pca = pca_after_model.fit_transform(X_test_fs)

plt.figure(figsize=(7,6))
sns.scatterplot(x=X_test_pca[:,0], y=X_test_pca[:,1],
                hue=y_pred_test, palette="Set2", alpha=0.7)
plt.title("Visualisasi PCA Setelah Model SVM (Prediksi Test Set)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# ===========================
# ARTEFAK
# ===========================
artefacts = {
    "svm_model": svm_model,
    "scaler": scaler,
    "selector": selector,
    "ordinal_mapping": ordinal_mapping,
    "numerical_cols": numerical_cols,
    "selected_features": list(selected_features)
}

joblib.dump(artefacts, "model_kelulusan.pkl")