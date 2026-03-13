import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 1️⃣ قراءة ملف الداتا
# -----------------------------

data = pd.read_csv("fault_prediction_training_data.csv")

print("Dataset Loaded Successfully")
print(data.head())

# -----------------------------
# 2️⃣ تجهيز بيانات التدريب
# -----------------------------

features = [
    "irr","temp","dust",
    "v_oc","v_batt","v_charge","curr",
    "d_temp","d_curr","d_vbatt"
]

X = data[features]
y = data["future_fault"]

# -----------------------------
# 3️⃣ تقسيم البيانات
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ تدريب Random Forest
# -----------------------------

print("Training Random Forest Predictor...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------
# 5️⃣ تدريب KNN
# -----------------------------

print("Training KNN Predictor...")

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

# -----------------------------
# 6️⃣ تقييم الموديلات
# -----------------------------

rf_acc = rf.score(X_test, y_test)
knn_acc = knn.score(X_test, y_test)

print("\nModel Accuracy:")
print("Random Forest:", rf_acc)
print("KNN:", knn_acc)

# -----------------------------
# 7️⃣ حفظ الموديلات
# -----------------------------

joblib.dump(rf, "rf_predictor.pkl")
joblib.dump(knn, "knn_predictor.pkl")

print("\nPrediction models saved successfully!")