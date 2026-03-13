import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 1️⃣ قراءة ملف البيانات
# -----------------------------

data = pd.read_csv("fault_prediction_training_data.csv")

print("Dataset Loaded Successfully")
print(data.head())

# -----------------------------
# 2️⃣ اختيار الخصائص
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

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------------
# 4️⃣ تدريب KNN
# -----------------------------

print("Training KNN Predictor...")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

# -----------------------------
# 5️⃣ تقييم الموديل
# -----------------------------

accuracy = knn.score(X_test,y_test)

print("KNN Accuracy:",accuracy)

# -----------------------------
# 6️⃣ حفظ الموديل
# -----------------------------

joblib.dump(knn,"knn_predictor.pkl")

print("KNN model saved!")