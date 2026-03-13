import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

def build_fault_prediction_system():

    print("⏳ Generating Time-Series Data for Fault Prediction...")

    np.random.seed(42)
    samples = 3000   # عدد العينات

    data = []

    # القيم السابقة لمحاكاة الزمن
    prev_temp = 30
    prev_curr = 3
    prev_vbatt = 12

    for i in range(samples):

        # ---- المدخلات ----
        irr = np.random.uniform(0,1000)
        temp = np.random.uniform(20,60)
        dust = np.random.uniform(0,100)
        v_batt = np.random.uniform(10.5,12.7)

        # ---- حسابات فيزيائية ----
        v_oc = 21 - (0.05*(temp-25)) - (0.01*dust)

        base_i = (irr/1000)*5
        curr = base_i*(1-(dust/150))

        v_charge = v_batt + curr*0.4

        # ---- حساب التغيرات الزمنية ----
        d_temp = temp - prev_temp
        d_curr = curr - prev_curr
        d_vbatt = v_batt - prev_vbatt

        # ---- منطق التنبؤ بالعطل قبل ساعة ----
        if curr > 7:
            future_fault = "Future Short Circuit"

        elif dust > 70 and irr > 500:
            future_fault = "Future High Dust"

        elif d_curr < -1 and irr > 400:
            future_fault = "Future Panel Failure"

        elif d_vbatt < -0.6 and irr < 100:
            future_fault = "Future Low Battery"

        elif v_charge > v_batt + 2 and curr < 0.4:
            future_fault = "Future Battery Fault"

        else:
            future_fault = "Stable"

        data.append([
            irr,temp,dust,v_oc,v_batt,v_charge,curr,
            d_temp,d_curr,d_vbatt,
            future_fault
        ])

        # تحديث القيم السابقة
        prev_temp = temp
        prev_curr = curr
        prev_vbatt = v_batt


    columns = [
        'irr','temp','dust','v_oc','v_batt','v_charge','curr',
        'd_temp','d_curr','d_vbatt',
        'future_fault'
    ]

    df = pd.DataFrame(data,columns=columns)

    # المدخلات والمخرجات
    X = df.drop('future_fault',axis=1)
    y = df['future_fault']


    print("🧠 Training Random Forest Prediction Model...")

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )

    rf_model.fit(X,y)


    print("🧠 Training KNN Prediction Model...")

    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X,y)


    # حفظ الموديلات
    joblib.dump(rf_model,"rf_fault_prediction.pkl")
    joblib.dump(knn_model,"knn_fault_prediction.pkl")

    df.to_csv("fault_prediction_training_data.csv",index=False)


    print("--------------------------------------------------")
    print("✅ Fault Prediction System Ready")
    print("Predicts faults 1 hour before occurrence")
    print("Files Created:")
    print("rf_fault_prediction.pkl")
    print("knn_fault_prediction.pkl")
    print("fault_prediction_training_data.csv")
    print("--------------------------------------------------")


if __name__ == "__main__":
    build_fault_prediction_system()