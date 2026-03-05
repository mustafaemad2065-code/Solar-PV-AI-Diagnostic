import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

def build_and_train_hybrid_system():
    print("⏳ Generating Synthetic Data based on Hybrid Logic...")
    np.random.seed(42)
    samples = 2500 # زيادة العينات لدقة أعلى
    
    # --- 1. توليد المدخلات (Inputs) ---
    irr = np.random.uniform(0, 1000, samples)     # الإشعاع
    temp = np.random.uniform(20, 60, samples)    # الحرارة
    dust = np.random.uniform(0, 100, samples)    # نسبة الغبار
    v_batt = np.random.uniform(10.5, 12.7, samples) # جهد البطارية الحقيقي
    
    data = []
    for i in range(samples):
        # الحسابات الفيزيائية الأساسية
        v_oc = 21.0 - (0.05 * (temp[i] - 25)) - (0.01 * dust[i])
        base_i = (irr[i] / 1000) * 5.0
        curr = base_i * (1 - (dust[i]/150))
        v_charge = v_batt[i] + (curr * 0.4) # جهد الشحن المفترض
        
        # --- 2. تحديد الـ 7 حالات (Labeling Logic) ---
        if curr > 7.5: 
            status = "Short Circuit"
        elif 20 < irr[i] < 220: 
            status = "Cloudy Mode"
        elif dust[i] > 75 and irr[i] > 400: 
            status = "High Dust"
        elif curr < 0.15 and irr[i] > 450: 
            status = "Panel Disconnect"
        elif v_charge > v_batt[i] + 2.2 and curr < 0.3: 
            status = "Battery Fault"
        elif v_batt[i] < 11.2 and irr[i] < 50: 
            status = "Low Battery"
        else: 
            status = "Healthy"
        
        data.append([irr[i], temp[i], dust[i], v_oc, v_batt[i], v_charge, curr, status])

    # تحويل البيانات لجدول
    columns = ['irr', 'temp', 'dust', 'v_oc', 'v_batt', 'v_charge', 'curr', 'status']
    df = pd.DataFrame(data, columns=columns)
    
    # تجهيز البيانات للتدريب
    X = df.drop('status', axis=1)
    y = df['status']

    # --- 3. تدريب الموديلات (Hybrid Training) ---
    print("🧠 Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    print("🧠 Training KNN Model...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X, y)
    
    # --- 4. حفظ الموديلات والبيانات ---
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(knn_model, 'knn_model.pkl')
    df.to_csv('training_data_log.csv', index=False)
    
    print("-" * 30)
    print("✅ SUCCESS: Hybrid System is Ready!")
    print("📁 Files Created: 'rf_model.pkl', 'knn_model.pkl', 'training_data_log.csv'")
    print("-" * 30)

if __name__ == "__main__":
    build_and_train_hybrid_system()