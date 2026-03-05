import joblib
import numpy as np

def run_hybrid_diagnostic(irr, temp, dust, v_oc, v_batt, v_charge, curr):
    # 1. تحميل الموديلات الهجينة
    try:
        rf = joblib.load('rf_model.pkl')
        knn = joblib.load('knn_model.pkl')
    except:
        return "❌ Error: Models not found! Run the trainer first."

    # 2. تجهيز البيانات للـ AI
    input_features = np.array([[irr, temp, dust, v_oc, v_batt, v_charge, curr]])
    
    # 3. التوقع من الموديلين (Hybrid Decision)
    res_rf = rf.predict(input_features)[0]
    res_knn = knn.predict(input_features)[0]
    
    # 4. حسابات منطق الهاردوير (Logic-Based Actions)
    soc = ((v_batt - 10.5) / (12.7 - 10.5)) * 100
    soc = max(0, min(100, soc))
    
    # الأكشنز
    load_status = "🔴 OFF (Protection)" if (irr < 50 and soc <= 30) else "🟢 ON"
    pump_status = "🌊 RUNNING" if dust > 75 else "⚪ STANDBY"

    # 5. عرض التقرير النهائي
    print("\n" + "="*40)
    print("      🌞 SOLAR HYBRID DIAGNOSTIC 🧠")
    print("="*40)
    print(f"📡 Sensor Inputs: {irr}W/m², {temp}°C, {dust}% Dust")
    print(f"⚡ Voltages: Voc={v_oc}V | Vbatt={v_batt}V | Vcharge={v_charge}V")
    print(f"🔌 Current: {curr}A")
    print("-" * 40)
    print(f"🤖 RF Prediction:  {res_rf}")
    print(f"🤖 KNN Prediction: {res_knn}")
    print(f"📊 Battery SoC:    {soc:.1f}%")
    print("-" * 40)
    print(f"⚙️  ACTION -> Load Relay:    {load_status}")
    print(f"⚙️  ACTION -> Cleaning Pump: {pump_status}")
    print("="*40 + "\n")

# --- 🧪 منطقة التجارب (جرب الحالات اللي اتفقنا عليها) ---

# حالة 1: الجو مغيم (Cloudy) - تيار قليل طبيعي
print("Testing: Cloudy Weather...")
run_hybrid_diagnostic(irr=180, temp=25, dust=5, v_oc=20.5, v_batt=12.2, v_charge=12.4, curr=0.6)

# حالة 2: غبار عالي (High Dust) - محتاج تنظيف
print("Testing: High Dust...")
run_hybrid_diagnostic(irr=900, temp=40, dust=85, v_oc=19.5, v_batt=12.0, v_charge=13.2, curr=2.8)

# حالة 3: بطارية مملحة (Battery Fault) - فولت شحن عالي ومفيش تيار
print("Testing: Sulfated Battery...")
run_hybrid_diagnostic(irr=850, temp=35, dust=10, v_oc=20.0, v_batt=11.5, v_charge=14.5, curr=0.15)

# حالة 4: حماية البطارية بالليل (Low Battery/Night)
print("Testing: Low Battery Protection...")
run_hybrid_diagnostic(irr=10, temp=22, dust=5, v_oc=2.0, v_batt=11.0, v_charge=11.0, curr=0.0)