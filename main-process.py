import requests
import joblib
import time

# --- الإعدادات ---
CHANNEL_ID = 'YOUR_ID'
READ_KEY = 'YOUR_KEY'
# رابط سحب آخر قراءة فقط
TS_URL = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json'

def diagnostic_engine():
    # تحميل الموديلات اللي دربناها
    try:
        rf = joblib.load('rf_primary_model.pkl')
        knn = joblib.load('knn_validator_model.pkl')
        print("🚀 Diagnostic Engine Started... Monitoring Live Data.")
    except:
        print("❌ Error: Models not found!")
        return

    while True:
        try:
            # 1. جلب البيانات من الكلاود
            r = requests.get(TS_URL, params={'api_key': READ_KEY})
            feed = r.json()
            
            # 2. استخراج القيم (تأكد من ترتيب الـ Fields مع التيم)
            v_oc = float(feed['field1'])   # جهد الدائرة المفتوحة
            v_load = float(feed['field2']) # جهد الحمل
            curr = float(feed['field3'])   # التيار
            irr = float(feed['field4'])    # الإشعاع
            temp = float(feed['field5'])   # الحرارة
            
            # 3. حسابات الـ AI
            inputs = [[irr, temp, v_oc]]
            p_rf = rf.predict(inputs)[0]
            p_knn = knn.predict(inputs)[0]
            
            expected_p = (p_rf + p_knn) / 2 # متوسط التوقع
            actual_p = v_load * curr        # القدرة الفعلية
            v_drop = (v_oc - v_load) / v_oc # نسبة الهبوط في الجهد

            # 4. التحليل والتشخيص (The Logic)
            print(f"\n[Monitor] Actual: {actual_p:.1f}W | Expected: {expected_p:.1f}W")
            
            if curr < 0.05:
                diag = "STATUS: System Open Circuit (Disconnected)"
            elif v_drop > 0.35:
                diag = "FAULT: High Contact Resistance / Wiring Issue"
            elif actual_p < (0.7 * expected_p):
                if v_oc < 18.5:
                    diag = "FAULT: Partial Shading or Cell Damage"
                else:
                    diag = "ADVICE: Maintenance Needed (Soiling/Dust)"
            else:
                diag = "STATUS: Healthy ✅"

            print(f"🔎 Diagnosis: {diag}")
            
        except Exception as e:
            print(f"Waiting for data... ({e})")
            
        time.sleep(15) # تحديث كل 15 ثانية

if __name__ == "__main__":
    diagnostic_engine()