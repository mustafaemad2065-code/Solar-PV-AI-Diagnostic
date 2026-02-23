import joblib

# تحميل الموديل الذي تدرب
model = joblib.load('pv_brain.pkl')

def diagnose(v, i, irr, temp, act_ang, tar_ang, ref_p):
    measured_p = v * i
    expected_p = model.predict([[irr, temp, act_ang]])[0]
    
    loss_ratio = (expected_p - measured_p) / expected_p if expected_p > 0 else 0
    angle_error = abs(tar_ang - act_ang)
    ref_ideal_p = ref_p * 10 # تكبير قراءة الخلية المرجعية

    print(f"\n>> Results: Measured={measured_p:.1f}W, Expected={expected_p:.1f}W, Angle Error={angle_error:.1f}")

    # المنطق الذكي
    if angle_error > 15: return "Decision: ❌ Tracker Mechanical Fault"
    if v < 2.0: return "Decision: ❌ Disconnection"
    if (expected_p - measured_p) / expected_p > 0.20:
        if (ref_ideal_p - measured_p) / ref_ideal_p > 0.20:
            return "Decision: 🍂 Soiling (Dust)"
        else:
            return "Decision: ☁️ Shading/Clouds"
    return "Decision: ✅ System Healthy"

test_scenario("Broken Wire (Disconnection)", 
              v=0.5, i=0.0, irr=800, temp=30, act_ang=45, tar_ang=45, ref_p=8.0)

