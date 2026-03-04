import joblib

def smart_diagnostic(v_oc, v_load, i, irr, temp):
    try:
        rf_model = joblib.load('rf_primary_model.pkl')
        knn_model = joblib.load('knn_validator_model.pkl')
        
        inputs = [[irr, temp, v_oc]]
        p_rf = rf_model.predict(inputs)[0]
        p_knn = knn_model.predict(inputs)[0]
        
        expected_p = (p_rf + p_knn) / 2
        actual_p = v_load * i
        v_drop_ratio = (v_oc - v_load) / v_oc if v_oc > 0 else 0
        confidence_gap = abs(p_rf - p_knn)

        print(f"\n--- Results ---")
        print(f"Expected: {expected_p:.1f}W | Actual: {actual_p:.1f}W | Drop: {v_drop_ratio*100:.1f}%")

        if confidence_gap > (0.25 * expected_p):
            return "⚠️ Warning: Low AI Confidence"
        if i < 0.1:
            return "STATUS: Open Circuit"
        if v_drop_ratio > 0.35:
            return "FAULT: High Wiring Resistance"
        if actual_p < (0.7 * expected_p):
            return "FAULT: Shading or Dust Detected"

        return "STATUS: Healthy ✅"

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    # Test values: Voc, Vload, I, Irr, Temp
    print(f"Result: {smart_diagnostic(20.5, 15.0, 4.0, 900, 30)}")