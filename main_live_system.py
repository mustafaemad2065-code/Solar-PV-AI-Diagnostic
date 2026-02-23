import requests
import joblib
import time
import sys

# --- CONFIGURATION ---
# Replace these with your actual ThingSpeak Channel credentials
CHANNEL_ID = 'YOUR_CHANNEL_ID'
READ_API_KEY = 'YOUR_READ_API_KEY'
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json"

# AI Model Path
MODEL_PATH = 'pv_brain.pkl'

def load_ai_model():
    """Load the pre-trained Random Forest model."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] AI Model '{MODEL_PATH}' loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file '{MODEL_PATH}' not found. Please train the model first.")
        sys.exit(1)

def fetch_telemetry():
    """Fetch the latest sensor data from ThingSpeak Cloud."""
    try:
        params = {'api_key': READ_API_KEY}
        response = requests.get(THINGSPEAK_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None
            
        # Mapping ThingSpeak Fields to meaningful variables
        return {
            "v": float(data['field1']),        # Voltage
            "i": float(data['field2']),        # Current
            "irr": float(data['field3']),      # Irradiance
            "temp": float(data['field4']),     # Temperature
            "act_ang": float(data['field5']),  # Actual Motor Angle
            "tar_ang": float(data['field6']),  # Target GPS Angle
            "ref_p": float(data['field7'])     # Reference Cell Power
        }
    except Exception as e:
        print(f"[WARNING] Cloud connectivity issue: {e}")
        return None

def perform_diagnosis(model, data):
    """Analyze PV health using AI predictions and logical branching."""
    # 1. Calculate Power Metrics
    measured_p = data['v'] * data['i']
    expected_p = model.predict([[data['irr'], data['temp'], data['act_ang']]])[0]
    
    # 2. Safety & Mechanical Checks
    angle_error = abs(data['tar_ang'] - data['act_ang'])
    
    # 3. Environmental Analysis (Using Reference Cell)
    # Scaling factor for the reference cell to match the full panel capacity
    scaled_ref_p = data['ref_p'] * 10 
    loss_ratio = (expected_p - measured_p) / expected_p if expected_p > 0 else 0

    print(f"\n--- Diagnostic Pulse ---")
    print(f"Measured: {measured_p:.2f}W | Expected: {expected_p:.2f}W | Loss: {loss_ratio*100:.1f}%")

    # --- DECISION LOGIC ---
    if angle_error > 15:
        return "CRITICAL: Tracker Mechanical Failure detected."
    
    if data['v'] < 2.0:
        return "CRITICAL: System Disconnection / Cable Fault."
    
    if loss_ratio > 0.20:
        # Check if the reference cell also dropped (Shading/Clouds) 
        # or stayed high (Soiling on main panel)
        ref_loss_ratio = (scaled_ref_p - measured_p) / scaled_ref_p if scaled_ref_p > 0 else 0
        if ref_loss_ratio > 0.20:
            return "MAINTENANCE: Soiling/Dust accumulation detected on panel surface."
        else:
            return "ENVIRONMENTAL: Performance drop due to Shading or Cloud cover."
            
    return "HEALTHY: System operating within optimal parameters."

def main():
    """Main execution loop."""
    print("=== PV SMART DIAGNOSTIC SYSTEM INITIALIZED ===")
    pv_model = load_ai_model()
    
    while True:
        telemetry = fetch_telemetry()
        
        if telemetry:
            status = perform_diagnosis(pv_model, telemetry)
            print(f"STATUS REPORT: {status}")
        else:
            print("[RETRY] Waiting for fresh data from ThingSpeak...")
            
        time.sleep(20) # Standard ThingSpeak update interval

if __name__ == "__main__":
    main()