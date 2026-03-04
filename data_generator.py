import pandas as pd
import numpy as np

def generate_solar_data(samples=1000):
    np.random.seed(42)
    irr = np.random.uniform(200, 1000, samples)  
    temp = np.random.uniform(20, 60, samples)    
    
    # Voc (Open Circuit Voltage)
    v_oc = 21.0 - (0.05 * (temp - 25)) + np.random.normal(0, 0.1, samples)
    # Vload (Voltage under load)
    v_load = v_oc * 0.8 + np.random.normal(0, 0.2, samples)
    
    i = (irr / 1000) * 5.0 + np.random.normal(0, 0.1, samples)
    p_actual = v_load * i
    
    df = pd.DataFrame({
        'irradiance': irr,
        'temperature': temp,
        'v_oc': v_oc,
        'v_load': v_load,
        'current': i,
        'power': p_actual
    })
    
    df.to_csv('solar_data_v2.csv', index=False)
    print("✅ Created: solar_data_v2.csv")

if __name__ == "__main__":
    generate_solar_data()