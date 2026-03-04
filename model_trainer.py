import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib

def train_models():
    try:
        data = pd.read_csv('solar_data_v2.csv')
        X = data[['irradiance', 'temperature', 'v_oc']]
        y = data['power']
        
        # Train Primary Model (Random Forest)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        joblib.dump(rf, 'rf_primary_model.pkl')
        print("✅ RF Model Trained.")

        # Train Validator Model (KNN)
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)
        joblib.dump(knn, 'knn_validator_model.pkl')
        print("✅ KNN Model Trained.")
        
    except FileNotFoundError:
        print("❌ Error: Run data_generator.py first!")

if __name__ == "__main__":
    train_models()