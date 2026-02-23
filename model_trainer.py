import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# قراءة البيانات التي صنعناها
df = pd.read_csv('train_data.csv')
X = df[['field3', 'field4', 'field5']] # إشعاع، حرارة، زاوية
y = df['field1'] * df['field2']       # باور

# تدريب RF
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# حفظ "عقل" الموديل في ملف
joblib.dump(model, 'pv_brain.pkl')
print("--- [Step 2] AI Brain (Model) Trained and Saved! ---")