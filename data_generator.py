import pandas as pd
import numpy as np

# توليد بيانات لـ 1000 لحظة زمنية
n = 1000
np.random.seed(42)

irr = np.random.uniform(200, 1000, n)  # الإشعاع
temp = np.random.uniform(20, 55, n)    # الحرارة
angle = np.random.uniform(0, 180, n)   # الزاوية

# حساب الباور المثالي (يوم تدريب سليم)
power = (0.15 * irr) - (0.02 * temp) 
v = np.full(n, 18.0) + np.random.normal(0, 0.1, n)
i = power / v
ref_p = power / 10 # الخلية المرجعية (نظيفة)

# حفظ ملف التدريب
train_df = pd.DataFrame({'field1':v, 'field2':i, 'field3':irr, 'field4':temp, 'field5':angle, 'field6':angle, 'field7':ref_p})
train_df.to_csv('train_data.csv', index=False)
print("--- [Step 1] Training Data Created Successfully! ---")