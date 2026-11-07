# ==========================================
# JobLens ML: Skills & Experience Salary Prediction
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('salary_prediction_skills_data.csv')

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

# 2. Feature Selection & Train-Test Split
X = df[['Experience_Years', 'Technical_Skills_Count', 'Projects_Completed']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n--- Linear Regression Performance ---")
print(f"RMSE: ${lr_rmse:.2f}")
print(f"R2 Score: {lr_r2:.4f}")

# 4. Model Training: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Performance ---")
print(f"RMSE: ${rf_rmse:.2f}")
print(f"R2 Score: {rf_r2:.4f}")

# 5. Test with a Custom Prediction (e.g., 3 years exp, 5 skills, 2 projects)
sample_input = np.array([[3.0, 5, 2]])
predicted_salary = lr_model.predict(sample_input)
print(f"\nPredicted Salary for 3 Years Exp, 5 Skills, 2 Projects: ${predicted_salary[0]:,.2f}")