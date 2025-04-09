import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('data.csv')

# Convert Date column and extract features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['weekday'] = df['Date'].dt.weekday
    df.drop(columns=['Date'], inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Remove outliers in Weekly_Sales
q_high = df['Weekly_Sales'].quantile(0.99)
df = df[df['Weekly_Sales'] < q_high]

# Define features and target
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'day', 'month', 'year', 'weekday']
target = 'Weekly_Sales'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Model trained successfully using XGBoost!")
print(f"📈 R² Score (Accuracy): {r2:.4f}")
print(f"📉 RMSE: {rmse:.2f}")

# Save model and scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model and scaler saved.")
