import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# 1. Load raw data (assuming car_data.csv is your original dataset)
raw_data_path = "car_data.csv"
if not os.path.exists(raw_data_path):
    # Fallback check if named car_price.csv
    raw_data_path = "car_price.csv"

df = pd.read_csv(raw_data_path)

# 2. Data Cleaning & Feature Engineering
# Calculate Car Age based on Year-
from datetime import datetime
current_year = datetime.now().year
df['Age'] = current_year - df['Year']

# Drop unnecessary columns
if 'Car_Name' in df.columns:
    df.drop(columns=['Car_Name'], inplace=True)
if 'Year' in df.columns:
    df.drop(columns=['Year'], inplace=True)

# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Ensure all boolean columns are converted to float/int
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(float)

# 3. Train Model
X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predictions & Value Score Calculation
df['Predicted_Price'] = model.predict(X)
df['Value_Score'] = df['Predicted_Price'] - df['Selling_Price']

# 5. Save cleaned dataset and model files explicitly
df.to_csv("cleaned_car_price.csv", index=False)

with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SUCCESS: `cleaned_car_price.csv` and `linear_regression_model.pkl` have been generated successfully!")