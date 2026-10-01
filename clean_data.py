"""
clean_data.py
Phase 1, 2 & 3: Data Loading, Cleaning & Exploratory Data Analysis (EDA)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("  PHASE 1 & 2: DATA LOADING AND CLEANING")
    print("=" * 60)

    # 1. Load Raw Dataset
    raw_file = 'car_price.csv'
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Error: '{raw_file}' not found in current directory. Download it first!")

    df = pd.read_csv(raw_file)
    print(f"[✓] Raw dataset loaded successfully. Shape: {df.shape}")

    # Display initial columns
    print("\n[i] Raw Columns:", list(df.columns))

    # 2. Basic Cleaning & Normalization
    # Drop unnecessary identifier columns if present
    if 'Car_Name' in df.columns:
        df = df.drop(columns=['Car_Name'])
        print("[✓] Dropped 'Car_Name' column (unnecessary identifier).")

    # Handle Missing Values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[!] Found {missing_count} missing values. Dropping rows with nulls...")
        df = df.dropna()
    else:
        print("[✓] No missing values detected.")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"[✓] Removed {duplicates} duplicate records.")

    # 3. Feature Engineering
    # Calculate Age of the car assuming current year 2026
    current_year = 2026
    if 'Year' in df.columns:
        df['Age'] = current_year - df['Year']
        df = df.drop(columns=['Year'])
        print("[✓] Engineered 'Age' feature and dropped raw 'Year' column.")

    # 4. One-Hot Encoding Categorical Variables
    # Explicitly pass 'object' and 'string' to silence pandas deprecation warnings
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"[i] Categorical Columns to encode: {categorical_cols}")

    df_cleaned = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"[✓] Categorical encoding complete. Cleaned shape: {df_cleaned.shape}")

    # 5. Exploratory Data Analysis (EDA Visualizations)
    print("\n=" * 60)
    print("  PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    plt.figure(figsize=(12, 5))

    # Selling Price Distribution
    plt.subplot(1, 2, 1)
    if 'Selling_Price' in df_cleaned.columns:
        sns.histplot(df_cleaned['Selling_Price'], kde=True, color='skyblue')
        plt.title('Selling Price Distribution')
        plt.xlabel('Selling Price')

    # Correlation Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(df_cleaned.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('eda_summary.png')
    print("[✓] EDA plot saved as 'eda_summary.png'.")

    # 6. Save Cleaned Dataset
    output_file = 'cleaned_car_price.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n[✓] Data successfully cleaned and saved to '{output_file}'.\n")

if __name__ == '__main__':
    main()