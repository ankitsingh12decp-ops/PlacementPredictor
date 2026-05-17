"""
===================================================================
data_preprocessing.py — Clean, Transform & Prepare Data for ML
===================================================================
PURPOSE: Clean raw data (handle missing values, remove duplicates,
detect outliers, scale features) and split into train/test sets.
===================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(filepath):
    """Load CSV data into a Pandas DataFrame."""
    print("=" * 60)
    print("📂 STEP 1: LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    return df


def show_data_info(df, label=""):
    """Display basic information about the dataset."""
    print(f"\n{'='*60}")
    print(f"📊 DATA INFO {label}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\n🔍 Missing values:\n{df.isnull().sum()}")
    print(f"\nTotal missing: {df.isnull().sum().sum()}")
    print(f"\n📈 Statistics:\n{df.describe().round(2).to_string()}")


def handle_missing_values(df):
    """
    Fill missing values with MEDIAN of each column.
    WHY MEDIAN? It's robust to outliers (unlike mean).
    """
    print(f"\n{'='*60}")
    print("🧹 STEP 2: HANDLING MISSING VALUES")
    print(f"{'='*60}")
    missing_before = df.isnull().sum().sum()
    print(f"Missing values BEFORE: {missing_before}")

    if missing_before == 0:
        print("✅ No missing values found!")
        return df

    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled {count} NaN in '{col}' with median={median_val}")

    print(f"✅ Missing values AFTER: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """Remove duplicate rows to prevent model bias."""
    print(f"\n{'='*60}")
    print("🗑️ STEP 3: REMOVING DUPLICATES")
    print(f"{'='*60}")
    rows_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {rows_before - len(df)} duplicates ({rows_before} → {len(df)} rows)")
    return df


def detect_outliers(df):
    """
    Detect outliers using IQR method.
    IQR = Q3 - Q1. Outliers are below Q1-1.5*IQR or above Q3+1.5*IQR.
    We detect but DON'T remove — students can have genuine extremes.
    """
    print(f"\n{'='*60}")
    print("🔍 STEP 4: DETECTING OUTLIERS (IQR Method)")
    print(f"{'='*60}")
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != 'placement']
    total = 0
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        total += outliers
        if outliers > 0:
            print(f"   '{col}': {outliers} outliers")
    print(f"📌 Total outliers: {total} (keeping them)")
    return df


def scale_features(X_train, X_test, models_dir):
    """
    Scale features using StandardScaler (mean=0, std=1).
    WHY? So features on different scales (CGPA 4-10, IQ 80-140)
    are treated equally by the model.
    FIT ONLY ON TRAIN DATA to prevent data leakage.
    """
    print(f"\n{'='*60}")
    print("⚖️ STEP 5: FEATURE SCALING")
    print(f"{'='*60}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Learn + transform
    X_test_scaled = scaler.transform(X_test)          # Only transform
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Features scaled. Scaler saved to '{scaler_path}'")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath, models_dir):
    """
    Run the FULL preprocessing pipeline:
    Load → Clean → Split → Scale → Return ready-to-train data.
    """
    print("\n" + "🚀" * 20)
    print("   STARTING PREPROCESSING PIPELINE")
    print("🚀" * 20)

    df = load_data(filepath)
    show_data_info(df, "(BEFORE CLEANING)")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = detect_outliers(df)
    show_data_info(df, "(AFTER CLEANING)")

    # Split features (X) and target (y)
    feature_names = [col for col in df.columns if col != 'placement']
    X = df[feature_names]
    y = df['placement']
    print(f"\n✂️ Features: {feature_names}")
    print(f"   Placed: {(y==1).sum()} | Not Placed: {(y==0).sum()}")

    # Train/Test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, models_dir)

    print("\n✅ PREPROCESSING COMPLETE!")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_data(
        os.path.join(project_root, "data", "placement.csv"),
        os.path.join(project_root, "models")
    )
