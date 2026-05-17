"""
===================================================================
train_model.py — Train, Compare & Save ML Models
===================================================================
PURPOSE: Train 3 models (Logistic Regression, Decision Tree, Random
Forest), compare their performance, and save the best one.

ML CONCEPTS EXPLAINED:
  - Logistic Regression: Simple linear model for classification.
    Good baseline. Fast, interpretable, but can't capture complex
    non-linear patterns.
  - Decision Tree: Makes decisions by asking yes/no questions about
    features. Easy to understand but tends to OVERFIT (memorize
    training data instead of learning patterns).
  - Random Forest: An ENSEMBLE of many decision trees. Each tree
    sees a random subset of data. Their votes are combined → more
    accurate, less overfitting. Usually the best performer!

KEY TERMS:
  - Overfitting: Model learns training data TOO well (including noise).
    Performs great on training data, poorly on new data.
  - Underfitting: Model is too simple to capture patterns.
    Performs poorly on BOTH training and test data.
  - Bias vs Variance:
    High Bias = underfitting (model too simple)
    High Variance = overfitting (model too complex)
    Goal: find the sweet spot (bias-variance tradeoff).
===================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# Import our preprocessing module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train 3 ML models and compare their performance.

    Returns dict of {model_name: (model, metrics_dict)}.
    """
    print("\n" + "🤖" * 20)
    print("   TRAINING ML MODELS")
    print("🤖" * 20)

    # Define our 3 models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,       # Max training iterations
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,        # Limit depth to prevent overfitting
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,    # 100 trees in the forest
            max_depth=15,        # Limit each tree's depth
            random_state=42,
            n_jobs=-1            # Use all CPU cores for speed
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔄 Training: {name}")
        print(f"{'='*60}")

        # Train the model (fit = learn from training data)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }

        results[name] = (model, metrics, y_pred)

        # Print results
        print(f"   Accuracy:  {acc:.4f}  (% of correct predictions)")
        print(f"   Precision: {prec:.4f} (of predicted placed, how many actually placed?)")
        print(f"   Recall:    {rec:.4f}  (of actually placed, how many did we find?)")
        print(f"   F1 Score:  {f1:.4f}  (harmonic mean of precision & recall)")
        print(f"\n📋 Classification Report:\n{classification_report(y_test, y_pred)}")

    return results


def compare_and_save_best(results, models_dir, feature_names):
    """
    Compare all models, pick the best one by F1-score, and save it.

    WHY F1-SCORE? Because accuracy alone can be misleading with
    imbalanced data. F1 balances precision AND recall.
    """
    print("\n" + "🏆" * 20)
    print("   MODEL COMPARISON")
    print("🏆" * 20)

    # Build comparison table
    comparison = []
    for name, (model, metrics, _) in results.items():
        comparison.append({"Model": name, **metrics})

    df_compare = pd.DataFrame(comparison)
    print(f"\n{df_compare.to_string(index=False)}")

    # Find the best model (highest F1-score)
    best_name = max(results, key=lambda k: results[k][1]["f1_score"])
    best_model = results[best_name][0]
    best_f1 = results[best_name][1]["f1_score"]

    print(f"\n🏆 BEST MODEL: {best_name} (F1 = {best_f1})")

    # Save the best model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "placement_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to '{model_path}'")

    # Save feature names for prediction
    features_path = os.path.join(models_dir, "feature_names.pkl")
    joblib.dump(feature_names, features_path)
    print(f"💾 Feature names saved to '{features_path}'")

    return best_name, best_model, df_compare


def main():
    """Run the full training pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "placement.csv")
    models_dir = os.path.join(project_root, "models")

    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
        data_path, models_dir
    )

    # Step 2: Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Step 3: Compare and save best
    best_name, best_model, df_compare = compare_and_save_best(
        results, models_dir, feature_names
    )

    return results, best_name, best_model, X_test, y_test, feature_names


if __name__ == "__main__":
    main()
