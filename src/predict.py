"""
===================================================================
predict.py — Load Saved Model & Make Predictions
===================================================================
PURPOSE: Provides a simple function to load the trained model and
scaler, accept raw feature values, and return a prediction.

This module is used by the Flask app (app.py) to serve predictions.
===================================================================
"""

import os
import joblib
import numpy as np


def load_model_and_scaler(models_dir):
    """
    Load the saved model and scaler from disk.

    Parameters
    ----------
    models_dir : str
        Path to the models/ directory.

    Returns
    -------
    tuple : (model, scaler, feature_names)
    """
    model_path = os.path.join(models_dir, "placement_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    # Check if files exist
    for path, name in [(model_path, "Model"), (scaler_path, "Scaler")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ {name} file not found at '{path}'. "
                "Please run train_model.py first!"
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else None

    print(f"✅ Model and scaler loaded from '{models_dir}'")
    return model, scaler, feature_names


def make_prediction(model, scaler, features_dict, feature_names):
    """
    Make a placement prediction for a single student.

    Parameters
    ----------
    model : sklearn estimator
        The trained ML model.
    scaler : StandardScaler
        The fitted scaler.
    features_dict : dict
        Raw feature values, e.g. {"cgpa": 8.5, "iq": 110, ...}
    feature_names : list
        Ordered list of feature names the model expects.

    Returns
    -------
    dict with keys:
        - prediction: 1 (Placed) or 0 (Not Placed)
        - probability: float (0.0 to 1.0)
        - label: "Placed" or "Not Placed"
    """
    # Build input array in the CORRECT feature order
    input_values = [features_dict[feat] for feat in feature_names]
    input_array = np.array(input_values).reshape(1, -1)

    # Scale the input using the SAME scaler used during training
    input_scaled = scaler.transform(input_array)

    # Get prediction (0 or 1)
    prediction = model.predict(input_scaled)[0]

    # Get probability (if model supports it)
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_scaled)[0][1]  # P(Placed)
    else:
        probability = float(prediction)

    result = {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "label": "Placed ✅" if prediction == 1 else "Not Placed ❌"
    }

    print(f"🔮 Prediction: {result['label']} "
          f"(Probability: {result['probability']:.2%})")

    return result


# ---------------------------------------------------------------
# MAIN: Test the prediction function
# ---------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    model, scaler, feature_names = load_model_and_scaler(models_dir)

    # Test with a sample student
    sample_student = {
        "cgpa": 8.5,
        "iq": 115,
        "communication_skill": 8,
        "internships": 2,
        "projects": 4,
        "coding_skill": 7,
        "attendance": 85.0,
        "aptitude_score": 78.0,
        "logical_reasoning": 72.0
    }

    print(f"\n📋 Input: {sample_student}")
    result = make_prediction(model, scaler, sample_student, feature_names)
    print(f"📊 Result: {result}")
