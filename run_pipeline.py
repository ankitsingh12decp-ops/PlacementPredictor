"""
===================================================================
run_pipeline.py — One-Click Full ML Pipeline Runner
===================================================================
PURPOSE: Run the ENTIRE machine learning pipeline with a single
command. This script orchestrates all steps in order:

  1. Generate synthetic dataset
  2. Run EDA visualizations
  3. Preprocess data
  4. Train & compare ML models
  5. Evaluate & generate plots
  6. Save the best model

USAGE:
  python run_pipeline.py

After running, start the Flask app with:
  python app.py
===================================================================
"""

import sys
# Ensure standard streams support UTF-8 on Windows to prevent UnicodeEncodeError with emojis
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os

# Add project root to path so we can import our modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    """Run the full ML pipeline step by step."""

    print("\n" + "=" * 70)
    print("🚀  STUDENT PLACEMENT PREDICTION — FULL PIPELINE")
    print("=" * 70)

    # Paths
    data_path = os.path.join(project_root, "data", "placement.csv")
    models_dir = os.path.join(project_root, "models")
    eda_dir = os.path.join(project_root, "static", "eda_plots")

    # ===========================================
    # STEP 1: Generate Dataset
    # ===========================================
    print("\n" + "=" * 70)
    print("📊  STEP 1/5: GENERATING DATASET")
    print("=" * 70)
    from src.generate_dataset import generate_placement_dataset
    generate_placement_dataset(n_samples=1000, output_path=data_path)

    # ===========================================
    # STEP 2: Run EDA
    # ===========================================
    print("\n" + "=" * 70)
    print("📈  STEP 2/5: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    from src.eda import generate_eda_plots
    generate_eda_plots(data_path, eda_dir)

    # ===========================================
    # STEP 3-4: Preprocess + Train Models
    # ===========================================
    print("\n" + "=" * 70)
    print("🤖  STEP 3/5: PREPROCESSING + TRAINING")
    print("=" * 70)
    from src.train_model import main as train_main
    results, best_name, best_model, X_test, y_test, feature_names = train_main()

    # ===========================================
    # STEP 5: Evaluate Models
    # ===========================================
    print("\n" + "=" * 70)
    print("📉  STEP 4/5: MODEL EVALUATION")
    print("=" * 70)
    from src.evaluate_model import generate_all_evaluation_plots
    generate_all_evaluation_plots(results, X_test, y_test, feature_names, eda_dir)

    # ===========================================
    # STEP 6: Test Prediction
    # ===========================================
    print("\n" + "=" * 70)
    print("🔮  STEP 5/5: TEST PREDICTION")
    print("=" * 70)
    from src.predict import load_model_and_scaler, make_prediction
    model, scaler, feat_names = load_model_and_scaler(models_dir)

    test_student = {
        "cgpa": 8.5, "iq": 115, "communication_skill": 8,
        "internships": 2, "projects": 4, "coding_skill": 7,
        "attendance": 85.0, "aptitude_score": 78.0, "logical_reasoning": 72.0
    }
    print(f"\n📋 Test student: {test_student}")
    result = make_prediction(model, scaler, test_student, feat_names)

    # ===========================================
    # DONE!
    # ===========================================
    print("\n" + "=" * 70)
    print("✅  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n🏆 Best Model: {best_name}")
    print(f"📁 Model saved: models/placement_model.pkl")
    print(f"📊 EDA plots:   static/eda_plots/")
    print(f"\n🌐 To start the web app, run:")
    print(f"   python app.py")
    print(f"   Then open http://127.0.0.1:5000 in your browser")
    print("=" * 70)


if __name__ == "__main__":
    main()
