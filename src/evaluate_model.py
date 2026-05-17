"""
===================================================================
evaluate_model.py — Generate Evaluation Plots & Metrics
===================================================================
PURPOSE: Create visual evaluations: confusion matrix, ROC curve,
accuracy comparison chart, and feature importance chart.

METRICS EXPLAINED (simply):
  - Confusion Matrix: A 2x2 table showing:
      True Positives  (correctly predicted Placed)
      True Negatives  (correctly predicted Not Placed)
      False Positives (predicted Placed, but actually Not Placed)
      False Negatives (predicted Not Placed, but actually Placed)
  - ROC Curve: Shows the tradeoff between True Positive Rate
    and False Positive Rate. Area Under Curve (AUC) closer to
    1.0 = better model.
===================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)


def plot_confusion_matrices(results, y_test, output_dir):
    """Plot confusion matrix for each model."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight='bold')

    for idx, (name, (model, metrics, y_pred)) in enumerate(results.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"])
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f"{name}\n(Accuracy: {metrics['accuracy']:.2%})")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrices saved to '{path}'")


def plot_roc_curves(results, X_test, y_test, output_dir):
    """Plot ROC curves for all models on one chart."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, (name, (model, metrics, _)) in enumerate(results.items()):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                 label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Model Comparison", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved to '{path}'")


def plot_accuracy_comparison(results, output_dir):
    """Bar chart comparing accuracy of all models."""
    os.makedirs(output_dir, exist_ok=True)

    names = list(results.keys())
    accuracies = [results[n][1]["accuracy"] for n in names]
    f1_scores = [results[n][1]["f1_score"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score',
                   color='#e74c3c', edgecolor='white')

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Accuracy vs F1-Score",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2%}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Accuracy comparison saved to '{path}'")


def plot_feature_importance(model, feature_names, output_dir):
    """Plot feature importance from a tree-based model."""
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model doesn't support feature importance.")
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx],
             color=colors, edgecolor='white')
    plt.yticks(range(len(sorted_idx)),
               [feature_names[i] for i in sorted_idx], fontsize=11)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Feature Importance (Random Forest)", fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance saved to '{path}'")


def generate_all_evaluation_plots(results, X_test, y_test,
                                  feature_names, output_dir):
    """Run all evaluation visualizations."""
    print("\n" + "📊" * 20)
    print("   GENERATING EVALUATION PLOTS")
    print("📊" * 20)

    plot_confusion_matrices(results, y_test, output_dir)
    plot_roc_curves(results, X_test, y_test, output_dir)
    plot_accuracy_comparison(results, output_dir)

    # Feature importance from Random Forest (if available)
    if "Random Forest" in results:
        rf_model = results["Random Forest"][0]
        plot_feature_importance(rf_model, feature_names, output_dir)

    print("\n✅ ALL EVALUATION PLOTS GENERATED!")
