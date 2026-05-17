"""
===================================================================
eda.py — Exploratory Data Analysis (EDA) Visualizations
===================================================================
PURPOSE: Generate professional EDA plots to understand the dataset
before training. Saves all plots to static/eda_plots/.

EDA EXPLAINED:
  EDA is the process of examining your data BEFORE building models.
  It helps you understand patterns, spot anomalies, and decide
  which features are most useful. Think of it as "getting to know
  your data" before making predictions.
===================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_eda_plots(data_path, output_dir):
    """Generate all EDA visualizations and save them."""

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path).dropna().drop_duplicates()

    # Set a clean visual style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    print("📊 Generating EDA plots...")

    # ---- 1. Placement Distribution ----
    # Shows how many students are placed vs not placed.
    # Helps check if data is BALANCED (roughly equal classes).
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['placement'].value_counts()
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(['Not Placed (0)', 'Placed (1)'],
                  [counts.get(0, 0), counts.get(1, 0)],
                  color=colors, edgecolor='white', width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', fontsize=13, fontweight='bold')
    ax.set_title("Placement Distribution", fontsize=15, fontweight='bold')
    ax.set_ylabel("Number of Students", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "placement_distribution.png"), dpi=150)
    plt.close()
    print("   ✅ Placement distribution")

    # ---- 2. CGPA vs Placement ----
    # Box plot shows how CGPA differs between placed and not placed.
    # If placed students have higher CGPA, it's a useful feature.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='placement', y='cgpa', data=df, palette=colors, ax=ax, width=0.4)
    ax.set_xticklabels(['Not Placed', 'Placed'], fontsize=12)
    ax.set_title("CGPA vs Placement Status", fontsize=15, fontweight='bold')
    ax.set_ylabel("CGPA", fontsize=12)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cgpa_vs_placement.png"), dpi=150)
    plt.close()
    print("   ✅ CGPA vs Placement")

    # ---- 3. Internships vs Placement ----
    # Grouped bar chart: for each internship count, how many placed vs not.
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df['internships'], df['placement'])
    ct.columns = ['Not Placed', 'Placed']
    ct.plot(kind='bar', color=colors, edgecolor='white', ax=ax, width=0.7)
    ax.set_title("Internships vs Placement", fontsize=15, fontweight='bold')
    ax.set_xlabel("Number of Internships", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "internships_vs_placement.png"), dpi=150)
    plt.close()
    print("   ✅ Internships vs Placement")

    # ---- 4. Correlation Heatmap ----
    # Shows how strongly each feature correlates with every other.
    # Values close to +1 or -1 = strong correlation.
    # Helps identify which features predict placement.
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("   ✅ Correlation heatmap")

    # ---- 5. Skill Scores Distribution ----
    # Distribution of coding, communication, aptitude scores by placement.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    skill_cols = ['coding_skill', 'communication_skill', 'aptitude_score']
    titles = ['Coding Skill', 'Communication Skill', 'Aptitude Score']

    for i, (col, title) in enumerate(zip(skill_cols, titles)):
        sns.histplot(data=df, x=col, hue='placement', kde=True,
                     palette={0: colors[0], 1: colors[1]},
                     ax=axes[i], alpha=0.6)
        axes[i].set_title(f"{title} by Placement", fontsize=13, fontweight='bold')
        axes[i].legend(labels=['Not Placed', 'Placed'], fontsize=10)

    plt.suptitle("Skill Score Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skill_score_analysis.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Skill score analysis")

    print("\n✅ ALL EDA PLOTS SAVED!")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_eda_plots(
        os.path.join(project_root, "data", "placement.csv"),
        os.path.join(project_root, "static", "eda_plots")
    )
