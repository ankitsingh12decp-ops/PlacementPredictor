"""
===================================================================
generate_dataset.py — Create a Realistic Synthetic Placement Dataset
===================================================================

PURPOSE:
    Since we don't have a real-world dataset, we CREATE one that mimics
    real student data.  This is a very common practice in ML projects
    when real data isn't available.

HOW IT WORKS:
    1. We define realistic ranges for each feature (e.g., CGPA 4–10).
    2. We generate random values using NumPy's random functions.
    3. We compute a "placement score" — a weighted sum of features.
       Students with higher scores are MORE LIKELY to be placed.
    4. We add random noise so the data isn't perfectly separable
       (just like real life!).
    5. We save the data as a CSV file.

FEATURES GENERATED:
    - cgpa              : College GPA (4.0 to 10.0)
    - iq                : IQ score (80 to 140)
    - communication_skill: Self-rated 1–10
    - internships       : Number of internships (0–4)
    - projects          : Number of projects done (0–6)
    - coding_skill      : Self-rated 1–10
    - attendance        : Attendance percentage (40–100)
    - aptitude_score    : Aptitude test score (30–100)
    - logical_reasoning : Logical reasoning score (30–100)
    - placement         : TARGET — 1 = Placed, 0 = Not Placed
===================================================================
"""

# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import numpy as np      # NumPy: for generating random numbers & math
import pandas as pd     # Pandas: for creating and saving DataFrames
import os               # os: for file path operations

def generate_placement_dataset(n_samples=1000, output_path="data/placement.csv"):
    """
    Generate a synthetic student placement dataset.

    Parameters
    ----------
    n_samples : int
        Number of student records to create (default: 1000).
    output_path : str
        Where to save the CSV file.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """

    # Set a random seed so results are REPRODUCIBLE
    # (same seed = same random numbers every time you run)
    np.random.seed(42)

    print(f"🔧 Generating synthetic dataset with {n_samples} student records...")

    # -----------------------------------------------------------
    # STEP 1: Generate each feature with realistic distributions
    # -----------------------------------------------------------

    # CGPA: Continuous value between 4.0 and 10.0
    # We use a normal (bell-curve) distribution centered at 7.0
    # np.clip ensures values stay within [4.0, 10.0]
    cgpa = np.clip(np.random.normal(loc=7.0, scale=1.2, size=n_samples), 4.0, 10.0)
    cgpa = np.round(cgpa, 2)  # Round to 2 decimal places

    # IQ: Integer between 80 and 140, centered around 105
    iq = np.clip(np.random.normal(loc=105, scale=12, size=n_samples), 80, 140).astype(int)

    # Communication Skill: Integer 1–10
    communication_skill = np.random.randint(1, 11, size=n_samples)

    # Internships: 0–4 (most students have 0–2)
    # We use a binomial distribution — it naturally gives more 0s and 1s
    internships = np.random.binomial(n=4, p=0.3, size=n_samples)

    # Projects: 0–6
    projects = np.random.binomial(n=6, p=0.4, size=n_samples)

    # Coding Skill: 1–10
    coding_skill = np.random.randint(1, 11, size=n_samples)

    # Attendance: 40–100% (most students attend 60–90%)
    attendance = np.clip(np.random.normal(loc=75, scale=12, size=n_samples), 40, 100)
    attendance = np.round(attendance, 1)

    # Aptitude Score: 30–100
    aptitude_score = np.clip(np.random.normal(loc=65, scale=15, size=n_samples), 30, 100)
    aptitude_score = np.round(aptitude_score, 1)

    # Logical Reasoning: 30–100
    logical_reasoning = np.clip(np.random.normal(loc=60, scale=15, size=n_samples), 30, 100)
    logical_reasoning = np.round(logical_reasoning, 1)

    # -----------------------------------------------------------
    # STEP 2: Compute placement using a WEIGHTED FORMULA
    # -----------------------------------------------------------
    # We create a "score" based on all features.
    # Higher score → higher chance of placement.
    # We then add NOISE so the data isn't perfectly predictable.

    # Normalize each feature to 0–1 range for fair weighting
    def normalize(arr, low, high):
        """Scale values from [low, high] to [0, 1]."""
        return (arr - low) / (high - low)

    score = (
        0.25 * normalize(cgpa, 4.0, 10.0)              # CGPA is most important
        + 0.10 * normalize(iq, 80, 140)                 # IQ contributes somewhat
        + 0.10 * normalize(communication_skill, 1, 10)  # Soft skill matters
        + 0.15 * normalize(internships, 0, 4)            # Internships are key
        + 0.10 * normalize(projects, 0, 6)               # Projects help
        + 0.10 * normalize(coding_skill, 1, 10)          # Technical skill
        + 0.05 * normalize(attendance, 40, 100)           # Attendance minor factor
        + 0.08 * normalize(aptitude_score, 30, 100)       # Aptitude test
        + 0.07 * normalize(logical_reasoning, 30, 100)    # Reasoning ability
    )

    # Add random noise (simulates real-world unpredictability)
    noise = np.random.normal(loc=0, scale=0.08, size=n_samples)
    score = score + noise

    # Convert score to binary placement (1 or 0)
    # Students scoring above the median are "placed"
    # We use the 45th percentile as threshold to get ~55% placement rate
    threshold = np.percentile(score, 40)
    placement = (score >= threshold).astype(int)

    # -----------------------------------------------------------
    # STEP 3: Build the DataFrame and save to CSV
    # -----------------------------------------------------------
    df = pd.DataFrame({
        "cgpa": cgpa,
        "iq": iq,
        "communication_skill": communication_skill,
        "internships": internships,
        "projects": projects,
        "coding_skill": coding_skill,
        "attendance": attendance,
        "aptitude_score": aptitude_score,
        "logical_reasoning": logical_reasoning,
        "placement": placement
    })

    # -----------------------------------------------------------
    # STEP 4: Inject a few missing values & duplicates for realism
    # -----------------------------------------------------------
    # In real datasets, data is NEVER perfect. There are always
    # some missing values and duplicate entries.  We inject a few
    # so our preprocessing code has something to clean.

    # Randomly set ~2% of values to NaN (missing) in some columns
    for col in ["cgpa", "iq", "attendance", "aptitude_score"]:
        mask = np.random.random(n_samples) < 0.02  # 2% chance per row
        df.loc[mask, col] = np.nan

    # Add 5 duplicate rows
    duplicates = df.sample(n=5, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    # -----------------------------------------------------------
    # STEP 5: Save to CSV
    # -----------------------------------------------------------
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to '{output_path}'")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Placement distribution:\n{df['placement'].value_counts().to_string()}")

    return df


# ---------------------------------------------------------------
# MAIN: Run this file directly to generate the dataset
# ---------------------------------------------------------------
if __name__ == "__main__":
    # __name__ == "__main__" means this code ONLY runs when you
    # execute this file directly (not when importing it).

    # Get the project root directory (one level up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "data", "placement.csv")

    df = generate_placement_dataset(n_samples=1000, output_path=output_path)

    # Show a preview of the data
    print("\n📋 First 5 rows of the dataset:")
    print(df.head().to_string())
    print("\n📊 Dataset statistics:")
    print(df.describe().to_string())
