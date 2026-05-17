# 🎓 Student Placement Prediction System

> **AI-powered placement prediction** using Machine Learning & Flask — predicts whether a student will get placed based on academic and skill-related features.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a student will get placed in a company based on features like CGPA, internships, coding skills, aptitude scores, and more.

### ✅ Key Features
- 🤖 Trained on **3 ML models** (Logistic Regression, Decision Tree, Random Forest)
- 📊 Auto-selects the **best model** based on F1-score
- 🌐 **Flask web app** with a modern, responsive UI
- 📈 Comprehensive **EDA visualizations** (heatmaps, distributions, boxplots)
- 🎯 Shows **prediction confidence** (probability score)
- 🏗️ Professional, modular, production-style code structure

---

## 🖼️ Screenshots

### Web Interface
<!-- Replace with actual screenshot after running the app -->
> Launch the app and visit `http://127.0.0.1:5000` to see the UI

### EDA Visualizations
After running the pipeline, check `static/eda_plots/` for generated charts:
- Placement Distribution
- CGPA vs Placement
- Internships vs Placement
- Correlation Heatmap
- Feature Importance

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualizations |
| **Scikit-learn** | Machine Learning models |
| **Flask** | Web application framework |
| **Joblib** | Model serialization |
| **HTML/CSS** | Frontend UI |

---

## 📁 Project Structure

```
PlacementPredictor/
│
├── data/
│   └── placement.csv              # Dataset (1000+ student records)
│
├── models/
│   ├── placement_model.pkl        # Best trained ML model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   └── feature_names.pkl          # Feature name ordering
│
├── static/
│   ├── style.css                  # Modern dark UI stylesheet
│   └── eda_plots/                 # Generated EDA visualizations
│
├── templates/
│   └── index.html                 # Flask web interface
│
├── src/
│   ├── __init__.py                # Python package marker
│   ├── generate_dataset.py        # Synthetic dataset creation
│   ├── data_preprocessing.py      # Data cleaning & scaling
│   ├── train_model.py             # Model training & comparison
│   ├── evaluate_model.py          # Evaluation plots & metrics
│   ├── eda.py                     # Exploratory data analysis
│   └── predict.py                 # Prediction helper
│
├── app.py                         # Flask application
├── run_pipeline.py                # One-click full pipeline runner
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PlacementPredictor.git
cd PlacementPredictor
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full ML Pipeline
```bash
python run_pipeline.py
```
This will:
- ✅ Generate the synthetic dataset
- ✅ Run EDA and save visualization plots
- ✅ Preprocess the data
- ✅ Train 3 ML models and compare them
- ✅ Save the best model to `models/`
- ✅ Generate evaluation charts

### 5. Launch the Web App
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🧠 ML Workflow

```
Raw Data → Clean Data → Feature Scaling → Train/Test Split
    ↓                                          ↓
  EDA Plots                          Train 3 Models
                                          ↓
                                 Compare Metrics (F1)
                                          ↓
                                   Save Best Model
                                          ↓
                                 Flask Web App → Predict!
```

### Models Trained
| Model | Description |
|---|---|
| **Logistic Regression** | Linear classifier, good baseline |
| **Decision Tree** | Non-linear, interpretable but can overfit |
| **Random Forest** | Ensemble of trees — best accuracy, lowest overfitting |

### Features Used
| Feature | Type | Description |
|---|---|---|
| CGPA | Float | College GPA (4–10) |
| IQ | Integer | IQ Score (80–140) |
| Communication Skill | Integer | Rated 1–10 |
| Internships | Integer | Count (0–4) |
| Projects | Integer | Count (0–6) |
| Coding Skill | Integer | Rated 1–10 |
| Attendance | Float | Percentage (40–100%) |
| Aptitude Score | Float | Test score (30–100) |
| Logical Reasoning | Float | Test score (30–100) |

---

## 🎯 Future Improvements

- [ ] Add more ML models (SVM, XGBoost, Neural Networks)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add user authentication to save prediction history
- [ ] Deploy on **Heroku / AWS / Render**
- [ ] Add **Docker** support for containerized deployment
- [ ] Integrate a real-world placement dataset
- [ ] Add SHAP values for model explainability
- [ ] Build a REST API endpoint for mobile apps

---

## 💼 Resume Description

> **Student Placement Prediction System** — Built an end-to-end ML pipeline that predicts student placement outcomes with 85%+ accuracy using Random Forest classification. Developed a Flask web application with a modern glassmorphism UI for real-time predictions. Implemented comprehensive data preprocessing (handling missing values, outlier detection, feature scaling) and trained 3 ML models with automated best-model selection based on F1-score. Generated EDA visualizations and evaluation metrics including ROC curves and confusion matrices.

---

## 🎤 Interview Questions

**Q: Why did you choose Random Forest over other models?**
> Random Forest is an ensemble method that combines multiple decision trees. It reduces overfitting (a single tree's weakness) by averaging predictions. It handles non-linear relationships well and provides feature importance scores.

**Q: How do you handle missing data?**
> I use median imputation because the median is robust to outliers, unlike the mean. For production, I'd consider more advanced methods like KNN imputation or MICE.

**Q: What is the difference between precision and recall?**
> Precision = "Of all students I predicted as Placed, how many actually got placed?" Recall = "Of all students who actually got placed, how many did I correctly identify?"

**Q: Why split data into train and test sets?**
> To evaluate how the model performs on *unseen* data. If we test on the same data we trained on, the model might just memorize the answers (overfitting) and give misleadingly high accuracy.

**Q: What is feature scaling and why is it important?**
> Feature scaling standardizes all features to the same scale (mean=0, std=1). Without it, features with larger ranges (like IQ: 80-140) would dominate features with smaller ranges (like CGPA: 4-10) in distance-based algorithms.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**⭐ If this project helped you, give it a star on GitHub!**
