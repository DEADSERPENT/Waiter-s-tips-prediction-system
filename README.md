# Waiter's Tips Prediction System

> End-to-end ML system for predicting restaurant tip amounts — 6 tuned regression models, SHAP + LIME explainability, and an interactive Streamlit web app.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

---

## Quick Start

```bash
# 1. Install dependencies
pip install --upgrade setuptools wheel
pip install -r requirements.txt

# 2. Train all models
python src/main.py

# 3. Launch the web app
streamlit run app.py
```

> **Windows:** double-click `run_app.bat` — installs and launches automatically.

---

## Dataset

5000 restaurant transactions · 7 features · no missing values · avg tip $2.99 (~15%)

| Feature | Type | Description |
|---|---|---|
| `total_bill` | Numerical | Total bill (USD) |
| `tip` | Numerical | Tip amount (USD) — **target** |
| `sex` / `smoker` | Categorical | Customer demographics |
| `day` / `time` | Categorical | Dining period |
| `size` | Numerical | Party size |

---

## Models and Performance

All models tuned with `GridSearchCV` (5-fold CV).

| Model | R² | RMSE ($) | MAE ($) |
|---|---|---|---|
| **Random Forest** | **0.480** | **0.950** | **0.720** |
| Gradient Boosting | 0.470 | 0.970 | 0.730 |
| Linear Regression | 0.450 | 1.000 | 0.750 |
| Ridge Regression | 0.440 | 1.010 | 0.760 |
| Lasso Regression | 0.430 | 1.020 | 0.770 |
| Decision Tree | 0.380 | 1.150 | 0.850 |

---

## Web App Pages

| Page | Description |
|---|---|
| Predict Tip | Instant prediction with 18% industry benchmark comparison |
| Data Explorer | EDA — histograms, box plots, correlation heatmap, scatter plots |
| Model Comparison | Side-by-side R², RMSE, MAE charts for all 6 models |
| SHAP Explainability | Feature importance, beeswarm, waterfall, and dependence plots |
| LIME Explanation | Local surrogate explanation for any custom input |

---

## Project Structure

```
├── src/
│   ├── main.py                 # End-to-end pipeline
│   ├── data_preprocessing.py   # Encoding and train/test split
│   ├── model_training.py       # GridSearchCV training
│   ├── explainability.py       # SHAP + LIME module
│   └── prediction.py           # CLI prediction interface
├── models/                     # Serialised .pkl models (auto-generated)
├── data/tips.csv               # Dataset (auto-downloaded)
├── app.py                      # Streamlit web application
├── run_app.bat                 # Windows one-click launcher
└── requirements.txt
```

---

## Tech Stack

`scikit-learn` · `SHAP` · `LIME` · `Streamlit` · `pandas` · `matplotlib` · `seaborn`

---

## Academic Information

**M.Tech Mini-Project — Machine Learning, 2026**

| Milestone | Date |
|---|---|
| Synopsis Submission | 24 Feb 2026 |
| Mid-Term Evaluation | 18 Mar 2026 |
| Final Evaluation | 3 Apr 2026 |
