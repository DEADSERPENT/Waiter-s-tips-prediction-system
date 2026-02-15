# QUICK START GUIDE

## Waiter's Tips Prediction System

### 🚀 Getting Started

Follow these steps to set up and run the complete system:

---

## 1️⃣ Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- joblib

---

## 2️⃣ Running the System

### Option A: Run Complete Pipeline (Recommended)

Execute the entire system in one go:

```bash
cd src
python main.py
```

This will:
1. ✓ Download the dataset from Kaggle
2. ✓ Preprocess the data
3. ✓ Generate EDA visualizations
4. ✓ Train all 6 models
5. ✓ Evaluate and compare models
6. ✓ Save trained models
7. ✓ Generate result visualizations
8. ✓ Run demo predictions

**Output Locations:**
- Dataset: `data/tips.csv`
- Models: `models/*.pkl`
- Visualizations: `results/*.png`

---

### Option B: Run Step-by-Step

#### Step 1: Download Dataset
```bash
cd src
python download_data.py
```

#### Step 2: Train Models
```bash
python model_training.py
```

#### Step 3: Make Predictions
```bash
python prediction.py
```

---

### Option C: Use Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook
```

Then open: `notebooks/tips_prediction_analysis.ipynb`

---

## 3️⃣ Making Predictions

### Interactive Mode

```bash
cd src
python prediction.py
```

You'll be prompted to enter:
- Total Bill ($)
- Gender (Male/Female)
- Smoker (Yes/No)
- Day (Thur/Fri/Sat/Sun)
- Time (Lunch/Dinner)
- Party Size

**Example:**
```
Total Bill ($): 25.50
Gender (Male/Female): Male
Smoker (Yes/No): No
Day (Thur/Fri/Sat/Sun): Sat
Time (Lunch/Dinner): Dinner
Party Size: 2

→ Predicted Tip: $3.85 (15.1%)
```

---

## 4️⃣ Project Structure

```
Waiter's tips prediction system/
│
├── data/                      # Dataset files
│   └── tips.csv              # Downloaded tips dataset
│
├── src/                       # Source code
│   ├── download_data.py      # Dataset downloader
│   ├── data_preprocessing.py # Data preprocessing
│   ├── model_training.py     # Model training
│   ├── visualization.py      # Visualization tools
│   ├── prediction.py         # Prediction system
│   └── main.py              # Complete pipeline
│
├── models/                    # Trained models
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── lasso_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── results/                   # Visualizations
│   ├── data_distribution.png
│   ├── categorical_analysis.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── predictions_vs_actual_*.png
│
├── notebooks/                 # Jupyter notebooks
│   └── tips_prediction_analysis.ipynb
│
├── requirements.txt           # Dependencies
├── README.md                 # Project documentation
├── PROJECT_SYNOPSIS.md       # Academic synopsis
└── QUICK_START.md           # This file
```

---

## 5️⃣ Understanding the Output

### Model Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in dollars
  - Lower is better
  - Example: MAE = $0.75 means predictions are off by $0.75 on average

- **RMSE (Root Mean Squared Error)**: Standard deviation of errors
  - Lower is better
  - Penalizes large errors more than MAE

- **R² Score**: How well the model fits the data
  - Range: 0 to 1 (higher is better)
  - 0.45 = Model explains 45% of variance
  - > 0.40 is considered good for this dataset

### Expected Results

Based on the dataset, you should see:
- **Best Model**: Random Forest or Gradient Boosting
- **R² Score**: 0.40 - 0.50
- **RMSE**: $0.90 - $1.20

---

## 6️⃣ Visualizations Explained

### 1. Data Distribution
- Shows how bill amounts, tips, and party sizes are distributed
- Helps identify outliers and patterns

### 2. Categorical Analysis
- Box plots showing tips by gender, smoker status, day, and time
- Reveals which categories tend to tip more

### 3. Correlation Heatmap
- Shows relationships between all numerical features
- Strong correlation between total_bill and tip

### 4. Model Comparison
- Bar charts comparing all 6 models
- Helps identify the best performing model

### 5. Actual vs Predicted
- Scatter plot of actual vs predicted tips
- Points closer to the red line = better predictions

### 6. Residual Analysis
- Shows prediction errors
- Should be randomly distributed around zero

---

## 7️⃣ Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'src'"
**Solution:** Make sure you're in the project root directory
```bash
cd "d:\PROJECTS\Waiter's tips prediction system"
```

### Issue: "Dataset not found"
**Solution:** Run the download script first
```bash
cd src
python download_data.py
```

### Issue: "Model not found"
**Solution:** Train the models first
```bash
cd src
python model_training.py
```

---

## 8️⃣ For Academic Submission

### Mid-Term Evaluation (18 Mar 2026)

Submit:
1. ✓ `PROJECT_SYNOPSIS.md` - Project synopsis
2. ✓ `data/tips.csv` - Dataset
3. ✓ `src/data_preprocessing.py` - Preprocessing code
4. ✓ At least one trained model
5. ✓ Basic visualizations

**Demo:** Run `python src/main.py` to show working system

### Final Evaluation (3 April 2026)

Submit:
1. ✓ Complete source code
2. ✓ All 6 trained models
3. ✓ All visualizations
4. ✓ Jupyter notebook with analysis
5. ✓ Final report
6. ✓ Working demonstration

**Demo:** 
- Run complete pipeline
- Show model comparison
- Make live predictions
- Explain visualizations

---

## 9️⃣ Customization

### Add New Features

Edit `src/data_preprocessing.py`:
```python
# Add custom features
df['custom_feature'] = df['total_bill'] * df['size']
```

### Try Different Models

Edit `src/model_training.py`:
```python
from sklearn.svm import SVR

models['SVM'] = SVR(kernel='rbf')
```

### Adjust Hyperparameters

```python
RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=15,      # Deeper trees
    random_state=42
)
```

---

## 🔟 Next Steps

1. **Explore the Jupyter Notebook**
   - Interactive analysis
   - Detailed explanations
   - Custom experiments

2. **Experiment with Features**
   - Try different feature combinations
   - Create new engineered features

3. **Optimize Models**
   - Tune hyperparameters
   - Try ensemble methods

4. **Deploy the System**
   - Create a web interface
   - Build a REST API
   - Mobile application

---

## 📞 Support

For issues or questions:
1. Check the README.md
2. Review the Jupyter notebook
3. Examine the code comments
4. Check scikit-learn documentation

---

## ✅ Checklist

Before submission, ensure:
- [ ] All dependencies installed
- [ ] Dataset downloaded successfully
- [ ] All models trained
- [ ] Visualizations generated
- [ ] Predictions working
- [ ] Documentation complete
- [ ] Code is well-commented
- [ ] Results are reproducible

---

**Happy Coding! 🎉**
