"""
Generate or load the Tips dataset (1,000 rows).
If data/tips.csv already exists with ≤ 1000 rows it is kept as-is.
Otherwise a synthetic 1,000-row dataset is generated from a realistic
distribution that mirrors the seaborn tips dataset.
"""

import os
import numpy as np
import pandas as pd


def download_tips_dataset():
    """
    Return a 1,000-row tips DataFrame and save it to data/tips.csv.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'tips.csv')

    # Use existing file only if it has exactly the right number of rows
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) == 1000:
            print(f"[OK] Dataset found ({len(existing)} rows): {output_path}")
            _print_info(existing)
            return existing
        print(f"Existing file has {len(existing)} rows — regenerating to 1,000 rows.")

    tips_df = _generate_synthetic(n=1000, seed=42)
    tips_df.to_csv(output_path, index=False)
    print(f"[OK] Dataset saved: {output_path}")
    _print_info(tips_df)
    return tips_df


def _generate_synthetic(n=1000, seed=42):
    """Generate a realistic synthetic tips dataset of n rows."""
    rng = np.random.default_rng(seed)

    # Categorical features (match seaborn proportions)
    sex    = rng.choice(['Male', 'Female'], n, p=[0.64, 0.36])
    smoker = rng.choice(['No', 'Yes'],      n, p=[0.62, 0.38])
    day    = rng.choice(['Thur', 'Fri', 'Sat', 'Sun'], n, p=[0.25, 0.10, 0.35, 0.30])
    time   = rng.choice(['Dinner', 'Lunch'],            n, p=[0.72, 0.28])

    # Party size (1-6, mostly 2)
    size = rng.choice([1, 2, 3, 4, 5, 6], n,
                      p=[0.04, 0.45, 0.20, 0.18, 0.08, 0.05])

    # Total bill: base per-person spend + noise
    base_per_person = rng.uniform(6.0, 14.0, n)
    dinner_adj      = np.where(time == 'Dinner', 1.25, 1.0)
    weekend_adj     = np.where(np.isin(day, ['Sat', 'Sun']), 1.10, 1.0)
    total_bill = size * base_per_person * dinner_adj * weekend_adj
    total_bill += rng.normal(0, 1.5, n)
    total_bill  = np.round(np.clip(total_bill, 3.07, 58.0), 2)

    # Tip: ~15-18 % of bill with variability
    base_rate   = rng.normal(0.165, 0.045, n)
    smoker_adj  = np.where(smoker == 'Yes', rng.normal(1.04, 0.08, n), 1.0)
    male_adj    = np.where(sex == 'Male',   rng.normal(1.03, 0.06, n), 1.0)
    tip = total_bill * base_rate * smoker_adj * male_adj
    tip += rng.normal(0, 0.15, n)
    tip  = np.round(np.clip(tip, 1.00, 10.0), 2)

    df = pd.DataFrame({
        'total_bill': total_bill,
        'tip':        tip,
        'sex':        sex,
        'smoker':     smoker,
        'day':        day,
        'time':       time,
        'size':       size,
    })
    return df


def _print_info(df):
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total Records : {len(df)}")
    print(f"Columns       : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStatistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    download_tips_dataset()
