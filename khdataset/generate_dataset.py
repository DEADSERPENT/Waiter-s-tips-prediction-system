"""
Synthetic Dataset Generator — Waiter's Tips Prediction System
Generates 150,000 rows with realistic relationships between variables.

Output:
  khdataset/kaggle/waiters_tips.csv
  khdataset/huggingface/waiters_tips.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
RNG  = np.random.default_rng(SEED)
N    = 150_000

# ── Helper ─────────────────────────────────────────────────────────────────────
def choice(options, weights, size):
    return RNG.choice(options, size=size, p=weights)

# ── 1. Categorical columns ─────────────────────────────────────────────────────
sex            = choice(["Male", "Female"],        [0.56, 0.44],              N)
smoker         = choice(["No",   "Yes"],            [0.62, 0.38],              N)
day            = choice(["Sun",  "Sat", "Fri", "Thur"],
                         [0.31,   0.30,  0.20,  0.19], N)
time           = np.where(np.isin(day, ["Fri", "Thur"]),
                           choice(["Lunch", "Dinner"], [0.55, 0.45], N),
                           choice(["Lunch", "Dinner"], [0.20, 0.80], N))
payment_method = choice(["Credit Card", "Cash", "Debit Card"],
                         [0.50,          0.30,   0.20],          N)

# ── 2. Party size (correlated with day/time) ───────────────────────────────────
base_size = RNG.choice([1, 2, 3, 4, 5, 6],
                        size=N,
                        p=[0.08, 0.40, 0.22, 0.18, 0.08, 0.04])
# Weekend dinner -> slightly larger tables
weekend_dinner = ((np.isin(day, ["Sat", "Sun"])) & (time == "Dinner")).astype(int)
size = np.clip(base_size + RNG.binomial(1, 0.25 * weekend_dinner, N), 1, 6)

# ── 3. Total bill (driven by size, time, day) ──────────────────────────────────
base_bill = (
    size * RNG.uniform(8.0, 14.0, N)          # per-person spend
    + np.where(time == "Dinner", 6.0, 0.0)    # dinner premium
    + np.where(np.isin(day, ["Sat", "Sun"]), 4.0, 0.0)  # weekend premium
    + np.where(smoker == "Yes", 2.5, 0.0)     # smokers tend to order more drinks
    + RNG.normal(0, 3.5, N)                   # noise
)
total_bill = np.clip(np.round(base_bill, 2), 3.07, 120.0)

# ── 4. Service rating 1–5 (slightly better on weekends / dinner) ───────────────
service_base = (
    3.2
    + np.where(time == "Dinner",               0.25, 0.0)
    + np.where(np.isin(day, ["Sat", "Sun"]),   0.15, 0.0)
    + RNG.normal(0, 0.7, N)
)
service_rating = np.clip(np.round(service_base).astype(int), 1, 5)

# ── 5. Wait time in minutes ────────────────────────────────────────────────────
wait_base = (
    12.0
    + size * 1.8                                           # bigger party waits longer
    + np.where(time == "Dinner",               5.0, 0.0)  # dinner busier
    + np.where(np.isin(day, ["Sat", "Sun"]),   7.0, 0.0)  # weekend rush
    - (service_rating - 3) * 2.0                          # better service -> less wait
    + RNG.exponential(4.0, N)                             # random right skew
)
wait_time = np.clip(np.round(wait_base, 1), 2.0, 75.0)

# ── 6. Tip amount ──────────────────────────────────────────────────────────────
#   Base tip rate depends on service, bill, smoker status, payment method
tip_rate = (
    0.155                                                  # baseline ~15.5 %
    + (service_rating - 3) * 0.018                        # each star ±1.8 pp
    + np.where(smoker == "No",           0.008,  0.0)     # non-smokers tip more
    + np.where(time == "Dinner",         0.010,  0.0)     # dinner diners tip more
    + np.where(payment_method == "Credit Card", 0.012, 0.0)  # card -> easier to tip
    + np.where(np.isin(day, ["Sat", "Sun"]), 0.005, 0.0)
    - np.where(wait_time > 30,           0.015,  0.0)     # long wait hurts tips
    + RNG.normal(0, 0.025, N)                             # individual variation
)
tip_rate = np.clip(tip_rate, 0.05, 0.25)                  # 5–25 % range

tip = np.round(total_bill * tip_rate, 2)

# ── 7. Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "total_bill":     total_bill,
    "sex":            sex,
    "smoker":         smoker,
    "day":            day,
    "time":           time,
    "size":           size,
    "service_rating": service_rating,
    "wait_time":      wait_time,
    "payment_method": payment_method,
    "tip":            tip,
})

# ── 8. Shuffle ─────────────────────────────────────────────────────────────────
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ── 9. Sanity checks ───────────────────────────────────────────────────────────
assert df.isnull().sum().sum() == 0,          "Found missing values!"
assert len(df) == N,                           f"Row count mismatch: {len(df)}"
assert (df["tip"] > 0).all(),                 "Tip must be positive!"
tip_pct = df["tip"] / df["total_bill"]
assert tip_pct.between(0.04, 0.26).all(), (
    f"Tip % out of expected range! min={tip_pct.min():.3f} max={tip_pct.max():.3f}"
)

# ── 10. Save ───────────────────────────────────────────────────────────────────
base_dir = Path(__file__).parent
kaggle_path      = base_dir / "kaggle"      / "waiters_tips.csv"
huggingface_path = base_dir / "huggingface" / "waiters_tips.csv"

kaggle_path.parent.mkdir(parents=True, exist_ok=True)
huggingface_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(kaggle_path,      index=False)
df.to_csv(huggingface_path, index=False)

# ── 11. Summary ────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Dataset generated  ->  {N:,} rows, {len(df.columns)} columns")
print("=" * 60)
print(df.describe(include="all").T.to_string())
print()
print(f"  Avg tip rate : {(df['tip']/df['total_bill']).mean()*100:.2f} %")
print(f"  Tip range    : ${df['tip'].min():.2f}  –  ${df['tip'].max():.2f}")
print()
print(f"  Saved -> {kaggle_path}")
print(f"  Saved -> {huggingface_path}")
print("=" * 60)
