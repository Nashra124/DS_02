import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

PASSWORD = "Aank#1234"   # <-- YOUR password
os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(42)

engine = create_engine(f"postgresql://postgres:{PASSWORD}@localhost:5432/churn_db")
df = pd.read_sql("SELECT * FROM telco_customers", engine)
df["total_charges"] = df["total_charges"].fillna(0)

# SIMULATED offer: half the customers randomly get a retention offer
df["treatment"] = rng.integers(0, 2, len(df))
# SIMULATED response: the offer saves some churners, mostly month-to-month, high-bill customers
persuadable = (df["contract"] == "Month-to-month") & (df["monthly_charges"] > df["monthly_charges"].median())
save_prob = np.where(persuadable, 0.40, 0.05)
saved = (df["treatment"] == 1) & (df["churn_value"] == 1) & (rng.random(len(df)) < save_prob)
df["retained"] = np.where(saved, 1, 1 - df["churn_value"])

drop = ["churn_value", "churn_label", "churn_score", "churn_reason", "cltv", "customerid",
        "city", "zip_code", "latitude", "longitude", "treatment", "retained"]
X = pd.get_dummies(df.drop(columns=[c for c in drop if c in df.columns]), drop_first=True).astype(float)

idx_tr, idx_te = train_test_split(df.index, test_size=0.3, random_state=42, stratify=df["treatment"])

def fit(rows):
    m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=8, verbose=-1)
    m.fit(X.loc[rows], df.loc[rows, "retained"])
    return m

# T-learner: one model for treated, one for control; uplift = difference
m_treat = fit(idx_tr[df.loc[idx_tr, "treatment"] == 1])
m_ctrl = fit(idx_tr[df.loc[idx_tr, "treatment"] == 0])
df["uplift_score"] = m_treat.predict_proba(X)[:, 1] - m_ctrl.predict_proba(X)[:, 1]

# Check on unseen customers
te = df.loc[idx_te].copy()
te["group"] = pd.qcut(te["uplift_score"].rank(method="first", ascending=False), 5,
                      labels=["Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%"])
rows = []
for g, s in te.groupby("group", observed=True):
    t = s[s.treatment == 1]["retained"].mean()
    c = s[s.treatment == 0]["retained"].mean()
    rows.append({"group": g, "treated_retention": round(t, 3),
                 "control_retention": round(c, 3), "real_uplift": round(t - c, 3)})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("outputs/uplift_by_group.csv", index=False)
df[["customerid", "uplift_score"]].to_csv("outputs/uplift_scores.csv", index=False)
print("Saved outputs/uplift_scores.csv")