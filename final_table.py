import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PASSWORD = "Aank#1234"   

engine = create_engine(f"postgresql://postgres:{PASSWORD}@localhost:5432/churn_db")
df = pd.read_sql("SELECT * FROM telco_customers", engine)
df["total_charges"] = df["total_charges"].fillna(0)

# Churn probability for EVERY customer (5-fold, so each score comes from a model that did not see that customer)
drop = ["churn_value", "churn_label", "churn_score", "churn_reason", "cltv", "customerid",
        "city", "zip_code", "latitude", "longitude"]
X = df.drop(columns=[c for c in drop if c in df.columns])
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = X[col].astype("category")
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=30, random_state=42, verbose=-1)
df["churn_probability"] = cross_val_predict(model, X, df["churn_value"], cv=5,
                                            method="predict_proba")[:, 1]

# Merge survival + uplift results
df = df.merge(pd.read_csv("outputs/expected_lifetime.csv"), on="customerid")
df = df.merge(pd.read_csv("outputs/uplift_scores.csv"), on="customerid")

# CLV = monthly bill x expected months of life (simple version)
df["clv"] = df["monthly_charges"] * df["expected_months_72"]

# Segmentation with KMeans
feats = df[["clv", "monthly_charges", "tenure_months", "churn_probability"]]
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(
    StandardScaler().fit_transform(feats))
print(df.groupby("cluster")[["clv", "monthly_charges", "tenure_months", "churn_probability"]]
        .mean().round(2).assign(customers=df.groupby("cluster").size()))

df[["customerid", "contract", "monthly_charges", "tenure_months", "churn_probability",
    "uplift_score", "expected_months_72", "clv", "cluster"]].to_csv("outputs/customers_final.csv", index=False)
print("Saved outputs/customers_final.csv")