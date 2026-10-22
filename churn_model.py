import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report

PASSWORD = "Aank#1234"   # <-- put YOUR password here
os.makedirs("outputs", exist_ok=True)

# 1. Read the data FROM PostgreSQL
engine = create_engine(f"postgresql://postgres:{PASSWORD}@localhost:5432/churn_db")
df = pd.read_sql("SELECT * FROM telco_customers", engine)
print("Loaded from database:", df.shape)

# 2. Feature engineering (new columns that help the model)
df["total_charges"] = df["total_charges"].fillna(0)

service_cols = ["phone_service", "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies"]
service_cols = [c for c in service_cols if c in df.columns]
df["num_services"] = sum((df[c] == "Yes").astype(int) for c in service_cols)

df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
df["charge_vs_average"] = df["monthly_charges"] / df["avg_monthly_spend"].clip(lower=1)
df["tenure_group"] = pd.cut(df["tenure_months"], bins=[-1, 12, 24, 48, 100],
                            labels=["0-12", "13-24", "25-48", "49+"]).astype(str)

# 3. Choose target and remove columns that would LEAK the answer
y = df["churn_value"]
leak_or_useless = ["churn_value", "churn_label", "churn_score", "churn_reason", "cltv",
                   "customerid", "city", "zip_code", "latitude", "longitude"]
X = df.drop(columns=[c for c in leak_or_useless if c in df.columns])

for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# 4. Train / test split (80% / 20%, same churn ratio in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Train LightGBM
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=30, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# 6. Evaluate on the unseen test set
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\nAUC      :", round(roc_auc_score(y_test, proba), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("\n", classification_report(y_test, pred))

# 7. SHAP: what drives churn?
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)
if isinstance(sv, list):      # some versions return one array per class
    sv = sv[1]
shap.summary_plot(sv, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150)
plt.close()

# 8. Save results for later steps (CLV, uplift, Streamlit)
out = X_test.copy()
out["churn_probability"] = proba
out["actual_churn"] = y_test.values
out.to_csv("outputs/test_predictions.csv", index=False)
print("\nSaved outputs/shap_summary.png and outputs/test_predictions.csv")