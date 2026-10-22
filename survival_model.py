import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from lifelines import KaplanMeierFitter, CoxPHFitter

PASSWORD = "Aank#1234"   # <-- put YOUR password here
os.makedirs("outputs", exist_ok=True)

# 1. Read data from PostgreSQL
engine = create_engine(f"postgresql://postgres:{PASSWORD}@localhost:5432/churn_db")
df = pd.read_sql("SELECT * FROM telco_customers", engine)
df["total_charges"] = df["total_charges"].fillna(0)

# duration = how long the customer stayed, event = did they leave (1) or not (0)
df["duration"] = df["tenure_months"].clip(lower=1)
df["event"] = df["churn_value"]

# 2. Kaplan-Meier curves by contract type
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))
for contract, group in df.groupby("contract"):
    kmf.fit(group["duration"], group["event"], label=contract)
    kmf.plot_survival_function(ci_show=False)
plt.title("Customer survival by contract type")
plt.xlabel("Months")
plt.ylabel("Probability still a customer")
plt.tight_layout()
plt.savefig("outputs/km_by_contract.png", dpi=150)
plt.close()

# 3. Cox model with a few simple features
cox_df = pd.DataFrame({
    "duration": df["duration"],
    "event": df["event"],
    "monthly_charges": df["monthly_charges"],
    "senior_citizen": (df["senior_citizen"] == "Yes").astype(int),
    "paperless_billing": (df["paperless_billing"] == "Yes").astype(int),
    "fiber_optic": (df["internet_service"] == "Fiber optic").astype(int),
    "no_internet": (df["internet_service"] == "No").astype(int),
    "one_year": (df["contract"] == "One year").astype(int),
    "two_year": (df["contract"] == "Two year").astype(int),
})
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col="duration", event_col="event")
cph.print_summary()
print("\nConcordance index:", round(cph.concordance_index_, 3))

# 4. Expected remaining lifetime per customer (used later for CLV)
surv = cph.predict_survival_function(cox_df.drop(columns=["duration", "event"]), times=list(range(1, 73)))
df["expected_months_72"] = surv.sum().values   # sum of survival probability = expected months within 6 years
df[["customerid", "expected_months_72"]].to_csv("outputs/expected_lifetime.csv", index=False)
print("Saved outputs/km_by_contract.png and outputs/expected_lifetime.csv")