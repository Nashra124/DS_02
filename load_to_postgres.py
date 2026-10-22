import pandas as pd
from sqlalchemy import create_engine, text

PASSWORD = "Aank#1234"   # <-- put YOUR password here

# 1. Create the database if it doesn't exist
admin = create_engine(
    f"postgresql://postgres:{PASSWORD}@localhost:5432/postgres",
    isolation_level="AUTOCOMMIT",
)
with admin.connect() as conn:
    exists = conn.execute(
        text("SELECT 1 FROM pg_database WHERE datname = 'churn_db'")
    ).scalar()
    if not exists:
        conn.execute(text("CREATE DATABASE churn_db"))

# 2. Read and clean the CSV (it is in the same folder as this file)
df = pd.read_csv("Telco_customer_churn.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df = df.drop(columns=["count", "country", "state", "lat_long"], errors="ignore")

# 3. Load into PostgreSQL
engine = create_engine(f"postgresql://postgres:{PASSWORD}@localhost:5432/churn_db")
df.to_sql("telco_customers", engine, if_exists="replace", index=False)

# 4. Read it back through SQL to prove it worked
print(pd.read_sql("SELECT COUNT(*) AS rows_loaded FROM telco_customers", engine))