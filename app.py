import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn & CLV ROI Simulator", layout="wide")
df = pd.read_csv("outputs/customers_final.csv")

st.title("Customer Churn, Lifetime Value and Retention ROI")
tab1, tab2, tab3 = st.tabs(["ROI simulator", "Segments", "Model results"])

with tab1:
    c1, c2, c3 = st.columns(3)
    discount = c1.slider("Discount offered (% of monthly bill)", 5, 30, 10)
    share = c2.slider("Share of customers targeted (%)", 5, 100, 20)
    strategy = c3.selectbox("Who to target", ["Highest uplift (responders)", "Highest churn risk", "Random"])
    months = st.slider("Months the discount lasts", 1, 12, 6)

    n = int(len(df) * share / 100)
    if strategy.startswith("Highest uplift"):
        target = df.nlargest(n, "uplift_score")
    elif strategy.startswith("Highest churn"):
        target = df.nlargest(n, "churn_probability")
    else:
        target = df.sample(n, random_state=1)

    # Assumption: uplift scores were estimated for a 10% discount; effect scales with discount size
    effect = np.clip(target["uplift_score"] * discount / 10, 0, 1)
    saved_customers = effect.sum()
    savings = (effect * target["clv"]).sum()
    cost = (target["monthly_charges"] * discount / 100 * months).sum()
    net = savings - cost

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers targeted", f"{n:,}")
    m2.metric("Expected customers saved", f"{saved_customers:,.0f}")
    m3.metric("Expected savings", f"${savings:,.0f}")
    m4.metric("Offer cost", f"${cost:,.0f}")
    st.metric("Net benefit", f"${net:,.0f}", delta=f"ROI {net / cost * 100:,.0f}%" if cost else None)
    st.caption("Offer response is SIMULATED (Telco has no real offer data). "
               "CLV = monthly charges x expected lifetime months from the Cox model.")

with tab2:
    seg = df.groupby("cluster").agg(customers=("clv", "size"), avg_clv=("clv", "mean"),
                                    avg_monthly=("monthly_charges", "mean"),
                                    avg_tenure=("tenure_months", "mean"),
                                    avg_churn_risk=("churn_probability", "mean")).round(2)
    st.dataframe(seg)
    st.bar_chart(seg["avg_clv"])

with tab3:
    st.write("**Churn classifier (LightGBM):** AUC 0.852, precision 0.654, recall 0.527")
    st.write("**Survival model (Cox):** concordance index 0.826")
    try:
        st.write("**Uplift by group (simulated offer):**")
        st.dataframe(pd.read_csv("outputs/uplift_by_group.csv"))
    except FileNotFoundError:
        pass
    st.image("outputs/shap_summary.png", caption="SHAP: what drives churn")
    st.image("outputs/km_by_contract.png", caption="Survival by contract type")