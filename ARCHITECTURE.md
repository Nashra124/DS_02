Script             |       Input	                       |           Output
____________________________________________________________________________________________________________          
load_to_postgres.py    |Telco_customer_churn.csv	           |table telco_customers
churn_model.py	       |database table	                       |shap_summary.png, test_predictions.csv
survival_model.py	   |database table	                       |km_by_contract.png, expected_lifetime.csv
uplift_model.py	       |database table	                       |uplift_by_group.csv, uplift_scores.csv
final_table.py	       |table plus the three CSVs above	       |customers_final.csv
app.py	               |customers_final.csv and images	       |the dashboard