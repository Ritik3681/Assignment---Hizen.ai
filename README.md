Overview
This project focuses on predicting customer churn based on user interaction data from an event-based dataset. 
The goal is to build a machine learning pipeline that processes the data, engineers relevant features, and trains a
predictive model to classify users as churned or active. Additionally, the model's predictions are explained
using SHAP (SHapley Additive exPlanations) for interpretability.

Steps in the Pipeline: -
1) Data Loading and Preprocessing

Load event data from a CSV file.
Convert event timestamps to datetime format.
Remove duplicate records.
Handle missing values by either dropping them or filling with default values.

2) Exploratory Data Analysis (EDA)

Visualize the distribution of event types (e.g., views, carts, purchases) using a count plot.
Summarize key user activity metrics such as event counts, session recency, and spending patterns.

3) Feature Engineering
Aggregate user activity data using features such as:
Recency: Days since the last recorded event.
Session Count: Unique session activities by each user.
Views, Carts, and Purchases: Counts of specific actions.
Total and Average Spending: Calculated from price data.
Assign a churn label based on inactivity for more than 30 days.

4) Model Training
Split the dataset into training and testing sets (80-20 split).
Train a RandomForestClassifier to predict churn.
Evaluate the model using:
Classification report (Precision, Recall, F1-Score).
ROC AUC Score.

5) Explainability with SHAP
Use SHAP values to interpret the feature importance.
Generate a SHAP summary plot to visualize the impact of features on churn prediction.

6) Model and Data Export
Save the engineered feature dataset (user_features.csv) for reproducibility.
Export the trained model as a .pkl file for deployment.

***Key Dependencies***


1) python Libraries :
pandas for data manipulation.
numpy for numerical operations.
matplotlib and seaborn for data visualization.
scikit-learn for machine learning models and evaluation.
xgboost for alternative model training (optional).
shap for explainability.
joblib for model persistence.


***How to Run***
***Install Dependencies***:

->pip install -r requirements.txt
Run the Script: Ensure the events.csv file is in the same directory as the script and execute:

->python churn_prediction.py


***Outputs:***

->Model evaluation metrics printed in the console.
->SHAP summary plot displayed.

->Generated files:
churn_model.pkl: Trained Random Forest model.

->Files in Repository

churn_prediction.py: Main script for feature engineering, model training, and evaluation.

***Results and Insights***

-> Key Features:
Recency, session count, and purchase behavior were the most impactful predictors of churn.

->Model Performance:
High ROC AUC score indicates strong predictive capability.
SHAP analysis revealed the model's decision-making process, enabling transparency.

-> Future Improvements
Incorporate time-series analysis for dynamic churn prediction.
Experiment with additional algorithms such as XGBoost or LightGBM.
Deploy the model using a REST API or integrate it with a customer analytics dashboard.
