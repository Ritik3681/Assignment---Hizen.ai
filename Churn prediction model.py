#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import shap


# In[3]:


get_ipython().system('pip install xgboost')


# In[4]:


import xgboost as xgb
print(xgb.__version__)


# In[5]:


get_ipython().system('pip install shap')


# In[6]:


data = pd.read_csv('events.csv')

# Preprocessing
data['event_time'] = pd.to_datetime(data['event_time'])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)


# In[7]:


sns.countplot(x='event_type', data=data)
plt.title('Event Type Distribution')
plt.show()


# In[8]:


from datetime import timedelta

# Define churn date
define_churn_date = data['event_time'].max() - timedelta(days=30)


data['churn'] = data.groupby('user_id')['event_time'].transform(lambda x: x.max() < define_churn_date)
data['churn'] = data['churn'].astype(int)


# In[9]:


# Feature engineering
features = data.groupby('user_id').agg({
    'event_time': [
        lambda x: (define_churn_date - x.max()).days,
        lambda x: x.nunique()
    ],
    'event_type': [
        lambda x: (x == 'view').sum(),
        lambda x: (x == 'cart').sum(),
        lambda x: (x == 'purchase').sum()
    ],
    'price': ['sum', 'mean']
}).reset_index()

features.columns = ['user_id', 'recency', 'session_count', 'views', 'carts', 'purchases', 'total_spent', 'avg_spent']


features = features.merge(data[['user_id', 'churn']].drop_duplicates(), on='user_id')


# In[10]:


# Prepare train/test split
X = features.drop(columns=['user_id', 'churn'])
y = features['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))


# In[12]:


import shap


explainer = shap.TreeExplainer(model)  


shap_values = explainer.shap_values(X_test)

# Verify the structure of shap_values
print(f"Type of shap_values: {type(shap_values)}")
print(f"Length of shap_values (for list): {len(shap_values) if isinstance(shap_values, list) else 'N/A'}")
print(f"Shape of shap_values (for array): {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")


if isinstance(shap_values, list) and len(shap_values) > 1:
    shap.summary_plot(shap_values[1], X_test)
else:
    shap.summary_plot(shap_values, X_test)


# In[13]:


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)


# In[14]:


import joblib

# Save features and model
features.to_csv('user_features.csv', index=False)
joblib.dump(model, 'churn_model.pkl')


# In[17]:


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)


# In[ ]:




