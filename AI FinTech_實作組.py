# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# %%
train_all = pd.read_csv('./train.csv')
train_all.head()
train_all = train_all.reindex(sorted(train_all.columns), axis=1)

# %%
train_y = train_all.is_fraud

# %%
train_X = train_all.drop(['is_fraud','device_fraud_cases'],
                                 axis='columns')

# %%
train_X_num_describe = train_X.describe()
train_X_num = train_X[train_X_num_describe.columns]
train_X_cat = train_X[train_X.columns.difference(train_X_num_describe.columns)]

# %%
test_all = pd.read_csv('./X_test.csv')
test_all = test_all.reindex(sorted(test_all), axis=1)

# %%
test_X = test_all.drop(['device_fraud_cases'],
                                 axis='columns')

# %% [markdown]
# ## Replacement

# %%
col_replace = ['current_address_months', 'online_session_duration', 'previous_address_months', 
    'previous_bank_account_months','unique_device_emails_8w']

train_X[col_replace] = train_X[col_replace].replace(-1, np.nan)
test_X[col_replace] = test_X[col_replace].replace(-1, np.nan)

# %%
col_cat_idx = train_X_cat.columns

# %%
for c in col_cat_idx:
    le = LabelEncoder()
    le.fit(train_X[c])
    train_X[c] = le.transform(train_X[c])

for c in col_cat_idx:
    le = LabelEncoder()
    le.fit(test_X[c])
    test_X[c] = le.transform(test_X[c])

# %% [markdown]
# ## Select features!

# %%
train_X_features = train_X
test_X_features = test_X

# %% [markdown]
# ## XGBoost and RandomizedSearchCV

# %%
param_grid = {'max_depth': [4],
              'learning_rate': [0.12],
              'n_estimators': [100,200,220,230]
              }

# %%
estimator = XGBClassifier(objective= 'binary:logistic', scale_pos_weight=8)

random_search = RandomizedSearchCV(estimator=estimator, 
                                   param_distributions=param_grid, 
                                   n_iter=15,
                                   scoring='f1', 
                                   refit='f1', 
                                   n_jobs=-1, 
                                   cv=5, 
                                   verbose=1)

random_search.fit(train_X_features, train_y)

print("Best parameters: ", random_search.best_params_)
print(f"Mean F1 score (cross-validation): {random_search.best_score_:.2%}")
print(f"Best F1 score at the weight {8} on the training data: {random_search.score(train_X, train_y):.2%}")

# %%
best_model = random_search.best_estimator_
print("Best estimator:", best_model)
print(f"F1 on Train data: {random_search.score(train_X_features, train_y):.2%}")

# %% [markdown]
# ## Output to the submission file

# %%
test_y_pred  = random_search.predict(test_X_features)
test_y_pred_df = pd.DataFrame(data=test_y_pred, columns = ['is_fraud'])
test_y_pred_df.to_csv('./submission_y_pred_1121_2491.csv', index=False)


