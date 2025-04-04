# Read data set
import pandas as pd
n_classes = 2 # binary
n_rows = 300000
df = pd.read_csv("/Users/matthewyacovone/Desktop/ml-by-example/PACT/train", nrows=n_rows)
print(df.head(5))

# The target variable is the click column
Y = df['click'].values

# Features
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
print(X.shape)

# Split the data into training and testing sets
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# Initialize OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
print(X_train_enc[0]) # Each converted sample is a sparse vector

# Transform the testing set using the trained one-hot encoder
X_test_enc = enc.transform(X_test)

# Find class distrubtion to check for imbalances
import numpy as np
class_weights = {}

values, counts = np.unique(Y, return_counts=True)
for value, count in zip(values, counts):
    print(f'Number of users in {value} class: {count}')
    ctr_percent = (count / len(Y)) * 100
    print(f"{ctr_percent:.1f}% of training samples are {value}.")

    # specify class weights
    weight = n_rows / (n_classes * count)
    class_weights[value] = weight

print(class_weights)

# Specify hyperparameters
from sklearn.tree import DecisionTreeClassifier
# parameters = {'max_depth': [3, 10, None]} # original parameters
# parameters = {'max_depth': [3, 8, 10, 15, 20, 30, None], 'min_samples_split': [10, 30, 50]} # tuning
parameters = {'max_depth': [20], 'min_samples_split': [50]} # best parameters after tuning
  
# Initialize decision tree
decision_tree = DecisionTreeClassifier(criterion='gini', class_weight=class_weights, min_samples_split=30)

# 3-fold cross validation since training set is relatively small
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')

grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)

# Use the model with the optimal parameter to predict any future test cases
decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]

from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set using a decision tree is: {roc_auc_score(Y_test, pos_prob):.3f}')

# Compare decision tree results to random selection
pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test), int(len(Y_test) *  51211.0/300000), replace=False)
pos_prob[click_index] = 1

print(f'The ROC AUC on testing set using random sampling is: {roc_auc_score(Y_test, pos_prob):.3f}')

# Compare to a random forest ensemble
from sklearn.ensemble import RandomForestClassifier

parameters = {'max_depth': [None], 'min_samples_split': [50]} # best parameters after tuning
random_forest = RandomForestClassifier(n_estimators=100, max_features="sqrt", criterion='gini', n_jobs=-1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print(f'The ROC AUC on testing set with a random forest is: {roc_auc_score(Y_test, pos_prob):.3f}')

# Compare to Gradient Boosted Trees
import xgboost as xgb
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=1000)

# Train the GBT
model.fit(X_train_enc, Y_train)

pos_prob = model.predict_proba(X_test_enc)[:, 1]
print(f'The ROC AUC on testing set using GBT is: {roc_auc_score(Y_test, pos_prob):.3f}')