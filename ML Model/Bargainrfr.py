#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:32:25 2019

@author: kunal
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Customerdata.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Random Forest Regression to the dataset
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 600, random_state = 42)
regressor.fit(X_train, y_train)


# Predicting a new result 
y_pred =  regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# =============================================================================
# param_grid = { 
#     'n_estimators': [10, 20, 30, 35],
#     'max_features': ['auto']
# }
# 
# regressor1 = GridSearchCV(estimator=regressor, param_grid=param_grid, cv= 5)
# regressor1.fit(X, y)
# print(regressor1.best_params_)
# =============================================================================




regressor = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(regressor.get_params())



from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
regressor = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_






def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)
















# Accuracy prediction
from array import *
lst_y_test = y_test.tolist()
lst_y_pred = y_pred.tolist()

lst_accurate = 0
lst_error = 0
correct_count = 0
actual = 0
error_count = 0

for i, j in zip(lst_y_test, lst_y_pred):
  actual+=1
  a = (i*0.05)
  l = i - a
  m = i + a
  if(l<j<m):
    #print(l," - ", j ," - ",m)
    lst_accurate += j
    correct_count += 1
  else:
    lst_error += j
    error_count += 1
print("Actaul count:", actual)
print("Predicted count for model (-5%,+5%) :", correct_count)
print("Error count:", error_count)
print("Acc:",(correct_count/actual)*100)




# =============================================================================
# print("Sum of predicted count values", lst_accurate)
# print("Sum of Error values:", lst_error)
# print("Total Sum:", lst_accurate + lst_error)
# total_accuracy = (lst_accurate + lst_error)/actual
# predicted_accuracy = lst_accurate/correct_count
# print("Total accuracy:", total_accuracy)
# print("Predicted accuracy:", predicted_accuracy)
# accuracy = total_accuracy - predicted_accuracy
# print("Accuracy of the model", 100 - accuracy)
# 
# =============================================================================
print("Acc:",(correct_count/actual)*100)

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

