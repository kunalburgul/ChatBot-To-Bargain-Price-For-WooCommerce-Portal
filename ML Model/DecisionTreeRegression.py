 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:19:45 2019

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


# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics
regressor = DecisionTreeRegressor(random_state=42)
r = regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

import pickle
pickle.dump(regressor, open("model.pkl","wb"))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(r, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("tree.pdf") 


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
  a = (i*0.01)
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
