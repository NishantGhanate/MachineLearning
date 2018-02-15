# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:10:20 2018

@author: NishantGhanate
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# X = indepent variables 
X = dataset.iloc[:, :-1].values # Selecting all coulums expect last column
# y = depent variables 
y = dataset.iloc[:, 1].values  

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Regression to the Training Set  
# learning correlations between salary and exp
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Training data X_train = Years of exp and Y_train = salary 
regressor.fit(X_train,y_train)

# Predicting the Test set results 
# y_pred will contain predication values given by X test = Experience 
y_pred = regressor.predict(X_test)

# Visualising the the Training Data
plt.scatter(X_train,y_train,color = 'r') # real values data fed
plt.plot(X_train,regressor.predict(X_train),color = 'b') # predicted values
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the the Testing Data
plt.scatter(X_test,y_test,color = 'r') # real values data fed
plt.plot(X_train,regressor.predict(X_train),color = 'b') # predicted values
plt.title('Salary vs Experience(Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

