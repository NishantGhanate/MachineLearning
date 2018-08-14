# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:44:10 2018

@author: Nishant Ghanate
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # matirx form  job level 
y = dataset.iloc[:, 2].values   # vector form salary 

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y) 

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # default poly deg value = 2 
X_poly = poly_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly,y)

# Visual the Linear Regression results
#plt.scatter(X,y,color='r')
#plt.plot(X,linear_reg.predict(X),color='b')
#plt.title('Aint no  bluff  (Linear Regression)')
#plt.xlabel('JOB postion level')
#plt.ylabel('Salary')

# Visual the Polynomial Regression results
plt.scatter(X,y,color='r') # actaul data 
plt.plot(X,linear_reg2.predict(poly_reg.fit_transform(X)),color='b') # prediction 
plt.title('Aint no  bluff  (Linear Regression)')
plt.xlabel('Job postion level')
plt.ylabel('Salary')
plt.legend(['Prediction ', 'Actaul Data'], loc='upper left')

#predicting salary from Linear Regression
linear_reg.predict(6.5) # 6.5 = job level , select the line Press F9

#predicting salary from Polynomial Regression
linear_reg2.predict(poly_reg.fit_transform(6.5)) # 6.5 = job level ,select the line Press F9

