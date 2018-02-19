# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 20:56:35 2018

@author: Nishant
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #independ variables 
y = dataset.iloc[:, 4].values #depent variables i.e profits

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) #Creating dumby variable
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummby variable Trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
#plotting graph of prediction values
"""x = np.arange(10)
plt.plot(x,y_test)
plt.plot(x,y_pred)
plt.legend(['y_test', 'y_pred'], loc='upper left')
plt.ylabel('Profits')
plt.show() """

# Building the optimal model using Backward Elimination
# 50 lines 1 col
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis =1)
X_opt = X[:, [0,1,2,3,4,5] ] # col index 
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()   # check p value and remove higher p value 

X_opt = X[:, [0,3] ] # col index 2,4,5 removed it had highest p value ,index 3 = R&D col
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()   # check p value and remove higher p value 
