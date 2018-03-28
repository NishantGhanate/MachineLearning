# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:03:57 2018

@author: Nishant Ghanate
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
# X = indepent variables 
X = dataset.iloc[: , 1:2].values 
# y = depent variables 
y = dataset.iloc[:, 2].values  

from sklearn.preprocessing import StandardScaler
# StandarScaler method convert int to float 
# a warning log would be in log dont worry
sc_X = StandardScaler() 
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR 
# using default kernel for non linear model, SVR does not use Feature scaling class
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predictig the model 
# .transform method takes array datatype as input 
# putting np.arrat[][] inside passes single value array 
# if np.array was [] it would pass a vector
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)


#Visualizing the data
plt.scatter(X,y,color = 'r') # real values data fed
plt.plot(X,regressor.predict(X),color = 'b') # predicted values
plt.title('Salary vs Experience(Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
