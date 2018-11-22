# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:09:24 2018

@author: Nishant Ghanate

@Dataset link  https://in.finance.yahoo.com/quote/RELIANCE.NS/history?period1=1384885800&period2=1542652200&interval=1d&filter=history&frequency=1d

@TODO // Predict stock prices 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

""" -------------------------- @Data Precossing Step ------------------------"""

dataframe = pd.read_csv('Reliance.csv')

# Splitting Data set 80% train & 20 % test 
training_set, test_set = train_test_split(dataframe, test_size=0.2 ,shuffle = False)

# selecting first index i.ie  Open price 
training_set = training_set.iloc[:,1:2].values

# Feature scaling normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)

# X_train = today T0 ,Y_train = Tomrrow T1 
# : n - 1  
X_train = training_set[0:978]
# Shift by T + 1
y_train = training_set[1:979]

# Reshaping into 3 dimenson array size  = X_train size after spilt ,  timestamp = T+1 , feature = 1 i.e Stock price
X_train = np.reshape(X_train,(978 , 1, 1))

""" -------------------------- @Building Regression Model ---------------------"""

from keras.models import Sequential
from keras.layers import Dense , LSTM

# Creating regression model since stock prices relatively close values
regressor = Sequential()
regressor.add(LSTM(units = 4 , activation = 'sigmoid' , input_shape = [None,1])) # None = anytimestamp , 1 feature 

# Adding outputLayer since we have 1 , outcome T + 1
regressor.add(Dense(units = 1))

# Comiling model using adam since rmsprop & adam giving almost same result we use adam 
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,batch_size=32,epochs=200)

""" -------------------------- @Predicting new values ---------------------"""

real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = scaler.fit_transform(inputs)

# 246 observations
inputs = np.reshape(inputs , (246,1,1))

# Predication 
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# Visualation
plt.plot(real_stock_price , color ='red' , label =' Real stock price')
plt.plot(predicted_stock_price , color ='blue' , label =' Predicted stock price')
plt.title('Reliance Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

""" -------------------------- @Evaluating model ---------------------"""
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt( mean_squared_error(real_stock_price , predicted_stock_price ))

# divide rmse by median value of real stock price to get % value of error