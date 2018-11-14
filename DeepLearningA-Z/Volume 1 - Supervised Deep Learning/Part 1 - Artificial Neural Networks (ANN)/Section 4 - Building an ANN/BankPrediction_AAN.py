# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:47:08 2018

@author: Nishant Ghanate 

@website : https://www.superdatascience.com/deep-learning/

# Theano Library : pip install theano
# Tensorflow : pip install tensorflow
# Keras Library : pip install keras
# conda update --all 

@TODO : Make an ANN to predict whether a customer will stay or leave the bank

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Importing Bank dataset Contating rows ----
0.RowNumber,
1.CustomerId,
2.Surname,
3.CreditScore,
4.Geography,
4.Gender,
6.Age,
7.Tenure,
8.Balance,
9.NumOfProducts,
10.HasCrCard,
11.IsActiveMember
12.EstimatedSalary,
13Exited """

dataset = pd.read_csv('Churn_Modelling.csv')

# ---------------@DataPreprocssing removing unwated independent variables------
    
# Selecting coulmns from index 3 to 12 since upper bound is ignored we use +1 index 
X = dataset.iloc[:,3:13].values

# Selecting last index containing binray 1 = yes will bank or 0 = no 
y = dataset.iloc[:,13].values

# OneHotEncoder : https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# LabelEncoder , OneHotEncoder:  http://www.stephacking.com/encode-categorical-data-labelencoder-onehotencoder-python/

from sklearn.preprocessing  import LabelEncoder , OneHotEncoder

# converting country and gender string to label(0 to n int)
labelEncoder_Xcountry = LabelEncoder()
X[:,1] = labelEncoder_Xcountry.fit_transform(X[:,1])

labelEncoder_Xgender = LabelEncoder()
X[:,2] = labelEncoder_Xgender.fit_transform(X[:,2])

# Further remove any value depency from country label we will create dummby var
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Removing dummy varibale Trap by removing index 1 
X = X[:,1:]

# Splitting the Dataset into training and test 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state = 0)

# Feature scaling : https://www.geeksforgeeks.org/python-how-and-where-to-apply-feature-scaling/
from sklearn.preprocessing import StandardScaler 
# Initialise the Scaler 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# -------------------@Building ANN import keras library-----------------------
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initliazing ANN Model 
classifier = Sequential()

""" too look params select Dense and press ctrl + i
    units = Hidden layer = 6 : we have 11 input & 1 output = 12/2 = 6
    inputdim = no.of input / Independent varibales
    kernel_initializer = assigning wieghts to our Neural network close to zero 
    Activation functions : https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
"""
# First Layer must req no of input
classifier.add(Dense(units = 6 ,  kernel_initializer = "uniform", activation = "relu" , input_dim=11)) 

# Adding second hidden layer
classifier.add(Dense(units = 6 ,  kernel_initializer = "uniform", activation = "relu" )) 

# Final layer has only 1 output i.e either custmer will bank yes or no
# if there is multiple outout add  units = no. of outout and activation = "softmax"
classifier.add(Dense(units = 1 ,  kernel_initializer = "uniform", activation = "sigmoid" )) 

""" @Complining model
    Adam : https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
"""  
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])

# Fitting dataset into our model 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100 )

# y_pred = Predticiting whether a customer will leave bank % 
y_pred = classifier.predict(X_test)

# converting % value to True or value threshold = 50%
y_pred = (y_pred > 0.5)

# Making Confusion matrix to validate prediction from testing result 
from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test,y_pred)