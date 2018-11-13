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

# @DataPreprocssing removing unwated independent variables
    
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

