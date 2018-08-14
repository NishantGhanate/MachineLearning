# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

 #importing datasetfile.csv (keep csv and py file in same folder)
dataset = pd.read_csv('Data.csv')

#.iloc means choose coulums : = all columns and -1 = dont take last columns
X = dataset.iloc[:, :-1].values #Enter X in console to see values
Y = dataset.iloc[:, 3].values  #Enter Y in console  yes or no column

#handling missing data in coulmns ctrl+i on Imputer to know more
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0) #function passsing
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#converting text to int labels
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#selecting first col of dataset (country)
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#creating dumby variable creating category 
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#here we are encodinng yes/no coulmn to 0/1 
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.cross_validation import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#Feature scaling (Euclidean distance) model 
from sklearn.cross_validation import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


