# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:37:04 2018

@author: Nishant Ghanate
"""

import numpy as np 
import pandas as pd
from sklearn.cross_validation import train_test_split 
from sklearn import tree
dataset = pd.read_csv('iris_training.csv')

""".iloc means choose coulums : = all columns and -1 = dont take last columns
 Featurres data set length , width  """
 
X_train = dataset.iloc[20:, :-1].values 
X_test = dataset.iloc[-20:, :-1].values 


"""Labels Variable set 
 Select last column index = 4  where  0 = Setosa , 1 = veriscolor , 2 = virginica """
Y_train = dataset.iloc[20:, 4].values
Y_test =  dataset.iloc[-20:, 4].values

clf =   tree.DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)

# X = Features of test set , y = label 
print(X_test[0] , Y_test[0] )

print( clf.predict( [ X_test[0] ])  )  
