# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:18:51 2018

@author: Nishant Ghanate
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree

#No of Dogs
greyhounds= 100
labs = 100

#grehounds hieght 61 in centimeter +/- 10 cm height difference
grey_height =  61 + 10 * np.random.randn(greyhounds) 

lab_height = 71 + 10 * np.random.randn(labs) 

plt.hist([grey_height , lab_height ] , stacked = True , color = ['red' , 'blue' ])

# Repetition creatre 100 Zeros 
label_labs = [0] * 100
labe_greyhounds  = [1] * 100



#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(features,lables)
#
#
#print(clf.predict( ) ) 