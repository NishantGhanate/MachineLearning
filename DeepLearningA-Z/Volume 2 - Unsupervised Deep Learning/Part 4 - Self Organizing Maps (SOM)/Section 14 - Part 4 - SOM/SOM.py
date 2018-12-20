# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:21:49 2018

@author: Nishant Ghanate
"""

import numpy as np
import pandas as pd
import matplotlib as plt

# Importing data set http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
dataset = pd.read_csv('Credit_Card_Applications.csv')

# select all except last
X = dataset.iloc[:, :-1].values

# select only last col customer approved
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
# we want our feature values between 0 and 1
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)


# keep minisom.py as  same working DIR
from minisom import MiniSom
som = MiniSom(x = 10 , y = 10 , input_len = 15 , sigma = 1.0 , learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X , num_iteration = 100)

# Visualizing results
from pylab import bone , pcolor , colorbar ,plot,show
bone()
# transpose matrix
pcolor(som.distance_map().T)
colorbar()
markers = ['o' , 's']
colors = ['r' , 'g']

for i , x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5 , 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2
         )
show()

# Finding the frauds
mappings = som.win_map(X)
# plots on SOM 
frauds = np.concatenate( (mappings[5,4] , mappings[5,5]) , axis = 0)
# This will return list of customer by rescaling 
frauds = sc.inverse_transform(frauds)

