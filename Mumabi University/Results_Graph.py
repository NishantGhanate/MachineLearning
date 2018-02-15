# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:08:30 2018

@author: NishantGhanate
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style 
#
df = pd.read_csv("Results_demo.csv")
full = df.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder
label_encoder_subj = LabelEncoder()
#selecting first col of dataset (country)

full[:,0] = label_encoder_subj.fit_transform(full[:,0])

full = sorted(full, key=lambda x: x[0], reverse=False)
chem_passed = []
chem_failed = []

phy_passed = []
phy_failed = []

maths_passed = []
maths_failed = []

 # chem_passed.append(n[2])
for n in full:  
#print (n)
     if n[0] == 0:
         # dipshit rookie coding skills 
          if n[2] > 24:
              chem_passed.append(n[2])
          else :
               chem_failed.append(n[2])
     elif n[0] ==1:
         # print(n[0],n[1],[2])
          if n[2] > 24:
              maths_passed.append(n[2])
          else :
               maths_failed.append(n[2])
     else :
         # print(n[0],n[1],[2])
          if n[2] > 24:
              phy_passed.append(n[2])
          else :
               phy_failed.append(n[2])
               
x = [1,3,5] # x - axis index
y = [len(chem_passed),len(phy_passed),len(maths_passed)] # y - axis data 

plt.bar(x,y,align='center')
x2 = [2,4,6] # x - axis index
y2 = [len(chem_passed),len(phy_passed),len(maths_passed)] # y - axis data 

plt.bar(x2,y2,color='g',align='center')

plt.title('Reval Results Histogram')
plt.xlabel('Subject names')
plt.ylabel('Ratio to pass/fail after reval')
plt.show()

