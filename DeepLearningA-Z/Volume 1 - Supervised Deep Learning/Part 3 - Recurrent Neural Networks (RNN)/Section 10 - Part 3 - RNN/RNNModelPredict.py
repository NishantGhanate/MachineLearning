# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:35:50 2018

@author: Nishant Ghanate
"""

from keras.models import load_model
import numpy as np


# inputs = scaler.fit_transform(inputs)

# 1 observation , 1 time stamp  ,1 feature 
# using unscaled data 
inputs = np.reshape( inputs, (1,1,1))

# load model from single file
model = load_model('StockModel.h5')
output = model.predict(inputs)
output = scaler.inverse_transform(output)
print('from loaded model')
print(output)
