# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 10:04:23 2018

@author: Nishant Ghanate
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

scores = [3.0, 1.0, 0.2]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x) , axis = 0)


print(softmax(scores))


#([start, ]stop, [step, ]dtype=None)
x = np.arange(-2.0, 6.0, 0.1)

#print(x)

scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)

blue_patch = mpatches.Patch(color='blue', label='value of x')
orange_patch = mpatches.Patch(color='orange', label='1.0')

plt.legend(handles=[blue_patch,orange_patch])

plt.ylabel('Soft max ')
plt.xlabel(' x ')

plt.show()
