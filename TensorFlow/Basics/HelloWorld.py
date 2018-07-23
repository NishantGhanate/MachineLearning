# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:40:49 2018

@author: Nishant Ghanate
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
