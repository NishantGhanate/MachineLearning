from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Conv2D, MaxPooling2D , Input, Dense, Dropout, Activation, Flatten # keras cnn layers req
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

# print(X_train.shape) #(50000, 32, 32, 3)

# for i in range(0, 9):
# 	plt.subplot(330 + 1 + i)
# 	plt.imshow(X_train[i])
# # show the plot
# plt.show()

# (n, depth, width, height). a full-color image with all 3 RGB channels will have a depth of 3.
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = x_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = x_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = x_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print('x_train shape:', x_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10 
# num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
# num_classes = np.unique(y_train).shape[0] # there are 10 image classes

