#im using tensorlow backend which keras select by default

import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential

# keras core layer 
from keras.layers import Dense, Dropout, Activation, Flatten

# keras cnn layers
from keras.layers import Convolution2D, MaxPooling2D
	
from keras.utils import np_utils

# load dataset of images 
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print (X_train.shape)

# from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

# (n, depth, width, height). a full-color image with all 3 RGB channels will have a depth of 3.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# The final preprocessing step for the input data is to convert our data type to float32 and normalize our data values to the range [0, 1].

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# PreProcess class labels 
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Declare linear Sequential model 
model = Sequential()

# CNN input layer 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))

print (model.output_shape)
# (None, 32, 26, 26)

# Dropout layers avoids overfitting 
# MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Connection CNN hidden layers 
# sotmax coverts CNN Layers weights into Probablity 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compling ourmodel 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Trainig our data 
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

# final evaluation 
score = model.evaluate(X_test, Y_test, verbose=0)