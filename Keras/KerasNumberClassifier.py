import keras

# load dataset of images 
from keras.datasets import mnist
from keras.models import Sequential
# keras core layer 
from keras.layers import Dense, Dropout, Flatten
# keras cnn layers
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# Load pre-shuffled MNIST data into train and test sets
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

# (n, depth, width, height). a full-color image with all 3 RGB channels will have a depth of 3.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# The final preprocessing step for the input data is to convert our data type to float32 and normalize our data values to the range [0, 1].
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# PreProcess class labels 
# Convert 1-dimensional class arrays to 10-dimensional class matrices
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Declare linear Sequential model 
model = Sequential()
# CNN input layer 
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Dropout layers avoids overfitting 
# MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Connection CNN hidden layers 
# sotmax coverts CNN Layers weights into Probablity 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# compling ourmodel 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Trainig our data 
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
