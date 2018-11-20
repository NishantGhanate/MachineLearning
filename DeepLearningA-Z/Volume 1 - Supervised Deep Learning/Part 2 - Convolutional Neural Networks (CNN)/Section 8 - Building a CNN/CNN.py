# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:05:28 2018

@Learnt from : https://www.superdatascience.com/deep-learning/

@author: Nishant Ghanate

@TODO Build CNN to classfy images of cat and dog
 
"""

# https://keras.io/layers/convolutional/

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

"""----------------------- @Building CNN ----------------------------- """

# Declaring Sequential network
classifier = Sequential()

# Step 1 - Cnn Input layer
# 32 Feature Detector of size 3x3
# Dimension of 2d array = 64 
classifier.add(Conv2D( 32,  (3,3) , input_shape=(64,64,3) , activation="relu" ) )

# Step 2 - MaxPooling
# Reducing the image dimension by 2 
classifier.add(MaxPooling2D(pool_size=(2,2) ))

# Adding another layer 
classifier.add(Conv2D( 32,  (3,3)  , activation="relu" ) )
classifier.add(MaxPooling2D(pool_size=(2,2) ))

# Step 3 - Flattning array 
classifier.add(Flatten())

# Step 4 - Hidden layer size 
classifier.add(Dense(output_dim = 128  , activation = "relu"))

# outputLayer with 1 output 
classifier.add(Dense(output_dim = 1  , activation = "sigmoid"))

# Compile the CNN latyer 
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"]) 

"""----------------------- @ImagePreprocessing ----------------------------- """

# https://keras.io/preprocessing/image/#image-preprocessing

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Traing image = 8000 , Train image = 200  
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=800)

"""----------------------- @Prediction ----------------------------- """

import numpy as np
from keras.preprocessing import image

test_image = image.load_image('dataset/single_prediction/cat_or_dog_1.jpg' , target_size = (64 , 64 ))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis = 0)

result = classifier.predict(test_image)

# Gives label assigned to cateogry i.e Cat = 0 & Dog = 1
training_set.class_indices