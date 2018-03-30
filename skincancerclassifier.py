# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 03:28:13 2018

@author: Somanshu Singla
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = Sequential()

#adding convolution layer
model.add(Convolution2D(32,(3,3),input_shape=(64 ,64, 3),activation='relu'))

#adding maxpooling2d
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting cnn to images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'CANCER/train',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory(
                                            'CANCER/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

model.fit_generator(
        training_set,
        steps_per_epoch=15,
        epochs=10,
        validation_data=test_set,
        validation_steps=400)

model.save_weights('skincancerclassifier_weights.h5')

#testing single image
test_img = image.load_img('E:\deeplearningproject\envs\cnn\CANCER\singlepredict\data\ISIC_0000002.jpg',target_size=(64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis = 0)
result = model.predict(test_img)
training_set.class_indices

#image data - isic website
