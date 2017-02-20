#!/usr/bin/python3.5

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
# Convolutional Neural Network for clasifying SEM images
#
# FRANCISCO CARRILLO PÉREZ
# IMAGE ANALYSIS AND COMPUTER VISION
# POLITECNICO DI MILANO (2017)
#
#
#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
import PIL

"""
Different parameters that allow to change variables of the whole network.

@param: BATCH SIZE_TRAIN, is going to depend on the number of samples that we have
@param: IMAGE_HEIGHT, height of the images
@param: IMAGE_WIDTH, width of the images
@param: IMAGE_WIDTH_ORIGINAL, original width of the SEM images
@param: IMAGE_HEIGHT_ORIGINAL, original height of the SEM images
@param: dimension_first_conv, how many dimensions do you want in the first convolutional layer of the network
@param: dimension_second_conv, how many dimensions do you want in the second convolutional layer of the network
@param: dimension_fc, how many neurons do you want in the fully conected layer

"""

BATCH_SIZE_TRAIN = 1000
NUM_EPOCHS = 200
IMAGE_HEIGHT =29
IMAGE_WIDTH = 29
IMAGE_WIDTH_ORIGINAL = 1024
IMAGE_HEIGHT_ORIGINAL = 696
dimension_first_conv = 16
dimension_second_conv = 32
dimension_fc = 64

'''
Full path of the different directories for loading the dataset to the network, and also values of the dataset.

@param: train_data_dir, full path where the training samples are
@param: validation_data_dir, full path where the validation samples are
@param: test_data_dir, full path where the validation samples are
@param: nb_train_samples, number of train samples
@param: nb_validation_samples, number of validation samples
@param: nb_test_samples, number of test samples

'''
train_data_dir = './DataSet/keras_dataset/train'
validation_data_dir = './DataSet/keras_dataset/validation'
test_data_dir = './DataSet/keras_dataset/test'
nb_train_samples = 4807 + 5307
nb_validation_samples = 9 + 9
nb_test_samples = 9 + 9

'''Options for performing training, restore a model or test'''
restore = False
train = True
test = True

'''Assign the value of the prediction to the whole patch of an image'''
def assign_to_matrix(patch_prediction, matrix,x,y,patch_height,patch_width):

    '''This is becuase the program is taking anomalous as 0 and normal as 1, and when
       we want to paint it should be the other way around'''
    color = 0
    if(patch_prediction == 0):
        color = 1 #anomalous
    elif(patch_prediction == 1):
        color = 0 #normal
    for i in range(x,x+patch_width):
        for j in range(y,y+patch_height):
            matrix.itemset((i,j),color)


    return matrix

'''
Creation of the architecture for the CNN.

- One convolutional layer
- Activation ReLU.
- Max Pooling (2,2)
- One convolutional layer
- Activation ReLU
- Max Pooling (2,2)
- One convolutional layer
- Activation Relu
- Max Pooling (2,2)
- Fully conected layer
- Dropout layer
- Output layer
- Activation sigmoid

'''
print("Creating the model")

print("creating first layer")
model = Sequential()
model.add(Convolution2D(dimension_first_conv, 5, 5, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

print("creating second layer")
model.add(Convolution2D(dimension_second_conv, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

print("creating third layer")
model.add(Convolution2D(dimension_fc, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

print("creating output layer")
model.add(Flatten())
model.add(Dense(dimension_fc))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

'''
For visualizing the model

@param: mode, the model we have declared above
@param: to_file, name of the file is going to be saved to

'''
from keras.utils.visualize_util import plot
print("Plotting the model")
plot(model, to_file='model.png')

print("augmentation configuration for training")
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
print("augmentation configuration for testing")
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=18,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=18,
        class_mode='binary')

'''
Restoring the wights, the name could be changed depending the name of out file, but they are saved as weights.h5
'''
if(restore == True):
    model.load_weights('weights.h5')

if(train == True):
    #Using the early stopping technique to prevent overfitting
    earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    print("Fitting the model")
    history = model.fit_generator(
            train_generator,
            samples_per_epoch=BATCH_SIZE_TRAIN,
            callbacks=[earlyStopping],
            nb_epoch=NUM_EPOCHS,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)

    print("Saving the weights")
    model.save_weights('weights.h5')  # always save your weights after training or during training

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if(test == True):
    test_loss = model.evaluate_generator(test_generator,val_samples = nb_test_samples)
    print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))

'''
Here the full images are going to be predicted

@param: image_patch, where the original iamges for testing are, with their gt images
@param: image_name, name of the original image
@param: image_name_gt, name of the gt image corresponding to the original image

'''

image_path = "./DataSet/NormalImages/Test/"
image_name = "ITIA1131.jpg"
image_name_gt = "ITIA1131_gt.png"

#data = misc.imread(image_path+image_name)
data = PIL.Image.open(image_path+image_name).convert('RGB')
data_gt = PIL.Image.open(image_path+image_name_gt).convert('RGB')
data = numpy.array(data)
data_gt = numpy.array(data_gt)

#This is the matrix for defining the image of predictions for each pixel.
E_image = numpy.zeros((696,1024))


for x1 in range(0,696,10):
    for y1 in range(0,1024,10):
        #if i'm out of boundaries don't do anything
        if(x1 + 29 > 696 or y1 + 29 > 1024):
            whylord = 0 #Just need to do something
        else:
            #otherwhise
            #Take the patch from the image
            rect = numpy.copy(data[x1:x1+IMAGE_HEIGHT,y1:y1+IMAGE_HEIGHT])
            #convert it to numpy array
            rect = numpy.array(rect)
            #expand first dimension to represent batch size
            rect = numpy.expand_dims(rect,axis=0)
            #resize it
            rect.astype(float)
            #get the prediction for the patch
            patch_prediction = model.predict(rect,verbose=0)
            #assign this prediction to the prediction image's matrix
            E_image = assign_to_matrix(patch_prediction[0], E_image,x1,y1,IMAGE_HEIGHT,IMAGE_WIDTH)


print("Plotting the image...")
fig = plt.figure()
imgplot = plt.imshow(E_image,cmap='gray')
fig.canvas.set_window_title("Prediction "+image_name)
plt.show()
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(E_image, cmap='gray')
a.set_title("Prediction")
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(data_gt)
a.set_title("Original "+image_name)
plt.show()
