'''
Viva Questions:-

MNIST - 
MNIST stands for Modified National Institute of Standards and Technology database.
It is a large database of handwritten digits commonly used for training various image processing systems, particularly in the field of machine learning and computer vision. 
The database contains 60,000 training images and 10,000 testing images, each of which is a grayscale image of size 28x28 pixels, representing handwritten digits from 0 to 9. 
MNIST is widely used as a benchmark dataset for developing and testing machine learning algorithms, especially for image classification tasks.

Compiler Warnings

1) set TF_ENABLE_ONEDNN_OPTS=0

Tensorflow- Open source machine learning library by google's brain team
keras - Open Source machine learning library integrated in pytorch, tensorflow

'''

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense



# Load and preprocess the MNIST dataset
# loads dataset of 28x28 pixels of grayscale images of 0 to 9
# x_train and y_train for training data - images
# x_test and y_test for testing - digits of each image label
# divide by 255 for normalization between [0,1]
# We have not divided y_test and y_train becuase they are labels and not images
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



# Define the CNN model
# input_shape = shape of the input 
# num_classes = no. of output

# Sequential = Data structure, linear stack of layers
# 9 layers
# Input is the first layer 

# Conv2D = Convolution Layer with 32 or 64 filters/kernels
# each filter is a size of 3,3, RELU activation function is used

# MaxPooling2D = It is a downsapling operation, reduces feature map
# size (2, 2)

# Flatten = Flattens the input
# Converts multi-dimensional input to 1D array

# Dense - fully connected layer with 64 neurons

# last line gives output using softmax function


def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create the CNN model
input_shape = (28, 28, 1)       # (height, width, channels) 28x28 pixels
num_classes = 10                # 10 classes 0-9
model = create_cnn_model(input_shape, num_classes)  # calls function

# Compile the model
model.compile(optimizer='adam',                         # adam - optimization algorithm - Adaptive Movement Estimation
              loss='sparse_categorical_crossentropy',   # loss function - calculates cross entropy loss
              metrics=['accuracy'])                     # accuracy of algoritm - higher means better

# Train the model
# 4 parameters
# reshapes the x_train data
# y_train - labels
# train 5 times
# provides validation data
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

# Evaluate the model
# test_loss - loss  values - incorrect classified samples
# test_acc - accuracy - correct classified samples


test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)

# print results
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)


