import random

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


np.set_printoptions(threshold=np.inf)

def vgg_16(num_class):
        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        print("#######hi")
        return model
if __name__ == '__main__':
    model1 = vgg_16()
    model1.summary()
    # from tensorflow.keras.applications.vgg16 import VGG16
    # model2 = VGG16()
    # model2.summary()

