
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential

def alexnet():
    model = keras.models.Sequential([
        # layer 1
        layers.Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation=keras.activations.relu,
            padding='valid',
            input_shape=(227, 227, 3)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        # layer 2
        layers.Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation=keras.activations.relu,
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        # layer 3
        layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=keras.activations.relu,
            padding='same'
        ),
        layers.BatchNormalization(),

        # layer 4
        layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=keras.activations.relu,
            padding='same'
        ),
        layers.BatchNormalization(),

        # layer 5
        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=keras.activations.relu,
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        # layer 6
        layers.Flatten(),
        layers.Dense(units=4096, activation=keras.activations.relu),
        layers.Dropout(rate=0.5),

        # layer 7
        layers.Dense(units=4096, activation=keras.activations.relu),
        layers.Dropout(rate=0.5),

        # layer 8
        layers.Dense(units=1000, activation=keras.activations.softmax)
    ])
    return model;