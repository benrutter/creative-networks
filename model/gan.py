import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from PIL import Image
from random import randint


def build_generator(
    dimension=8,
    depth=128,
    momentum=0.8,
    kernal=5,
    image_shape=(32, 32, 3),
    latent_dimensions=100,
):
    model = Sequential()

    model.add(Dense(
        dimension * dimension * depth,
        input_dim=latent_dimensions,
        activation='relu'
    ))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Reshape((dimension, dimension, depth)))

    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/2), kernal, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), kernal, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(Conv2D(int(depth/8), kernal, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(Conv2D(3, kernal, padding="same", activation='tanh'))

    input = Input(shape=(latent_dimensions,))
    generated_image = model(input)

    return Model(input, generated_image)

def build_discriminator(
    dropout=0.2,
    momentum=0.8,
    kernal=5,
    image_shape=(32, 32, 3),
    latent_dimensions=100,
):

    model = Sequential()

    model.add(Conv2D(32, kernal, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, kernal, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernal, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernal, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(0.0002, 0.005),
        metrics=['accuracy']
    )

    model.trainable = False

    return model

generator = build_generator()
discriminator = build_discriminator()
#network = Model(generator, discriminator)
