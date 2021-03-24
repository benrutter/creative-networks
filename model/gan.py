from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers


def build_generator(
    dimension=8,
    depth=128,
    momentum=0.8,
    kernal=5,
    image_shape=(32, 32, 3),
    latent_dimensions=100,
):
    model = keras.Sequential()

    model.add(layers.Dense(
        dimension * dimension * depth,
        input_dim=latent_dimensions,
        activation='relu'
    ))
    model.add(layers.BatchNormalization(momentum=momentum))
    model.add(layers.Reshape((dimension, dimension, depth)))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(int(depth/2), kernal, padding="same", activation='relu'))
    model.add(layers.BatchNormalization(momentum=momentum))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(int(depth/4), kernal, padding="same", activation='relu'))
    model.add(layers.BatchNormalization(momentum=momentum))

    model.add(layers.Conv2D(int(depth/8), kernal, padding="same", activation='relu'))
    model.add(layers.BatchNormalization(momentum=momentum))

    model.add(layers.Conv2D(3, kernal, padding="same", activation='tanh'))

    return model

def build_discriminator(
    dropout=0.2,
    momentum=0.8,
    kernal=5,
    image_shape=(32, 32, 3),
    latent_dimensions=100,
):

    model = keras.Sequential()

    model.add(layers.Conv2D(32, kernal, strides=2, input_shape=image_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, kernal, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=momentum))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(128, kernal, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=momentum))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(256, kernal, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=momentum))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(0.0002, 0.005),
        metrics=['accuracy']
    )

    model.trainable = False

    return model

generator = build_generator()
discriminator = build_discriminator()
