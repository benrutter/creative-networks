# built in part from example code in:
# Deep Learning with Tensorflow 2 and Keras
# Antonio Gulli, Amita Kapoor, Suji Pal

import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers
import matplotlib.pyplot as plt

class ArtHeistGAN():
    """
    DCGAN class designed to be train, saved and retrained on art data.
    Contains methods on:
        - Re/Training & Saving model
        - Fetching and processing images
    (At this point, almost everything is TODO)
    """
    def __init__(self, rows, cols, channels, z=10):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = z
        self.total_epoch = 0

        adam = optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'],
        )

        # Build the generator
        self.generator = self._build_generator()

        # The generator takes noise as input and generates images
        z = layers.Input(shape=(self.latent_dim,))
        image = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def _build_generator(self):
        """
        Internal method to build generator of GAN
        (void fucntion, takes no arguments)
        """
        generator = keras.Sequential()

        generator.add(layers.Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        generator.add(layers.Reshape((7, 7, 128)))
        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(128, kernel_size=3, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Activation("relu"))
        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(64, kernel_size=3, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Activation("relu"))
        generator.add(layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        generator.add(layers.Activation("tanh"))

        noise = layers.Input(shape=(self.latent_dim,))
        image = generator(noise)

        return keras.Model(noise, image)

    def _build_discriminator(self):
        """
        Internal method to build discriminator of GAN
        (void fucntion, takes no arguments)
        """

        discriminator = keras.Sequential()

        discriminator.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.25))
        discriminator.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.25))
        discriminator.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.25))
        discriminator.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.25))
        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1, activation='sigmoid'))

        image = layers.Input(shape=self.img_shape)
        validity = discriminator(image)

        return keras.Model(image, validity)

    def train(self, epochs=1, batch_size=256, image_save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Select a random half of images
            idx = np.random.randint(0, self.x_train.shape[0], batch_size)
            imgs = self.x_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            self.total_epoch += 1

            # If at save interval => save generated image samples
            if epoch % image_save_interval == 0:
                self.save_images(self.total_epoch)

    def save_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        generated_images = self.generator.predict(noise)

        # Rescale images 0 - 1
        generated_images = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/dcgan_mnist_%d.png" % epoch)
        plt.close()

    def get_data(self, size, term):
        """
        Method to fetch data, for training of model.
        Longterm plan is to use 'term' argument to:
            - search for n images
            - manipulate them into set resolution
        At the moment, this just sets self.x_train to mnist dataset
        """
        (x_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        self.x_train = xtrain

    def save(self):
        """
        Method to save copy of model for retraining later on
        """
        with open(f'art_heist_gan_e{self.total_epoch}.obj', w) as output_file:
            pickle.dump(self, output_file)
