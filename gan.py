import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers
import matplotlib.pyplot as plt

class ArtDCGAN():
    """
    DCGAN class designed to be train, saved and retrained on art data.
    Contains methods on:
        - Re/Training & Saving model
        - Fetching and processing images
    (At this point, almost everything is TODO)
    """
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.total_epoch = 0

    def build_gan(self):
        adam = optimizers.Adam(0.0002, 0.5)
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'],
        )
        self.generator = self._build_generator()
        noise_input = layers.Input(shape=(self.latent_dim,))
        image = self.generator(noise_input)
        self.discriminator.trainable = False
        valid = self.discriminator(image)
        self.combined = keras.Model(noise_input, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=adam)

    def load_gan(self, generator_weights='artifacts/generator_weights', discriminator_weights='artifacts/discriminator_weights'):
        self.build_gan()
        self.generator.load_weights(generator_weights)
        self.discriminator.load_weights(discriminator_weights)
        noise_input = layers.Input(shape=(self.latent_dim,))
        image = self.generator(noise_input)
        valid = self.discriminator(image)
        self.combined = keras.Model(noise_input, valid)
        adam = optimizers.Adam(0.0002, 0.5)
        self.combined.compile(loss='binary_crossentropy', optimizer=adam)

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

    def _train(self, epochs=1, batch_size=256, model_save_interval=100, image_save_interval=50):

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

            if epoch % model_save_interval == 0:
                self.discriminator.save_weights('artifacts/discriminator-weights')
                self.generator.save_weights('artifacts/generator-weights')

    def save_images(self, epoch):
        """
        currently using matplotlib to save images for a set epoch
        TODO: switch over for PIL to save directly as single images, rather than plots
        """
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        generated_images = self.generator.predict(noise)

        # Rescale images 0 - 1
        generated_images = 0.5 * generated_images + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_images[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("output_images/dcgan_mnist_%d.png" % epoch)
        plt.close()

    def _get_data(self, term, size=100):
        """
        Method to fetch data, for training of model.
        Longterm plan is to use 'term' argument to:
            - search for n images
            - manipulate them into set resolution
        At the moment, this just sets self.x_train to mnist dataset
        """
        (x_train, _), (_, _) = keras.datasets.mnist.load_data()
        # Rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        self.x_train = x_train

    def train_on(self, term, epochs, size=100, batch_size=256, model_save_interval=100, image_save_interval=50):
        self._get_data(term, size)
        self._train(epochs, batch_size, model_save_interval, image_save_interval)

test_gan = ArtDCGAN()
test_gan.build_gan()
test_gan.train_on('cubism', 1)
test_gan.load_gan()
test_gan.train_on('picasso', 1)
