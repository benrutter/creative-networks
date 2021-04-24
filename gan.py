import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from random import uniform

class ArtDCGAN():
    """
    DCGAN class designed to be train, saved and retrained on image data.
    Initialisation takes:
        size (int): translates to size of generated images
                    in order to fit with requirements of model
                    is of the form where (size * 4)^2 = area
    """
    def __init__(self, size=40):
        self.area_sqrt_by_four = size
        self.image_width = self.area_sqrt_by_four * 4
        self.image_height = self.area_sqrt_by_four * 4
        self.channels = 3
        self.image_shape = (self.image_height, self.image_width, self.channels)
        self.latent_dim = 100
        self.total_epoch = 0
        self._get_data()

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

    def load_gan(self):
        self.build_gan()
        self.generator.load_weights('artifacts/generator_weights')
        self.discriminator.load_weights('artifacts/discriminator_weights')
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

        generator.add(layers.Dense(256 * self.area_sqrt_by_four * self.area_sqrt_by_four, activation="relu", input_dim=self.latent_dim))
        generator.add(layers.Reshape((self.area_sqrt_by_four, self.area_sqrt_by_four, 256)))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(256, kernel_size=3, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.Activation("relu"))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(256, kernel_size=3, padding="same"))
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

        discriminator.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
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

        discriminator.add(layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.25))

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1, activation='sigmoid'))

        image = layers.Input(shape=self.image_shape)
        validity = discriminator(image)

        return keras.Model(image, validity)

    def train(self, epochs=1, batch_size=256, model_save_interval=100, image_save_interval=50):

        for epoch in range(epochs):

            # generate new batch of images
            noise = np.random.uniform(-1, 1, size=(batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # discriminator training
            train_batch = self.x_train.__next__()
            d_loss_real = self.discriminator.train_on_batch(train_batch[0], train_batch[1])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # generator training
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            self.total_epoch += 1

            if epoch % image_save_interval == 0:
                self.save_images(self.total_epoch)

            if epoch % model_save_interval == 0:
                self.save_weights()

    def save_weights(self):
        self.discriminator.save_weights('artifacts/discriminator-weights')
        self.generator.save_weights('artifacts/generator-weights')

    def save_images(self, epoch, images=5):
        """
        Generates and saves images into the 'output_images' folder
        TODO: switch over for PIL to save directly as single images, rather than plots
        """
        noise = np.random.normal(0, 1, (images, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images += 1
        generated_images *= 0.5
        generated_images *= 255
        for i in range(images):
            image = Image.fromarray(generated_images[i,:,:,0])
            image.convert('RGB').save(f'output_images/art-heist-gan-e{epoch}i{i}.jpg')

    def _get_data(self, size=100):
        """
        Method to fetch data, for training of model.
        """
        image_generator = ImageDataGenerator(rescale=1.0/255)
        image_iterator = image_generator.flow_from_directory(
            'train_images',
            target_size=(self.image_height, self.image_width),
            batch_size=256,
        )
        self.x_train = image_iterator

    def train_on(self, term, epochs, size=100, batch_size=256, model_save_interval=100, image_save_interval=50):
        self._get_data(term, size)
        self._train(epochs, batch_size, model_save_interval, image_save_interval)

test_gan = ArtDCGAN(5)
test_gan.build_gan()
test_gan.train(15, image_save_interval=5)

print('cool')
