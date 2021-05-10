import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image


class ArtDCGAN():
    """
    DCGAN class designed to be train, saved and retrained on image data.
    Initialisation takes:
        size (int): translates to size of generated images
    """
    def __init__(self, size=5):
        self.size_factor = size
        self.image_width = self.size_factor * 32
        self.image_height = self.size_factor * 32
        self.channels = 3
        self.image_shape = (self.image_height, self.image_width, self.channels)
        self.latent_dim = 128
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
        self.generator = keras.models.load_model('artifacts/generator')
        self.discriminator = keras.models.load_model('artifacts/discriminator')
        noise_input = layers.Input(shape=(self.latent_dim,))
        image = self.generator(noise_input)
        valid = self.discriminator(image)
        self.combined = keras.Model(noise_input, valid)
        adam = optimizers.Adam(0.0002, 0.5)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'],
        )

    def _build_generator(self):
        """
        Internal method to build generator of GAN
        (void fucntion, takes no arguments)
        """
        generator = keras.Sequential()

        generator.add(layers.Dense(
            256*(self.size_factor*self.size_factor),
            input_dim=self.latent_dim,
            activation='relu',
        ))
        generator.add(layers.Reshape((
            int(self.size_factor),
            int(self.size_factor),
            256
        )))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(256, kernel_size=4, padding="same"))
        generator.add(layers.Dropout(0.5))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.LeakyReLU())

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(256, kernel_size=4, padding="same"))
        generator.add(layers.Dropout(0.5))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.LeakyReLU())

        for i in range(3):
            generator.add(layers.UpSampling2D())
            generator.add(layers.Conv2D(256, kernel_size=4, padding="same"))
            generator.add(layers.Dropout(0.2))
            generator.add(layers.BatchNormalization(momentum=0.8))
            generator.add(layers.Activation("relu"))

        generator.add(layers.Conv2D(self.channels, kernel_size=4, padding="same"))
        generator.add(layers.Activation("sigmoid"))

        print('Generator')
        generator.summary()

        noise = layers.Input(shape=(self.latent_dim,))
        image = generator(noise)

        return keras.Model(noise, image)

    def _build_discriminator(self):
        """
        Internal method to build discriminator of GAN
        (void function, takes no arguments)
        """
        discriminator = keras.Sequential()

        discriminator.add(layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            input_shape=self.image_shape,
            padding="same",
        ))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.5))

        discriminator.add(layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
        discriminator.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(layers.Conv2D(256, kernel_size=4, strides=1, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.2))

        discriminator.add(layers.Conv2D(512, kernel_size=4, strides=1, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.2))

        discriminator.add(layers.Conv2D(1024, kernel_size=4, strides=1, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.2))

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1, activation='sigmoid'))

        image = layers.Input(shape=self.image_shape)
        validity = discriminator(image)

        print('discriminator')
        discriminator.summary()

        return keras.Model(image, validity)

    def train(self, epochs=100, batch_size=32, model_save_interval=100, image_save_interval=50):

        for epoch in range(epochs):

            # new batch of images
            noise = np.random.uniform(-1, 1, size=(batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            train_batch = self.x_train.__next__()

            # discriminator training
            discriminator_loss_real = self.discriminator.train_on_batch(train_batch[0], train_batch[1])
            discriminator_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))

            # generator training
            generator_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # display progress
            print(f'Epoch {epoch} - batch size {batch_size}')
            print(f'Discriminator loss (valid): {sum(discriminator_loss_real)/len(discriminator_loss_real)}')
            print(f'Discriminator loss (fake): {sum(discriminator_loss_fake)/len(discriminator_loss_fake)}')
            try:
                print(f'Generator loss: {sum(generator_loss)/len(generator_loss)}')
            except:
                print(f'Generator loss: {generator_loss}')

            if epoch % image_save_interval == 0:
                self.save_images(epoch)

            if epoch % model_save_interval == 0:
                self.save_model()

    def save_model(self):
        self.discriminator.save('artifacts/discriminator')
        self.generator.save('artifacts/generator')

    def save_images(self, epoch, images=5):
        """
        Generates and saves images into the 'output_images' folder
        TODO: switch over for PIL to save directly as single images, rather than plots
        """
        noise = np.random.normal(0, 1, (images, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images *= 255
        for i in range(images):
            image = Image.fromarray(generated_images[i, :, :, ].astype(np.uint8))
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
            color_mode='rgb',
        )
        self.x_train = image_iterator
