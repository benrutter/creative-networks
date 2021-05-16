import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class ArtDCGAN(keras.Model):
    """
    DCGAN class designed to be train, saved and retrained on image data.
    Initialisation takes:
        size (int): translates to size of generated images
    """
    def __init__(self, size=5, latent_dim=128, batch_size=32, train_directory='train_image'):
        super(ARTDCGAN, self).__init__()
        self.size_factor = size
        self.image_width = self.size_factor * 32
        self.image_height = self.size_factor * 32
        self.channels = 3
        self.image_shape = (self.image_height, self.image_width, self.channels)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self._get_data(train_directory)

    def build_gan(self):
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(0.0001),
        )
        self.generator = self._build_generator()
        noise_input = layers.Input(shape=(self.latent_dim,))
        image = self.generator(noise_input)
        self.discriminator.trainable = False
        valid = self.discriminator(image)
        self.combined = keras.Model(noise_input, valid)
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(0.0001),
        )

    def load_gan(self):
        self.generator = keras.models.load_model("artifacts/generator")
        self.discriminator = keras.models.load_model("artifacts/discriminator")
        noise_input = layers.Input(shape=(self.latent_dim,))
        image = self.generator(noise_input)
        self.discriminator.trainable = False
        valid = self.discriminator(image)
        self.combined = keras.Model(noise_input, valid)
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(0.0001),
        )

    def _build_generator(self):
        """
        Internal method to build generator of GAN
        (void fucntion, takes no arguments)
        """
        generator = keras.Sequential()

        generator.add(keras.Input(shape=(latent_dim,)))
        size_for_dense = 128*(self.size_factor*self.size_factor*8*8)
        generator.add(layers.Dense(size_for_dense))
        generator.add(layers.Reshape((int(self.size_factor*8), int(self.size_factor*8), 128)))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(256, kernel_size=5, strides=2, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.UpSampling2D())
        generator.add(layers.Conv2D(512, kernel_size=5, strides=2, padding="same"))
        generator.add(layers.BatchNormalization(momentum=0.8))
        generator.add(layers.LeakyReLU(alpha=0.2))

        generator.add(layers.Conv2D(self.channels, kernel_size=5, padding="same"))
        generator.add(layers.Activation("tanh"))

        print("\nGenerator")
        generator.summary()

        input = layers.Input(shape=(self.latent_dim,))
        image = generator(input)

        return keras.Model(input, image)

    def _build_discriminator(self):
        """
        Internal method to build discriminator of GAN
        (void function, takes no arguments)
        """
        discriminator = keras.Sequential()

        discriminator.add(keras.Input(shape=self.image_shape))
        discriminator.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same"))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Dropout(0.2))

        discriminator.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
        discriminator.add(layers.BatchNormalization(momentum=0.8))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dropout(0.2))
        discriminator.add(layers.Dense(1, activation="sigmoid"))

        image = layers.Input(shape=self.image_shape)
        validity = discriminator(image)

        print("discriminator")
        discriminator.summary()

        return keras.Model(image, validity)

    def train(self, minibatches=100, model_save_interval=100, image_save_interval=1000):

        for minibatch in range(minibatches):

            # new batch of images
            noise = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            train_batch = self.x_train.__next__()

            # changing fraction of labels to prevent one model overpowering the other
            valid_labels = [[np.random.uniform(0.9, 1)] for i in train_batch[1]]
            invalid_labels = [[np.random.uniform(0, 0.1)] for i in gen_imgs]
            valid_labels = np.array(valid_labels)
            invalid_labels = np.array(invalid_labels)

            # discriminator training

            discriminator_metrics_valid = self.discriminator.train_on_batch(train_batch[0], valid_labels)
            discriminator_metrics_fake = self.discriminator.train_on_batch(gen_imgs, invalid_labels)

            # generator training
            generator_metrics = self.combined.train_on_batch(noise, np.ones((self.batch_size, 1)))

            # display progress
            print("\n--------------------------------------------------------")
            print(f"Minibatch {minibatch} (size {self.batch_size})")
            average_discriminator_loss = (discriminator_metrics_valid + discriminator_metrics_fake) / 2
            print(f"Discriminator loss: {average_discriminator_loss}")
            print(f"Generator loss: {generator_metrics}")
            print("----------------------------------------------------------")

            if minibatch % image_save_interval == 0:
                self.save_images(minibatch)

            if minibatch % model_save_interval == 0:
                self.save_model()

    def save_model(self):
        self.discriminator.save("artifacts/discriminator")
        self.generator.save("artifacts/generator")

    def save_images(self, epoch, images=5):
        """
        Generates and saves images into the "output_images" folder
        TODO: switch over for PIL to save directly as single images, rather than plots
        """
        noise = np.random.uniform(-1, 1, size=(images, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images += 1
        generated_images *= 127.5
        for i in range(images):
            try:
                image = Image.fromarray(generated_images[i, :, :, ].astype(np.uint8))
            except:
                image = Image.fromarray(np.concatenate(generated_images[i, :, :, ], axis=-1).astype(np.uint8))
            image.convert("RGB").save(f"output_images/art-heist-gan-e{epoch}i{i}.jpg")


    def _get_data(self, directory_name="train_images"):
        """
        Method to fetch data, for training of model.
        """
        normalize = lambda x: (x.astype('float32')-127.5)/127.5
        #image_generator = ImageDataGenerator(rescale=1.0/255)
        image_generator = ImageDataGenerator(preprocessing_function=normalize)
        image_iterator = image_generator.flow_from_directory(
            directory_name,
            target_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            color_mode="rgb",
        )
        self.x_train = image_iterator
