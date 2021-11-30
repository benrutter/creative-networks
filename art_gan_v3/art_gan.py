import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ArtGAN(keras.Model):
    def __init__(self, latent_dimensions=128, image_directory="train_images"):
        super(ArtGAN, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.dataset = keras.preprocessing.image_dataset_from_directory(
            image_directory,
            label_mode=None,
            image_size=(128, 128),
            batch_size=32
        )
        self.dataset = self.dataset.map(lambda x: x / 255.0)

    def load(self, load_path="artifacts"):
        self.generator = keras.models.load_model(f"{load_path}/generator")
        self.discriminator = keras.models.load_model(f"{load_path}/discriminator")

    def build_generator(self):
        generator = keras.Sequential([
            keras.Input(shape=(self.latent_dimensions,)),
            layers.Dense(8*8*32),
            layers.Reshape((8, 8, 32)),
            layers.Conv2DTranspose(32, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Conv2DTranspose(128, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Conv2DTranspose(512, kernel_size=5, strides=(2, 2), padding="same", activation="relu"),
            layers.Conv2DTranspose(128, kernel_size=5, strides=(1, 1), padding="same", activation="relu"),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ], name="Generator")
        generator.summary()
        return generator

    def build_discriminator(self):
        discriminator = keras.Sequential([
            keras.Input(shape=(128, 128, 3)),
            layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(512, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ], name="Discriminator")
        discriminator.summary()
        return discriminator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ArtGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.g_loss_metric = keras.metrics.Mean(name="generator_loss")

    def load_weights(self, filepath):
        pass

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # random points in latent space into images via generator
        # used for disciminator training
        batch_size = tf.shape(real_images)[0]
        noise_input = tf.random.normal(shape=(batch_size, self.latent_dimensions))
        generated_images = self.generator(noise_input)

        # produce combined training set
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.1 * tf.random.uniform(tf.shape(labels))

        # train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # random points in latent space to turn to images via generator
        # used for generator training
        noise_input = tf.random.normal(shape=(batch_size, self.latent_dimensions))
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        # discriminator weights not updated
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise_input))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

class SaveImages(keras.callbacks.Callback):
    def __init__(self, number_of_images=5, latent_dimensions=128, save_path=""):
        self.number_of_images = number_of_images
        self.latent_dimensions = latent_dimensions
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        noise_input = tf.random.normal(shape=(self.number_of_images, self.latent_dimensions))
        generated_images = self.model.generator(noise_input)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.number_of_images):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(f"{self.save_path}gan-img-{epoch}-{i}.png")

class SaveGANWeights(keras.callbacks.Callback):
    def __init__(self, gan, save_path="artifacts"):
        self.gan = gan
        self.save_path=save_path

    def on_epoch_end(self, epoch, logs=None):
        self.gan.discriminator.save(f"{self.save_path}/discriminator")
        self.gan.generator.save(f"{self.save_path}/generator")
