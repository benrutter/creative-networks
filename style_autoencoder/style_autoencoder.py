import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = data_gen.flow_from_directory(
    directory= 'train_images',
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=32,
    shuffle=True,
    seed=42,
    class_mode='input',
)

latent_dim = 64

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      keras.Input(shape=(32, 32, 3)),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(300, activation='sigmoid'),
      layers.Reshape(target_shape=(32, 32, 3))
    ])

keras.Input(shape=(128, 128, 3)),
layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding="same", activation="relu"),
layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="same", activation="relu"),
layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
layers.Conv2D(512, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
layers.Flatten(),

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(
    train_generator,
    epochs=50,
    shuffle=True,
    validation_data=(train_generator),
)



def save_pics(n):
    images = train_generator.__next__()[0]
    encoded_imgs = autoencoder.encoder(images).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    for i in range(n):
        img = tf.keras.preprocessing.image.array_to_img(decoded_imgs[i])
        img.save(f"cool-{i}.png")


save_pics(5)
