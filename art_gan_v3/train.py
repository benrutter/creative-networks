import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from art_gan import ArtGAN, SaveImages, SaveGANWeights

art_gan = ArtGAN()
try:
    #art_gan.load()
    pass
except:
    print('W: Could not load GAN, training weights from new')

art_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

art_gan.fit(
    art_gan.dataset,
    epochs=1000,
    callbacks=[
        SaveImages(3, latent_dimensions=art_gan.latent_dimensions, save_path='output_images/'),
        SaveGANWeights(art_gan),
    ],
)
