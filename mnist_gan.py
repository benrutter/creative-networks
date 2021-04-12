# built from example code in:
# Deep Learning with Tensorflow 2 and Keras
# Antonio Gulli, Amita Kapoor, Suji Pal

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers
import matplotlib.pyplot as plt

# loading and then normalisingg 0:255 color range to -1:1
(x_train, __), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

# reshaping fro flat vector size of input
x_train = x_train.reshape(60000, 784)

# setting key variables
random_dim = 100
g_losses = []
d_losses = []
adam = optimizers.Adam(lr=0.0002, beta_1=0.5)

# function to plot loss for each batch
def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='Discriminative Loss')
    plt.plot(g_losses, label='Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../output_images/gan_loss_epoch_{epoch}.png')

# create display of generates images
def save_generated_images(epoch, examples=100, dim=(10, 10), figsize=(10,10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'../output_images/gan_generated_images_epoch_{epoch}.png')

# building generator
generator = keras.Sequential()
generator.add(layers.Dense(256, input_dim=random_dim))
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Dense(512))
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Dense(1024))
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Dense(784, activation='tanh'))

# building discriminator
discriminator = keras.Sequential()
discriminator.add(layers.Dense(
    1024,
    input_dim=784,
    kernel_initializer=initializers.RandomNormal(stddev=0.02)
))
discriminator.add(layers.LeakyReLU(0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Dense(512))
discriminator.add(layers.LeakyReLU(0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Dense(256))
discriminator.add(layers.LeakyReLU(0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Dense(1, activation='sigmoid'))

# building combined GAN model
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
discriminator.trainable = False
gan_input = layers.Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = keras.Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)

def train(epochs=1, batch_size=128):
    batch_count = int(x_train.shape[0]/batch_size)
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Batches per epoch: {batch_count}')

    for e in range(1, epochs+1):
        print(f'------Epoch {e}------')
        for _ in range(batch_count):
            # get a random set of unput noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # generate fake MNIST images
            generated_images = generator.predict(noise)
            x = np.concatenate([image_batch, generated_images])

            # labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # one-sides label smoothing
            y_dis[:batch_size] = 0.9

            # train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(x, y_dis)

            # train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        # store loss of most recent batch from this epoch
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if e == 1 or e % 20 == 0:
            save_generated_images(e)

    # finally plot losses
    plot_loss(e)

# actually train the GAN
train(200, 128)
