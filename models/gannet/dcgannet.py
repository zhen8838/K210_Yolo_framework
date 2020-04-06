import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from models.darknet import compose


def make_generator_model(noise_dim):
  model = k.Sequential([
      kl.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)),
      kl.BatchNormalization(),
      kl.LeakyReLU(),
      kl.Reshape((7, 7, 256)),
      kl.Conv2DTranspose(128, (5, 5), strides=1, padding='same', use_bias=False),
      kl.BatchNormalization(),
      kl.LeakyReLU(),
      kl.Conv2DTranspose(
          64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
      kl.BatchNormalization(),
      kl.LeakyReLU(),
      kl.Conv2DTranspose(
          1, (5, 5),
          strides=(2, 2),
          padding='same',
          use_bias=False,
          activation='tanh')
  ])
  return model


def make_discriminator_model(image_shape):
  model = k.Sequential([
      kl.Conv2D(
          64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape),
      kl.LeakyReLU(),
      kl.Dropout(0.3),
      kl.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
      kl.LeakyReLU(),
      kl.Dropout(0.3),
      kl.Flatten(),
      kl.Dense(1)
  ])

  return model


def dcgan_mnist(image_shape: list, noise_dim: int):

  generator = make_generator_model(noise_dim)
  discriminator = make_discriminator_model(image_shape)

  return generator, discriminator, None