import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers


class InstanceNormalization(kl.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x-mean) * inv
    return self.scale * normalized + self.offset

  def get_config(self):
    config = {
        'epsilon': self.epsilon,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def dcgan_mnist(image_shape: list, noise_dim: int):

  def make_generator_model():
    model = k.Sequential([
        kl.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)),
        kl.BatchNormalization(),
        kl.LeakyReLU(),
        kl.Reshape((7, 7, 256)),
        kl.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
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

  def make_discriminator_model():
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

  generator = make_generator_model()
  discriminator = make_discriminator_model()

  return generator, discriminator, None


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = k.Sequential([
      kl.Conv2D(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False)
  ])

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(kl.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(kl.LeakyReLU())

  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = k.Sequential([
      kl.Conv2DTranspose(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False)
  ])

  if norm_type.lower() == 'batchnorm':
    result.add(kl.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(kl.Dropout(0.5))

  result.add(kl.ReLU())

  return result


def Generator(output_channels=3, norm_type='batchnorm'):
  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
      downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
      downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
      downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = kl.Conv2DTranspose(
      output_channels,
      4,
      strides=2,
      padding='same',
      kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = kl.Concatenate()

  inputs = kl.Input(shape=[256, 256, 3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return k.Model(inputs=inputs, outputs=x)


def Discriminator(norm_type='batchnorm', target=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = kl.Input(shape=[256, 256, 3], name='input_image')
  x = inp

  if target:
    tar = kl.Input(shape=[256, 256, 3], name='target_image')
    x = kl.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = kl.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = kl.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = kl.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = kl.LeakyReLU()(norm1)

  zero_pad2 = kl.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = kl.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return k.Model(inputs=[inp, tar], outputs=last)
  else:
    return k.Model(inputs=inp, outputs=last)


def pix2pix_facde():
  generator = Generator()
  discriminator = Discriminator()
  return generator, discriminator, None