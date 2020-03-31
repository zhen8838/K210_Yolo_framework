import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers


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


def pix2pix_facde():

  def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = k.Sequential()
    result.add(
        kl.Conv2D(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False))

    if apply_batchnorm:
      result.add(kl.BatchNormalization())

    result.add(kl.LeakyReLU())

    return result

  def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = k.Sequential()
    result.add(
        kl.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False))

    result.add(kl.BatchNormalization())

    if apply_dropout:
      result.add(kl.Dropout(0.5))

    result.add(kl.ReLU())

    return result

  def Generator():
    inputs = kl.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = kl.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

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
      x = kl.Concatenate()([x, skip])

    x = last(x)

    return k.Model(inputs=inputs, outputs=x)

  def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = kl.Input(shape=[256, 256, 3], name='input_image')
    tar = kl.Input(shape=[256, 256, 3], name='target_image')

    x = kl.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = kl.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = kl.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = kl.BatchNormalization()(conv)

    leaky_relu = kl.LeakyReLU()(batchnorm1)

    zero_pad2 = kl.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = kl.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return k.Model(inputs=[inp, tar], outputs=last)

  generator = Generator()
  discriminator = Discriminator()
  return generator, discriminator, None