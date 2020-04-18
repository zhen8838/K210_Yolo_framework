import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from models.darknet import compose
from models.gannet.common import Conv2DSpectralNormal, InstanceNormalization


def Conv2D(filters, kernel_size=3, strides=1, padding='valid', use_bias=False):
  f = []
  if kernel_size == 3 or kernel_size == (3, 3):
    f.append(kl.ZeroPadding2D())
  f.append(
      kl.Conv2D(
          filters,
          kernel_size,
          strides=strides,
          padding=padding,
          use_bias=use_bias))
  return compose(*f)


def Conv2DNormLReLU(filters,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    use_bias=False):
  return compose(
      Conv2D(filters, kernel_size, strides, padding, use_bias),
      InstanceNormalization(), kl.LeakyReLU(0.2))


def dwiseConv2D(kernel_size=3,
                strides=1,
                padding='valid',
                depth_multiplier=1,
                use_bias=False):
  f = []
  if kernel_size == 3 or kernel_size == (3, 3):
    f.append(kl.ZeroPadding2D())
  f.append(
      kl.DepthwiseConv2D(
          kernel_size, strides, padding, depth_multiplier, use_bias=use_bias))
  return compose(*f)


def SeparableConv2D(filters,
                    kernel_size=3,
                    strides=1,
                    padding='valid',
                    use_bias=True):
  f = []
  if (kernel_size == 3 or kernel_size == (3, 3)) and (strides == 1 or
                                                      strides == (1, 1)):
    f.append(kl.ZeroPadding2D())
  if (strides == 2 or strides == (2, 2)):
    f.append(kl.ZeroPadding2D())
  f.extend([
      kl.SeparableConv2D(
          filters, kernel_size, strides, padding, use_bias=use_bias),
      InstanceNormalization(),
      kl.LeakyReLU(0.2)
  ])
  return compose(*f)


def Conv2DTransposeLReLU(filters,
                         kernel_size=2,
                         strides=2,
                         padding='same',
                         use_bias=False):
  return compose(
      kl.Conv2DTranspose(
          filters, kernel_size, strides, padding, use_bias=use_bias),
      InstanceNormalization(), kl.LeakyReLU(0.2))


def Unsample(filters, kernel_size=3):
  """
      An alternative to transposed convolution where we first resize, then convolve.
      See http://distill.pub/2016/deconv-checkerboard/
      For some reason the shape needs to be statically known for gradient propagation
      through tf.image.resize_images, but we only know that for fixed image size, so we
      plumb through a "training" argument
  """
  return compose(
      kl.UpSampling2D(interpolation='bilinear'),
      SeparableConv2D(filters, kernel_size))


def Downsample(filters=256, kernel_size=3):
  return compose(
      kl.Lambda(lambda x: tf.image.resize(x, (tf.shape(x)[1] // 2, tf.shape(x)[2]
                                              // 2))),
      SeparableConv2D(filters, kernel_size))


def InvertedRes_block(inputs, alpha, filters, strides, use_bias=False):

  # pw
  bottleneck_filters = round(alpha * inputs.get_shape().as_list()[-1])
  x = Conv2DNormLReLU(
      bottleneck_filters, kernel_size=1, use_bias=use_bias)(
          inputs)

  # dw
  x = compose(dwiseConv2D(), InstanceNormalization(), kl.LeakyReLU(0.2))(x)

  # pw & linear
  x = compose(Conv2D(filters, kernel_size=1), InstanceNormalization())(x)

  # element wise add, only for strides==1
  if (int(inputs.get_shape().as_list()[-1]) == filters) and (strides == 1 or
                                                             strides == (1, 1)):
    x = inputs + x

  return x


def generator(input_shape):
  inputs = k.Input(input_shape)
  # b1
  x = compose(Conv2DNormLReLU(64), Conv2DNormLReLU(64))(inputs)
  x = kl.Add()([SeparableConv2D(128, strides=2)(x), Downsample(128)(x)])
  # b2
  x = compose(Conv2DNormLReLU(128), SeparableConv2D(128))(x)
  x = kl.Add()([SeparableConv2D(256, strides=2)(x), Downsample(256)(x)])

  # m
  x = Conv2DNormLReLU(256)(x)
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r1
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r2
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r3
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r4
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r5
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r6
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r7
  x = InvertedRes_block(x, alpha=2, filters=256, strides=1)  # r8

  # u2
  x = compose(Unsample(128), SeparableConv2D(128), Conv2DNormLReLU(128))(x)
  # u1
  x = compose(Unsample(128), Conv2DNormLReLU(64), Conv2DNormLReLU(64))(x)
  # out
  x = compose(
      Conv2D(filters=3, kernel_size=1, strides=1),
      kl.Activation(tf.nn.tanh, dtype=tf.float32))(
          x)
  return k.Model(inputs, x)


def Conv2DSN(filters,
             kernel_size,
             strides,
             padding='valid',
             use_bias=True,
             use_sn=False):
  f = []
  if use_sn:
    f.append(
        Conv2DSpectralNormal(
            filters, kernel_size, strides, padding=padding, use_bias=use_bias))
  else:
    f.append(
        kl.Conv2D(
            filters, kernel_size, strides, padding=padding, use_bias=use_bias))

  return compose(*f)


def discriminator(input_shape: list, filters: int, nblocks: int,
                  use_sn: bool) -> k.Model:
  """
  
  Args:
      input_shape (list): 
      filters (int): filters
      nblocks (int): blocks numbers
      use_sn (bool): weather use SpectralNormalization
  
  Returns:
      k.Model: discriminator
  """
  inputs = k.Input(input_shape)
  inner_filters = filters // 2
  f = [
      Conv2DSN(
          inner_filters,
          kernel_size=3,
          strides=1,
          padding='same',
          use_bias=False,
          use_sn=use_sn),
      kl.LeakyReLU(0.2)
  ]

  for i in range(1, nblocks):
    f.extend([
        Conv2DSN(
            inner_filters * 2,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            use_sn=use_sn),
        kl.LeakyReLU(0.2),
        # kl.Dropout(0.2),
        Conv2DSN(
            inner_filters * 4,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            use_sn=use_sn),
        InstanceNormalization(),
        kl.LeakyReLU(0.2)
    ])

    inner_filters *= 2
  f.extend([
      Conv2DSN(
          inner_filters * 2,
          kernel_size=3,
          strides=1,
          padding='same',
          use_bias=False,
          use_sn=use_sn),
      InstanceNormalization(),
      kl.LeakyReLU(0.2),
      # kl.Dropout(0.2),
      Conv2DSN(
          1,
          kernel_size=3,
          strides=1,
          padding='same',
          use_bias=False,
          use_sn=use_sn),
      kl.Activation('linear', dtype=tf.float32)
  ])

  x = compose(*f)(inputs)
  return k.Model(inputs, x)


def animenet(image_shape: list,
             filters: int = 64,
             nblocks: int = 3,
             use_sn: bool = True):

  generator_model = generator(image_shape)
  discriminator_model = discriminator(image_shape, filters, nblocks, use_sn)

  return generator_model, discriminator_model, None