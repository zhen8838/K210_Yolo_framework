import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from models.darknet import compose
from models.gannet.common import (ReflectionPadding2D,
                                  InstanceNormalization,
                                  SpectralNormalization,
                                  ConstraintMinMax)
from typing import List, Tuple


class ResnetGenerator(object):

  def __init__(self, ngf=64, img_size=256, light=False):
    self.light = light

    self.ConvBlock1 = compose(
        ReflectionPadding2D((3, 3)),
        kl.Conv2D(ngf, kernel_size=7, strides=1, padding='valid', use_bias=False),
        InstanceNormalization(),
        kl.LeakyReLU())

    self.HourGlass1 = HourGlass(ngf, ngf)
    self.HourGlass2 = HourGlass(ngf, ngf)

    # Down-Sampling
    self.DownBlock1 = compose(
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(ngf * 2, kernel_size=3, strides=2, padding='valid', use_bias=False),
        InstanceNormalization(),
        kl.LeakyReLU())

    self.DownBlock2 = compose(
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(ngf * 4, kernel_size=3, strides=2, padding='valid', use_bias=False),
        InstanceNormalization(),
        kl.LeakyReLU())

    # Encoder Bottleneck
    self.EncodeBlock1 = ResnetBlock(ngf * 4)
    self.EncodeBlock2 = ResnetBlock(ngf * 4)
    self.EncodeBlock3 = ResnetBlock(ngf * 4)
    self.EncodeBlock4 = ResnetBlock(ngf * 4)

    # Class Activation Map
    self.gap_fc = kl.Dense(1)
    self.gmp_fc = kl.Dense(1)
    self.conv1x1 = kl.Conv2D(ngf * 4, kernel_size=1, strides=1)
    self.relu = kl.LeakyReLU()

    # Gamma, Beta block
    if self.light:
      self.FC = compose(
          kl.Dense(ngf * 4),
          kl.LeakyReLU(),
          kl.Dense(ngf * 4),
          kl.LeakyReLU())
    else:
      self.FC = compose(
          kl.Dense(ngf * 4),
          kl.LeakyReLU(),
          kl.Dense(ngf * 4),
          kl.LeakyReLU())

    # Decoder Bottleneck
    self.DecodeBlock1 = ResnetSoftAdaLINBlock(ngf * 4)
    self.DecodeBlock2 = ResnetSoftAdaLINBlock(ngf * 4)
    self.DecodeBlock3 = ResnetSoftAdaLINBlock(ngf * 4)
    self.DecodeBlock4 = ResnetSoftAdaLINBlock(ngf * 4)

    # Up-Sampling
    self.UpBlock1 = compose(
        kl.UpSampling2D((2, 2)),
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(ngf * 2, kernel_size=3, strides=1, padding='valid', use_bias=False),
        LIN(ngf * 2),
        kl.LeakyReLU())

    self.UpBlock2 = compose(
        kl.UpSampling2D((2, 2)),
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(ngf, kernel_size=3, strides=1, padding='valid', use_bias=False),
        LIN(ngf),
        kl.LeakyReLU())

    self.HourGlass3 = HourGlass(ngf, ngf)
    self.HourGlass4 = HourGlass(ngf, ngf, False)

    self.ConvBlock2 = compose(
        ReflectionPadding2D((3, 3)),
        kl.Conv2D(3, kernel_size=7, strides=1, padding='valid', use_bias=False),
        kl.Activation('tanh'))

  def __call__(self, x):
    x = self.ConvBlock1(x)
    x = self.HourGlass1(x)
    x = self.HourGlass2(x)

    x = self.DownBlock1(x)
    x = self.DownBlock2(x)

    x = self.EncodeBlock1(x)
    content_features1 = kl.GlobalAvgPool2D()(x)
    x = self.EncodeBlock2(x)
    content_features2 = kl.GlobalAvgPool2D()(x)
    x = self.EncodeBlock3(x)
    content_features3 = kl.GlobalAvgPool2D()(x)
    x = self.EncodeBlock4(x)
    content_features4 = kl.GlobalAvgPool2D()(x)

    gap = kl.GlobalAvgPool2D()(x)
    gap_logit = self.gap_fc(gap)  # [N,1]
    gap_weight = self.gap_fc.kernel  # [128,1]
    gap = x * tf.squeeze(gap_weight, [-1])  # [N,h,w,128]

    gmp = kl.GlobalMaxPool2D()(x)
    gmp_logit = self.gmp_fc(gmp)  # [N,1]
    gmp_weight = self.gmp_fc.kernel  # [128,1]
    gmp = x * tf.squeeze(gmp_weight, [-1])  # [N,h,w,128]

    cam_logit = kl.Concatenate(-1)([gap_logit, gmp_logit])  # [N,2]
    x = kl.Concatenate(-1)([gap, gmp])  # [N,h,w,256]
    x = self.relu(self.conv1x1(x))  # [N,h,w,128]

    heatmap = K.sum(x, axis=-1, keepdims=True)  # [N,h,w,1]

    if self.light:
      x_ = k.layers.GlobalAvgPool2D()(x)
      style_features = self.FC(x_)  # [N,128]
    else:
      x_ = kl.Reshape((-1,))(x)
      style_features = self.FC(x_)
    x = self.DecodeBlock1(x, content_features4, style_features)
    x = self.DecodeBlock2(x, content_features3, style_features)
    x = self.DecodeBlock3(x, content_features2, style_features)
    x = self.DecodeBlock4(x, content_features1, style_features)

    x = self.UpBlock1(x)
    x = self.UpBlock2(x)

    x = self.HourGlass3(x)
    x = self.HourGlass4(x)
    out = self.ConvBlock2(x)

    return out, cam_logit, heatmap


class ConvBlock(object):

  def __init__(self, dim_in, dim_out):
    self.dim_out = dim_out

    self.ConvBlock1 = compose(
        InstanceNormalization(),
        kl.LeakyReLU(),
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(dim_out // 2, kernel_size=3, strides=1, use_bias=False))

    self.ConvBlock2 = compose(
        InstanceNormalization(),
        kl.LeakyReLU(),
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(dim_out // 4, kernel_size=3, strides=1, use_bias=False))

    self.ConvBlock3 = compose(
        InstanceNormalization(),
        kl.LeakyReLU(),
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(dim_out // 4, kernel_size=3, strides=1, use_bias=False))

    self.ConvBlock4 = compose(
        InstanceNormalization(),
        kl.LeakyReLU(),
        kl.Conv2D(dim_out, kernel_size=1, strides=1, use_bias=False))

  def __call__(self, x):
    residual = x

    x1 = self.ConvBlock1(x)
    x2 = self.ConvBlock2(x1)
    x3 = self.ConvBlock3(x2)
    out = kl.Concatenate(-1)([x1, x2, x3])

    if residual.shape[-1] != self.dim_out:
      residual = self.ConvBlock4(residual)

    return kl.Add()([residual, out])


class HourGlass(object):

  def __init__(self, dim_in, dim_out, use_res=True):
    self.use_res = use_res

    self.HG = compose(
        HourGlassBlock(dim_in, dim_out),
        ConvBlock(dim_out, dim_out),
        kl.Conv2D(dim_out, kernel_size=1, strides=1, use_bias=False),
        InstanceNormalization(),
        kl.LeakyReLU())

    self.Conv1 = kl.Conv2D(3, kernel_size=1, strides=1)

    if self.use_res:
      self.Conv2 = kl.Conv2D(dim_out, kernel_size=1, strides=1)
      self.Conv3 = kl.Conv2D(dim_out, kernel_size=1, strides=1)

  def __call__(self, x):
    ll = self.HG(x)
    tmp_out = self.Conv1(ll)

    if self.use_res:
      ll = self.Conv2(ll)
      tmp_out_ = self.Conv3(tmp_out)
      return kl.Add()([x, ll, tmp_out_])

    else:
      return tmp_out


class HourGlassBlock(object):

  def __init__(self, dim_in, dim_out):

    self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
    self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
    self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
    self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
    self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
    self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
    self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
    self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)

    self.ConvBlock5 = ConvBlock(dim_out, dim_out)

    self.ConvBlock6 = ConvBlock(dim_out, dim_out)
    self.ConvBlock7 = ConvBlock(dim_out, dim_out)
    self.ConvBlock8 = ConvBlock(dim_out, dim_out)
    self.ConvBlock9 = ConvBlock(dim_out, dim_out)

  def __call__(self, x):
    skip1 = self.ConvBlock1_1(x)
    down1 = kl.AvgPool2D()(x)
    down1 = self.ConvBlock1_2(down1)

    skip2 = self.ConvBlock1_1(down1)
    down2 = kl.AvgPool2D()(down1)
    down2 = self.ConvBlock1_2(down2)

    skip3 = self.ConvBlock1_1(down2)
    down3 = kl.AvgPool2D()(down2)
    down3 = self.ConvBlock1_2(down3)

    skip4 = self.ConvBlock1_1(down3)
    down4 = kl.AvgPool2D()(down3)
    down4 = self.ConvBlock1_2(down4)

    center = self.ConvBlock5(down4)

    up4 = self.ConvBlock6(center)
    up4 = kl.UpSampling2D((2, 2))(up4)
    up4 = kl.Add()([skip4, up4])

    up3 = self.ConvBlock7(up4)
    up3 = kl.UpSampling2D((2, 2))(up3)
    up3 = kl.Add()([skip3, up3])

    up2 = self.ConvBlock8(up3)
    up2 = kl.UpSampling2D((2, 2))(up2)
    up2 = kl.Add()([skip2, up2])

    up1 = self.ConvBlock9(up2)
    up1 = kl.UpSampling2D((2, 2))(up1)
    up1 = kl.Add()([skip1, up1])

    return up1


class ResnetBlock(object):

  def __init__(self, dim, use_bias=False):
    conv_block = []
    conv_block += [ReflectionPadding2D((1, 1)),
                   kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
                   InstanceNormalization(),
                   kl.LeakyReLU()]

    conv_block += [ReflectionPadding2D((1, 1)),
                   kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
                   InstanceNormalization()]

    self.conv_block = compose(*conv_block)

  def __call__(self, x):
    out = kl.Add()([x, self.conv_block(x)])
    return out


class ResnetSoftAdaLINBlock(object):

  def __init__(self, dim, use_bias=False):
    self.pad1 = ReflectionPadding2D((1, 1))
    self.conv1 = kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
    self.norm1 = SoftAdaLIN(dim)
    self.relu1 = kl.LeakyReLU()

    self.pad2 = ReflectionPadding2D((1, 1))
    self.conv2 = kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
    self.norm2 = SoftAdaLIN(dim)

  def __call__(self, x, content_features, style_features):
    out = self.pad1(x)
    out = self.conv1(out)
    out = self.norm1([out, content_features, style_features])
    out = self.relu1(out)

    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2([out, content_features, style_features])
    return kl.Add()([out, x])


class ResnetAdaLINBlock(object):

  def __init__(self, dim, use_bias=False):
    self.pad1 = ReflectionPadding2D((1, 1))
    self.conv1 = kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
    self.norm1 = adaLIN(dim)
    self.relu1 = kl.LeakyReLU()

    self.pad2 = ReflectionPadding2D((1, 1))
    self.conv2 = kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
    self.norm2 = adaLIN(dim)

  def __call__(self, x, gamma, beta):
    out = self.pad1(x)
    out = self.conv1(out)
    out = self.norm1(out, gamma, beta)
    out = self.relu1(out)
    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2(out, gamma, beta)

    return kl.Add()([out, x])


class SoftAdaLIN(k.layers.Layer):

  def __init__(self, num_features, w_min=0, w_max=1.0, eps=1e-5, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.num_features = num_features
    self.eps = eps
    self.w_min = w_min
    self.w_max = w_max
    self.norm = adaLIN(num_features, self.eps)

    self.w_gamma = self.add_weight(
        name='w_gamma',
        dtype=tf.float32,
        trainable=True,
        shape=[num_features],
        initializer=tf.zeros_initializer())
    self.w_beta = self.add_weight(
        name='w_beta',
        dtype=tf.float32,
        trainable=True,
        shape=[num_features],
        initializer=tf.zeros_initializer())

    self.c_gamma = compose(
        kl.Dense(num_features),
        kl.LeakyReLU(),
        kl.Dense(num_features))
    self.c_beta = compose(
        kl.Dense(num_features),
        kl.LeakyReLU(),
        kl.Dense(num_features))
    self.s_gamma = kl.Dense(num_features)
    self.s_beta = kl.Dense(num_features)

  def call(self, inputs, **kwargs):
    x, content_features, style_features = inputs
    content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(
        content_features)
    style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(
        style_features)

    soft_gamma = (1. - self.w_gamma) * style_gamma + self.w_gamma * content_gamma
    soft_beta = (1. - self.w_beta) * style_beta + self.w_beta * content_beta

    out = self.norm([x, soft_gamma, soft_beta])
    return out

  def get_config(self):
    config = {
        'num_features': self.num_features,
        'eps': self.eps,
        'w_min': self.w_min,
        'w_max': self.w_max,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class adaLIN(k.layers.Layer):

  def __init__(self, num_features, rho_min=0, rho_max=1.0, eps=1e-5,
               trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.eps = eps
    self.num_features = num_features
    self.rho_min = rho_min
    self.rho_max = rho_max
    self.rho = self.add_weight(name='rho',
                               dtype=tf.float32,
                               trainable=True,
                               shape=[1, 1, 1, self.num_features],
                               initializer=k.initializers.Constant(0.9),
                               constraint=ConstraintMinMax(self.rho_min,
                                                           self.rho_max))

  def call(self, inputs, **kwargs):
    inputs, gamma, beta = inputs
    in_mean, in_var = K.mean(inputs, axis=[1, 2], keepdims=True), K.var(
        inputs, axis=[1, 2], keepdims=True)
    out_in = (inputs - in_mean) / K.sqrt(in_var + self.eps)
    ln_mean, ln_var = K.mean(inputs, axis=[1, 2, 3], keepdims=True), K.var(
        inputs, axis=[1, 2, 3], keepdims=True)

    out_ln = (inputs - ln_mean) / K.sqrt(ln_var + self.eps)
    out = self.rho * out_in + (1 - self.rho) * out_ln
    # NOTE expand dims for training batch > 1
    out = out * gamma[:, None, None, :] + beta[:, None, None, :]

    return out

  def get_config(self):
    config = {
        'num_features': self.num_features,
        'eps': self.eps,
        'rho_min': self.rho_min,
        'rho_max': self.rho_max
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class LIN(k.layers.Layer):

  def __init__(self, num_features, rho_min=0, rho_max=1.0, eps=1e-5,
               trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.eps = eps
    self.num_features = num_features
    self.rho_min = rho_min
    self.rho_max = rho_max
    self.rho = self.add_weight(name='rho',
                               dtype=tf.float32,
                               trainable=True,
                               shape=[self.num_features],
                               initializer=k.initializers.Constant(0.),
                               constraint=ConstraintMinMax(self.rho_min,
                                                           self.rho_max))
    self.gamma = self.add_weight(name='gamma',
                                 dtype=tf.float32,
                                 trainable=True,
                                 shape=[self.num_features],
                                 initializer=k.initializers.Ones())
    self.beta = self.add_weight(name='beta',
                                dtype=tf.float32,
                                trainable=True,
                                shape=[self.num_features],
                                initializer=k.initializers.Zeros())

  def call(self, inputs, **kwargs):
    in_mean, in_var = K.mean(inputs, axis=[1, 2], keepdims=True), K.var(
        inputs, axis=[1, 2], keepdims=True)
    out_in = (inputs - in_mean) / K.sqrt(in_var + self.eps)
    ln_mean, ln_var = K.mean(inputs, axis=[1, 2, 3], keepdims=True), K.var(
        inputs, axis=[1, 2, 3], keepdims=True)

    out_ln = (inputs - ln_mean) / K.sqrt(ln_var + self.eps)
    out = self.rho * out_in + (1 - self.rho) * out_ln
    out = out * self.gamma + self.beta

    return out

  def get_config(self):
    config = {
        'num_features': self.num_features,
        'eps': self.eps,
        'rho_min': self.rho_min,
        'rho_max': self.rho_max,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Discriminator(object):

  def __init__(self, input_nc, ndf=64, n_layers=5):
    model = [
        ReflectionPadding2D((1, 1)),
        SpectralNormalization(kl.Conv2D(ndf, kernel_size=4, strides=2,
                                        padding='valid', use_bias=True)),
        kl.LeakyReLU(0.2)
    ]

    for i in range(1, n_layers - 2):
      mult = 2**(i - 1)
      model += [
          ReflectionPadding2D((1, 1)),
          SpectralNormalization(kl.Conv2D(ndf * mult * 2, kernel_size=4, strides=2,
                                          padding='valid', use_bias=True)),
          kl.LeakyReLU(0.2)
      ]

    mult = 2**(n_layers - 2 - 1)
    model += [
        ReflectionPadding2D((1, 1)),
        SpectralNormalization(kl.Conv2D(ndf * mult * 2, kernel_size=4, strides=1,
                                        padding='valid', use_bias=True)),
        kl.LeakyReLU(0.2)
    ]

    # Class Activation Map
    mult = 2**(n_layers - 2)
    self.gap_fc = SpectralNormalization(kl.Dense(1, use_bias=False))
    self.gmp_fc = SpectralNormalization(kl.Dense(1, use_bias=False))
    self.conv1x1 = kl.Conv2D(ndf * mult, kernel_size=1, strides=1, use_bias=True)
    self.leaky_relu = kl.LeakyReLU(0.2)

    self.pad = ReflectionPadding2D((1, 1))
    self.conv = SpectralNormalization(
        kl.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False))

    self.model = compose(*model)

  def __call__(self, input):
    x = self.model(input)

    gap = kl.GlobalAvgPool2D()(x)
    gap_logit = self.gap_fc(gap)
    gap_weight = self.gap_fc.kernel
    gap = x * tf.squeeze(gap_weight, -1)

    gmp = kl.GlobalMaxPool2D()(x)
    gmp_logit = self.gmp_fc(gmp)
    gmp_weight = self.gmp_fc.kernel
    gmp = x * tf.squeeze(gmp_weight, -1)

    cam_logit = k.layers.Concatenate(-1)([gap_logit, gmp_logit])
    x = k.layers.Concatenate(-1)([gap, gmp])
    x = self.leaky_relu(self.conv1x1(x))

    heatmap = K.sum(x, axis=-1, keepdims=True)

    x = self.pad(x)
    out = self.conv(x)

    return out, cam_logit, heatmap


def ugatitnet(image_shape: list, batch_size: int, filters: int = 32,
              discriminator_G_layers: int = 7,
              discriminator_L_layers: int = 5, light: bool = True
              ) -> Tuple[List[k.Model], List[k.Model], k.Model]:
  """ ugatitent

  Args:
      image_shape (list): 
      filters (int, optional): . Defaults to 32.
      discriminator_layers (int, optional): . Defaults to 5.
      light (bool, optional): . Defaults to True.

  Returns:
      [List[k.Model], List[k.Model], k.Model]: generator_model, discriminator_model, val_model
  """
  assert image_shape[0] == image_shape[1], 'image must be square'
  genA2B = ResnetGenerator(ngf=filters, img_size=image_shape[0], light=light)
  genB2A = ResnetGenerator(ngf=filters, img_size=image_shape[0], light=light)
  disGA = Discriminator(3, filters, n_layers=discriminator_G_layers)
  disGB = Discriminator(3, filters, n_layers=discriminator_G_layers)
  disLA = Discriminator(3, filters, n_layers=discriminator_L_layers)
  disLB = Discriminator(3, filters, n_layers=discriminator_L_layers)

  def modeling(body):
    x = k.Input(image_shape, batch_size=batch_size)
    outputs = body(x)
    return k.Model(x, outputs)

  model_genA2B = modeling(genA2B)
  model_genB2A = modeling(genB2A)
  model_disGA = modeling(disGA)
  model_disGB = modeling(disGB)
  model_disLA = modeling(disLA)
  model_disLB = modeling(disLB)
  del genA2B, genB2A, disGA, disGB, disLA, disLB
  return [model_genA2B, model_genB2A], [model_disGA, model_disGB, model_disLA, model_disLB], model_genA2B
