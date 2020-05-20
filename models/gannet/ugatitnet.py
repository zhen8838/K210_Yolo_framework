import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from models.darknet import compose
from models.gannet.common import ReflectionPadding2D, InstanceNormalization, Conv2DSpectralNormal, DenseSpectralNormal


class ResnetGenerator(object):

  def __init__(self, ngf=64, img_size=256, light=False):
    super(ResnetGenerator, self).__init__()
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
    gap_logit = self.gap_fc(gap)
    gap_weight = self.gap_fc.kernel
    gap = x * tf.squeeze(gap_weight, [-1])

    gmp = kl.GlobalMaxPool2D()(x)
    gmp_logit = self.gmp_fc(gmp)
    gmp_weight = self.gmp_fc.kernel
    gmp = x * tf.squeeze(gmp_weight, [-1])

    cam_logit = kl.Concatenate(-1)([gap_logit, gmp_logit])
    x = kl.Concatenate(-1)([gap, gmp])
    x = self.relu(self.conv1x1(x))

    heatmap = K.sum(x, axis=-1, keepdims=True)

    if self.light:
      x_ = k.layers.GlobalAvgPool2D()(x)
      style_features = self.FC(x_)
    else:
      style_features = self.FC(x)
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
    super(ConvBlock, self).__init__()
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
    super(HourGlass, self).__init__()
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
    super(HourGlassBlock, self).__init__()

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
    super(ResnetBlock, self).__init__()
    conv_block = []
    conv_block += [
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
        kl.LeakyReLU()
    ]

    conv_block += [
        ReflectionPadding2D((1, 1)),
        kl.Conv2D(dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
    ]

    self.conv_block = compose(*conv_block)

  def __call__(self, x):
    out = kl.Add()([x, self.conv_block(x)])
    return out


class ResnetSoftAdaLINBlock(object):

  def __init__(self, dim, use_bias=False):
    super(ResnetSoftAdaLINBlock, self).__init__()
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
    out = self.norm1(out, content_features, style_features)
    out = self.relu1(out)

    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2(out, content_features, style_features)
    return kl.Add()([out, x])


class ResnetAdaLINBlock(object):

  def __init__(self, dim, use_bias=False):
    super(ResnetAdaLINBlock, self).__init__()
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


class SoftAdaLIN(object):

  def __init__(self, num_features, eps=1e-5):
    super(SoftAdaLIN, self).__init__()
    self.num_features = num_features
    self.norm = adaLIN(num_features, eps)

    self.w_gamma = tf.Variable(name='w_gamma',
                               dtype=tf.float32,
                               trainable=True,
                               initial_value=tf.zeros((num_features)))
    self.w_beta = tf.Variable(name='w_beta',
                              dtype=tf.float32,
                              trainable=True,
                              initial_value=tf.zeros((num_features)))

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

  def __call__(self, x, content_features, style_features):
    content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(
        content_features)
    style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(
        style_features)
    # print(style_gamma,self.w_gamma)
    # w_gamma, w_beta = self.w_gamma.expand(x.shape[0],
    #                                       -1), self.w_beta.expand(x.shape[0], -1)
    soft_gamma = (1. - self.w_gamma) * style_gamma + self.w_gamma * content_gamma
    soft_beta = (1. - self.w_beta) * style_beta + self.w_beta * content_beta

    out = self.norm(x, soft_gamma, soft_beta)
    return out


class adaLIN(object):

  def __init__(self, num_features, eps=1e-5):
    super(adaLIN, self).__init__()
    self.eps = eps
    self.rho = tf.Variable(name='rho',
                           dtype=tf.float32,
                           trainable=True,
                           initial_value=tf.ones((num_features)) * 0.9)

  def __call__(self, inputs, gamma, beta):
    in_mean, in_var = K.mean(inputs, axis=[1, 2], keepdims=True), K.var(
        inputs, axis=[1, 2], keepdims=True)
    out_in = (inputs - in_mean) / K.sqrt(in_var + self.eps)
    ln_mean, ln_var = K.mean(inputs, axis=[1, 2, 3], keepdims=True), K.var(
        inputs, axis=[1, 2, 3], keepdims=True)

    out_ln = (inputs - ln_mean) / K.sqrt(ln_var + self.eps)
    out = self.rho * out_in + (1 - self.rho) * out_ln

    out = out * gamma + beta

    return out


class LIN(object):

  def __init__(self, num_features, eps=1e-5):
    super(LIN, self).__init__()
    self.eps = eps
    self.rho = tf.Variable(name='rho',
                           dtype=tf.float32,
                           trainable=True,
                           initial_value=tf.zeros((num_features)))
    self.gamma = tf.Variable(name='gamma',
                             dtype=tf.float32,
                             trainable=True,
                             initial_value=tf.ones((num_features)))
    self.beta = tf.Variable(name='beta',
                            dtype=tf.float32,
                            trainable=True,
                            initial_value=tf.zeros((num_features)))

  def __call__(self, inputs):
    in_mean, in_var = K.mean(inputs, axis=[1, 2], keepdims=True), K.var(
        inputs, axis=[1, 2], keepdims=True)
    out_in = (inputs - in_mean) / K.sqrt(in_var + self.eps)
    ln_mean, ln_var = K.mean(inputs, axis=[1, 2, 3], keepdims=True), K.var(
        inputs, axis=[1, 2, 3], keepdims=True)
    out_ln = (inputs - ln_mean) / K.sqrt(ln_var + self.eps)
    # out = tf.tile(self.rho, (inputs.shape[0], 1, 1, 1)) * out_in + \
    #     (1 - tf.tile(self.rho, (inputs.shape[0], 1, 1, 1))) * out_ln
    # out = out * tf.tile(self.gamma, (inputs.shape[0], 1, 1, 1)) + \
    #     tf.tile(self.beta, (inputs.shape[0], 1, 1, 1))
    out = self.rho * out_in + (1 - self.rho) * out_ln
    out = out * self.gamma + self.beta

    return out


class Discriminator(object):

  def __init__(self, input_nc, ndf=64, n_layers=5):
    super(Discriminator, self).__init__()
    model = [
        ReflectionPadding2D((1, 1)),
        Conv2DSpectralNormal(ndf, kernel_size=4, strides=2,
                             padding='valid', use_bias=True),
        kl.LeakyReLU(0.2)
    ]

    for i in range(1, n_layers - 2):
      mult = 2**(i - 1)
      model += [
          ReflectionPadding2D((1, 1)),
          Conv2DSpectralNormal(ndf * mult * 2, kernel_size=4, strides=2,
                               padding='valid', use_bias=True),
          kl.LeakyReLU(0.2)
      ]

    mult = 2**(n_layers - 2 - 1)
    model += [
        ReflectionPadding2D((1, 1)),
        Conv2DSpectralNormal(ndf * mult * 2, kernel_size=4, strides=1,
                             padding='valid', use_bias=True),
        kl.LeakyReLU(0.2)
    ]

    # Class Activation Map
    mult = 2**(n_layers - 2)
    self.gap_fc = DenseSpectralNormal(1, use_bias=False)
    self.gmp_fc = DenseSpectralNormal(1, use_bias=False)
    self.conv1x1 = kl.Conv2D(ndf * mult, kernel_size=1, strides=1, use_bias=True)
    self.leaky_relu = kl.LeakyReLU(0.2)

    self.pad = ReflectionPadding2D((1, 1))
    self.conv = Conv2DSpectralNormal(1, kernel_size=4, strides=1, padding='valid', use_bias=False)

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


class RhoClipper(object):
  def __init__(self, min, max):
    self.clip_min = min
    self.clip_max = max
    assert min < max

  def __call__(self, module):
    if hasattr(module, 'rho'):
      w = module.rho.data
      w = w.clamp(self.clip_min, self.clip_max)
      module.rho.data = w


class WClipper(object):
  def __init__(self, min, max):
    self.clip_min = min
    self.clip_max = max
    assert min < max

  def __call__(self, module):
    if hasattr(module, 'w_gamma'):
      w = module.w_gamma.data
      w = w.clamp(self.clip_min, self.clip_max)
      module.w_gamma.data = w

    if hasattr(module, 'w_beta'):
      w = module.w_beta.data
      w = w.clamp(self.clip_min, self.clip_max)
      module.w_beta.data = w
