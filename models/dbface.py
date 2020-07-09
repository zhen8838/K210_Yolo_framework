import tensorflow as tf
import numpy as np
from models.darknet import compose
k = tf.keras
kl = tf.keras.layers


class SeModule(object):
  def __init__(self, in_size, reduction=4):
    super().__init__()
    self.pool = kl.GlobalAveragePooling2D()

    self.se = k.Sequential([
        kl.Conv2D(in_size // reduction, kernel_size=1, strides=1,
                  padding='valid', use_bias=False),
        kl.BatchNormalization(),
        kl.ReLU(),
        kl.Conv2D(in_size, kernel_size=1, strides=1,
                  padding='valid', use_bias=False),
        kl.BatchNormalization(),
        kl.Activation('sigmoid')
    ])

  def __call__(self, x):
    tmp = k.backend.expand_dims(k.backend.expand_dims(self.pool(x), 1), 1)
    return kl.Multiply()([x, self.se(tmp)])


class Block(object):
  def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, strides):
    super(Block, self).__init__()
    self.strides = strides
    self.se = semodule

    self.conv1 = kl.Conv2D(expand_size, kernel_size=1, strides=1, padding='valid', use_bias=False)
    self.bn1 = kl.BatchNormalization()
    self.nolinear1 = nolinear
    self.conv2 = kl.DepthwiseConv2D(kernel_size=kernel_size,
                                    strides=strides, padding='same', use_bias=False)
    self.bn2 = kl.BatchNormalization()
    self.nolinear2 = nolinear
    self.conv3 = kl.Conv2D(out_size, kernel_size=1,
                           strides=1, padding='valid',
                           use_bias=False)
    self.bn3 = kl.BatchNormalization()

    self.shortcut = lambda x: x
    if strides == 1 and in_size != out_size:
      self.shortcut = compose(
          kl.Conv2D(out_size, kernel_size=1, strides=1,
                    padding='valid', use_bias=False),
          kl.BatchNormalization(),
      )

  def __call__(self, x):
    out = self.nolinear1(self.bn1(self.conv1(x)))
    out = self.nolinear2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.se != None:
      out = self.se(out)
    if self.strides == 1:
      out = kl.Add()([out, self.shortcut(x)])
    else:
      out = out
    return out


class Mbv3SmallFast(object):
  def __init__(self):
    super(Mbv3SmallFast, self).__init__()

    self.keep = [0, 2, 7]
    self.uplayer_shape = [16, 24, 48]
    self.output_channels = 96

    self.conv1 = kl.Conv2D(16, kernel_size=3, strides=2,
                           padding='same', use_bias=False)
    self.bn1 = kl.BatchNormalization()
    self.hs1 = kl.ReLU()

    self.bneck = [
        Block(3, 16, 16, 16, kl.ReLU(), None, 2),          # 0 *
        Block(3, 16, 72, 24, kl.ReLU(), None, 2),          # 1
        Block(3, 24, 88, 24, kl.ReLU(), None, 1),          # 2 *
        Block(5, 24, 96, 40, kl.ReLU(), SeModule(40), 2),  # 3
        Block(5, 40, 240, 40, kl.ReLU(), SeModule(40), 1),  # 4
        Block(5, 40, 240, 40, kl.ReLU(), SeModule(40), 1),  # 5
        Block(5, 40, 120, 48, kl.ReLU(), SeModule(48), 1),  # 6
        Block(5, 48, 144, 48, kl.ReLU(), SeModule(48), 1),  # 7 *
        Block(5, 48, 288, 96, kl.ReLU(), SeModule(96), 2),  # 8
    ]

  def __call__(self, x) -> list:
    x = self.hs1(self.bn1(self.conv1(x)))

    outs = []
    for index, item in enumerate(self.bneck):
      x = item(x)
      if index in self.keep:
        outs.append(x)

    outs.append(x)
    return outs


# Conv BatchNorm Activation
class CBAModule(object):
  def __init__(self, in_channels, out_channels=24, kernel_size=3, strides=1, padding='valid', use_bias=False):
    super(CBAModule, self).__init__()
    self.conv = kl.Conv2D(out_channels, kernel_size,
                          strides, padding=padding, use_bias=use_bias)
    self.bn = kl.BatchNormalization()
    self.act = kl.ReLU()

  def __call__(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x


# Up Sample Module
class UpModule(object):
  def __init__(self, in_channels, out_channels, kernel_size=2, strides=2, use_bias=False, mode="UCBA"):
    super(UpModule, self).__init__()
    self.mode = mode

    if self.mode == "UCBA":
      # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
      self.up = kl.UpSampling2D(2)
      self.conv = CBAModule(in_channels, out_channels, 3,
                            padding='same', use_bias=use_bias)
    elif self.mode == "DeconvBN":
      self.dconv = kl.Conv2DTranspose(out_channels,
                                      kernel_size, strides, use_bias=use_bias)
      self.bn = kl.BatchNormalization()
    elif self.mode == "DeCBA":
      self.dconv = kl.Conv2DTranspose(out_channels,
                                      kernel_size, strides, use_bias=use_bias)
      self.conv = CBAModule(out_channels, out_channels, 3,
                            padding='same', use_bias=use_bias)
    else:
      raise RuntimeError(f"Unsupport mode: {mode}")

  def __call__(self, x):
    if self.mode == "UCBA":
      return self.conv(self.up(x))
    elif self.mode == "DeconvBN":
      return F.relu(self.bn(self.dconv(x)))
    elif self.mode == "DeCBA":
      return self.conv(self.dconv(x))


# SSH Context Module
class ContextModule(object):
  def __init__(self, in_channels):
    super(ContextModule, self).__init__()

    block_wide = in_channels // 4
    self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding='same')
    self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding='same')
    self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding='same')
    self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding='same')

  def __call__(self, x):
    x = self.inconv(x)
    up = self.upconv(x)
    down = self.downconv(x)
    down = self.downconv2(down)
    return kl.Concatenate(-1)([up, down])


# SSH Detect Module
class DetectModule(object):
  def __init__(self, in_channels):
    super(DetectModule, self).__init__()

    self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding='same')
    self.context = ContextModule(in_channels)

  def __call__(self, x):
    up = self.upconv(x)
    down = self.context(x)
    return kl.Concatenate(-1)([up, down])


# Job Head Module
class HeadModule(object):
  def __init__(self, in_channels, out_channels, name: str,
               has_ext=False, init_std=0.001, init_bias=0):
    super(HeadModule, self).__init__()
    self.head = kl.Conv2D(out_channels, kernel_size=1,
                          kernel_initializer=k.initializers.RandomNormal(stddev=init_std),
                          bias_initializer=k.initializers.constant(init_bias),
                          name=name)
    self.has_ext = has_ext

    if has_ext:
      self.ext = CBAModule(in_channels, kernel_size=3, padding='same', use_bias=False)

  def __call__(self, x):
    if self.has_ext:
      x = self.ext(x)
    return self.head(x)


class DBFaceModel(object):
  def __init__(self, nclass: int = 1, nlandmark: int = 5,
               wide=24, has_ext=False, upmode="UCBA"):
    self.nlandmark = nlandmark
    self.nclass = nclass

    # define backbone
    self.bb = Mbv3SmallFast()

    # Get the number of branch node channels
    # stride4, stride8, stride16
    c0, c1, c2 = self.bb.uplayer_shape

    self.conv3 = CBAModule(self.bb.output_channels, wide, kernel_size=1,
                           strides=1, padding='valid', use_bias=False)  # s32
    self.connect0 = CBAModule(c0, wide, kernel_size=1)  # s4
    self.connect1 = CBAModule(c1, wide, kernel_size=1)  # s8
    self.connect2 = CBAModule(c2, wide, kernel_size=1)  # s16

    self.up0 = UpModule(wide, wide, kernel_size=2,
                        strides='same', mode=upmode)  # s16
    self.up1 = UpModule(wide, wide, kernel_size=2,
                        strides='same', mode=upmode)  # s8
    self.up2 = UpModule(wide, wide, kernel_size=2,
                        strides='same', mode=upmode)  # s4
    self.detect = DetectModule(wide)

    self.center = HeadModule(wide, nclass, 'center',
                             has_ext=has_ext, init_std=0.001,
                             init_bias=-np.log((1 - 0.01) / 0.01))
    self.box = HeadModule(wide, 4, 'box',
                          has_ext=has_ext, init_std=0.001, init_bias=0)

    if self.nlandmark:
      self.landmark = HeadModule(wide, self.nlandmark * 2, 'landmark',
                                 has_ext=has_ext, init_std=0.001, init_bias=0)

  def __call__(self, x):
    s4, s8, s16, s32 = self.bb(x)
    s32 = self.conv3(s32)

    s16 = self.up0(s32) + self.connect2(s16)
    s8 = self.up1(s16) + self.connect1(s8)
    s4 = self.up2(s8) + self.connect0(s4)
    x = self.detect(s4)

    center = self.center(x)
    box = self.box(x)

    if self.nlandmark:
      landmark = self.landmark(x)
      return center, box, landmark

    return center, box


def dbface_k210_v1(input_shape: list,
                   nclass: int = 1, nlandmark: int = 5,
                   wide=64, has_ext=False, upmode="UCBA") -> [k.Model, k.Model]:
  body = DBFaceModel(nclass, nlandmark, wide=wide,
                     has_ext=has_ext, upmode=upmode)
  inputs = k.Input(input_shape, batch_size=None)
  outputs = body(inputs)
  infer_model = k.Model(inputs, outputs)
  return infer_model, infer_model, infer_model
