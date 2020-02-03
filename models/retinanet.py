import tensorflow as tf
from models.darknet import compose, DarknetConv2D
from typing import List, Callable, Tuple
k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


def Conv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis),
        kl.LeakyReLU(alpha=0.1))


def Conv2D_BN_Relu(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and ReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis),
        kl.ReLU())


def Conv2D_BN(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis))


def SSH(inputs: tf.Tensor, filters: int, depth: int = 3) -> tf.Tensor:
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if depth == 3:
        conv3X3 = Conv2D_BN(filters // 2, 3, 1, padding='same')(inputs)
        conv5X5_1 = Conv2D_BN_Leaky(filters // 4, 3, 1, padding='same')(inputs)
        conv5X5 = Conv2D_BN(filters // 4, 3, 1, padding='same')(conv5X5_1)
        conv7X7_2 = Conv2D_BN_Leaky(filters // 4, 3, 1, padding='same')(conv5X5_1)
        conv7X7 = Conv2D_BN(filters // 4, 3, 1, padding='same')(conv7X7_2)
        out = kl.Concatenate(channel_axis)([conv3X3, conv5X5, conv7X7])
        out = kl.ReLU()(out)
    elif depth == 2:
        conv3X3 = Conv2D_BN(filters // 2, 3, 1, padding='same')(inputs)
        conv5X5_1 = Conv2D_BN_Leaky(filters // 4, 3, 1, padding='same')(inputs)
        conv5X5 = Conv2D_BN(filters // 4, 3, 1, padding='same')(conv5X5_1)
        out = kl.Concatenate(channel_axis)([conv3X3, conv5X5])
        out = kl.ReLU()(out)
    else:
        raise ValueError(f'Depth must == [2,3]')
    return out


def FPN(inputs: List[tf.Tensor], filters: int) -> List[tf.Tensor]:
    # conv 1*1
    out1 = Conv2D_BN_Leaky(filters, 1, 1)(inputs[0])
    out2 = Conv2D_BN_Leaky(filters, 1, 1)(inputs[1])
    out3 = Conv2D_BN_Leaky(filters, 1, 1)(inputs[2])

    up3 = kl.UpSampling2D()(out3)
    out2 = kl.Add()([out2, up3])
    # conv 3*3
    out2 = Conv2D_BN_Leaky(filters, 3, 1, padding='same')(out2)

    up2 = kl.UpSampling2D()(out2)
    out1 = kl.Add()([out1, up2])
    out1 = Conv2D_BN_Leaky(filters, 3, 1, padding='same')(out1)
    return [out1, out2, out3]


def retinafacenet(input_shape: list, anchor: List[List],
                  filters: int, alpha=0.25) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    model: k.Model = k.applications.MobileNet(
        input_tensor=inputs, input_shape=tuple(input_shape),
        include_top=False, weights='imagenet', alpha=alpha)

    stage1 = model.get_layer('conv_pw_5_relu').output
    stage2 = model.get_layer('conv_pw_11_relu').output
    stage3 = model.get_layer('conv_pw_13_relu').output
    # FPN
    fpn = FPN([stage1, stage2, stage3], filters)
    """ ssh """
    features = [SSH(fpn[0], filters), SSH(fpn[1], filters), SSH(fpn[2], filters)]
    """ head """

    bbox_out = [kl.Conv2D(len(anchor[i]) * 4, 1, 1)(feat) for (i, feat) in enumerate(features)]  # BboxHead
    class_out = [kl.Conv2D(len(anchor[i]) * 2, 1, 1)(feat) for (i, feat) in enumerate(features)]  # ClassHead
    landm_out = [kl.Conv2D(len(anchor[i]) * 10, 1, 1)(feat) for (i, feat) in enumerate(features)]  # LandmarkHead

    bbox_out = [kl.Reshape((-1, 4))(b) for b in bbox_out]
    landm_out = [kl.Reshape((-1, 10))(b) for b in landm_out]
    class_out = [kl.Reshape((-1, 2))(b) for b in class_out]

    bbox_out = kl.Concatenate(1)(bbox_out)
    landm_out = kl.Concatenate(1)(landm_out)
    class_out = kl.Concatenate(1)(class_out)
    out = kl.Concatenate()([bbox_out, landm_out, class_out])

    infer_model = k.Model(inputs, [bbox_out, landm_out, class_out])
    train_model = k.Model(inputs, out)

    return infer_model, train_model


def conv_bn(filters, strides=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if strides == 2:
        return compose(
            kl.ZeroPadding2D(),
            kl.Conv2D(filters, 3, strides, 'valid', use_bias=False),
            kl.BatchNormalization(channel_axis),
            kl.ReLU()
        )
    else:
        return compose(
            kl.Conv2D(filters, 3, strides, 'same', use_bias=False),
            kl.BatchNormalization(channel_axis),
            kl.ReLU()
        )


def depth_conv2d(filters, kernel_size=1, strides=1, padding='valid'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if kernel_size == 3 and strides == 2 and padding == 'same':
        return compose(
            kl.ZeroPadding2D(),
            kl.DepthwiseConv2D(kernel_size, strides, 'valid'),
            kl.ReLU(),
            kl.Conv2D(filters, 1)
        )
    else:
        return compose(
            kl.DepthwiseConv2D(kernel_size, strides, padding),
            kl.ReLU(),
            kl.Conv2D(filters, 1)
        )


def conv_dw(filters, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if strides == 2:
        return compose(
            kl.ZeroPadding2D(),
            kl.DepthwiseConv2D(3, strides, 'valid', use_bias=False),
            kl.BatchNormalization(),
            kl.ReLU(),

            kl.Conv2D(filters, 1, 1, 'valid', use_bias=False),
            kl.BatchNormalization(),
            kl.ReLU()
        )
    else:
        return compose(
            kl.DepthwiseConv2D(3, strides, 'same', use_bias=False),
            kl.BatchNormalization(),
            kl.ReLU(),

            kl.Conv2D(filters, 1, 1, 'valid', use_bias=False),
            kl.BatchNormalization(),
            kl.ReLU()
        )


def retinaface_slim(input_shape: list, class_num: int, anchor: List[Tuple]) -> k.Model:
    inputs = k.Input(input_shape)

    x1 = conv_bn(16, 2)(inputs)
    x2 = conv_dw(32, 1)(x1)
    x3 = conv_dw(32, 2)(x2)
    x4 = conv_dw(32, 1)(x3)
    x5 = conv_dw(64, 2)(x4)
    x6 = conv_dw(64, 1)(x5)
    x7 = conv_dw(64, 1)(x6)
    x8 = conv_dw(64, 1)(x7)
    x9 = conv_dw(128, 2)(x8)
    x10 = conv_dw(128, 1)(x9)
    x11 = conv_dw(128, 1)(x10)
    x12 = conv_dw(256, 2)(x11)
    x13 = conv_dw(256, 1)(x12)
    x14 = compose(kl.Conv2D(64, 1),
                  kl.ReLU(),
                  depth_conv2d(256, 3, 2, 'same'),
                  kl.ReLU())(x13)

    loc_layers = [depth_conv2d(len(anchor[0]) * 4, 3, padding='same'),
                  depth_conv2d(len(anchor[1]) * 4, 3, padding='same'),
                  depth_conv2d(len(anchor[2]) * 4, 3, padding='same'),
                  kl.Conv2D(len(anchor[3]) * 4, 3, padding='same')]

    conf_layers = [depth_conv2d(len(anchor[0]) * class_num, 3, padding='same'),
                   depth_conv2d(len(anchor[1]) * class_num, 3, padding='same'),
                   depth_conv2d(len(anchor[2]) * class_num, 3, padding='same'),
                   kl.Conv2D(len(anchor[3]) * class_num, 3, padding='same')]

    landm_layers = [depth_conv2d(len(anchor[0]) * 10, 3, padding='same'),
                    depth_conv2d(len(anchor[1]) * 10, 3, padding='same'),
                    depth_conv2d(len(anchor[2]) * 10, 3, padding='same'),
                    kl.Conv2D(len(anchor[3]) * 10, 3, padding='same')]
    detections = [x8, x11, x13, x14]
    loc = []
    conf = []
    landm = []
    for (x, l, lam, c) in zip(detections, loc_layers, landm_layers, conf_layers):
        loc.append(l(x))
        landm.append(lam(x))
        conf.append(c(x))

    bbox_regressions = kl.Concatenate(1)([kl.Reshape((-1, 4))(o) for o in loc])
    ldm_regressions = kl.Concatenate(1)([kl.Reshape((-1, 10))(o) for o in landm])
    classifications = kl.Concatenate(1)([kl.Reshape((-1, 2))(o) for o in conf])
    infer_model = k.Model(inputs, [bbox_regressions, ldm_regressions, classifications])
    train_model = k.Model(inputs, kl.Concatenate(-1)([bbox_regressions, ldm_regressions, classifications]))

    return infer_model, train_model


def ullfd_slim(input_shape: list, class_num: int, anchor: List[Tuple]) -> k.Model:
    inputs = k.Input(input_shape)

    x1 = conv_bn(16, 2)(inputs)
    x2 = conv_dw(32, 1)(x1)
    x3 = conv_dw(32, 2)(x2)
    x4 = conv_dw(32, 1)(x3)
    x5 = conv_dw(64, 2)(x4)
    x6 = conv_dw(64, 1)(x5)
    x7 = conv_dw(64, 1)(x6)
    x8 = conv_dw(64, 1)(x7)
    x9 = conv_dw(128, 2)(x8)
    x10 = conv_dw(128, 1)(x9)
    x11 = conv_dw(128, 1)(x10)
    x12 = conv_dw(256, 2)(x11)
    x13 = conv_dw(256, 1)(x12)
    x14 = compose(kl.Conv2D(64, 1),
                  kl.ReLU(),
                  depth_conv2d(256, 3, 2, 'same'),
                  kl.ReLU())(x13)

    region_layers = [depth_conv2d(len(anchor[0]) * (4 + 1 + class_num), 3, padding='same'),
                     depth_conv2d(len(anchor[1]) * (4 + 1 + class_num), 3, padding='same'),
                     depth_conv2d(len(anchor[2]) * (4 + 1 + class_num), 3, padding='same'),
                     kl.Conv2D(len(anchor[3]) * (4 + 1 + class_num), 3, padding='same')]

    detections = [x8, x11, x13, x14]
    regressions = []
    for (x, region) in zip(detections, region_layers):
        regressions.append(region(x))

    regressions = kl.Concatenate(1)([kl.Reshape((-1, 4 + 1 + class_num))(regress) for regress in regressions])
    infer_model = k.Model(inputs, regressions)
    train_model = infer_model
    return infer_model, train_model


class basicrfb(object):
    def __init__(self, in_channels: int, filters: int, strides=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        self.scale = scale
        self.inter_filters = in_channels // map_reduce
        self.branch0 = compose(self.conv(self.inter_filters, 1, relu=False),
                               self.conv(2 * self.inter_filters, 3, strides, 'same'),
                               self.conv(2 * self.inter_filters, 3, 1, 'same', vision + 1, relu=False))
        self.branch1 = compose(self.conv(self.inter_filters, 1, relu=False),
                               self.conv(2 * self.inter_filters, 3, strides, 'same'),
                               self.conv(2 * self.inter_filters, 3, 1, 'same', vision + 2, relu=False))
        self.branch2 = compose(self.conv(self.inter_filters, 1, relu=False),
                               self.conv((self.inter_filters // 2) * 3, 3, 1, 'same'),
                               self.conv(2 * self.inter_filters, 3, strides, 'same'),
                               self.conv(2 * self.inter_filters, 3, 1, 'same', vision + 4, relu=False))

        self.ConvLinear = self.conv(filters, 1, 1, relu=False)
        self.shortcut = self.conv(filters, 1, strides, relu=False)
        self.relu = kl.ReLU()

    @staticmethod
    def conv(filters, kernel_size, strides=1, padding='valid', dilation=1, relu=True, bn=True) -> Callable:
        if kernel_size == 3 and strides == 2 and padding == 'same':
            l = [
                kl.ZeroPadding2D(),
                kl.Conv2D(filters, kernel_size, strides, 'valid', dilation_rate=dilation, use_bias=False)]
        else:
            l = [kl.Conv2D(filters, kernel_size, strides, padding, dilation_rate=dilation, use_bias=False)]

        if bn:
            l.append(kl.BatchNormalization(epsilon=1e-5, momentum=0.01))
        if relu:
            l.append(kl.ReLU())

        return compose(*l)

    def __call__(self, x: tf.Tensor):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = kl.Concatenate(-1)([x0, x1, x2])
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


def retinaface_rfb(input_shape: list, class_num: int, anchor: List[Tuple]) -> k.Model:
    inputs = k.Input(input_shape)

    x1 = conv_bn(16, 2)(inputs)
    x2 = conv_dw(32, 1)(x1)
    x3 = conv_dw(32, 2)(x2)
    x4 = conv_dw(32, 1)(x3)
    x5 = conv_dw(64, 2)(x4)
    x6 = conv_dw(64, 1)(x5)
    x7 = conv_dw(64, 1)(x6)
    x8 = basicrfb(x7.shape.as_list()[-1], 64, 1, 1.)(x7)
    x9 = conv_dw(128, 2)(x8)
    x10 = conv_dw(128, 1)(x9)
    x11 = conv_dw(128, 1)(x10)
    x12 = conv_dw(256, 2)(x11)
    x13 = conv_dw(256, 1)(x12)

    x14 = compose(kl.Conv2D(64, 1),
                  kl.ReLU(),
                  depth_conv2d(256, 3, 2, 'same'),
                  kl.ReLU())(x13)
    detections = [x8, x11, x13, x14]
    loc_layers = [depth_conv2d(len(anchor[0]) * 4, 3, padding='same'),
                  depth_conv2d(len(anchor[1]) * 4, 3, padding='same'),
                  depth_conv2d(len(anchor[2]) * 4, 3, padding='same'),
                  kl.Conv2D(len(anchor[3]) * 4, 3, padding='same')]

    conf_layers = [depth_conv2d(len(anchor[0]) * class_num, 3, padding='same'),
                   depth_conv2d(len(anchor[1]) * class_num, 3, padding='same'),
                   depth_conv2d(len(anchor[2]) * class_num, 3, padding='same'),
                   kl.Conv2D(len(anchor[3]) * class_num, 3, padding='same')]

    landm_layers = [depth_conv2d(len(anchor[0]) * 10, 3, padding='same'),
                    depth_conv2d(len(anchor[1]) * 10, 3, padding='same'),
                    depth_conv2d(len(anchor[2]) * 10, 3, padding='same'),
                    kl.Conv2D(len(anchor[3]) * 10, 3, padding='same')]
    loc = []
    conf = []
    landm = []
    for (x, l, lam, c) in zip(detections, loc_layers, landm_layers, conf_layers):
        loc.append(l(x))
        landm.append(lam(x))
        conf.append(c(x))

    bbox_regressions = kl.Concatenate(1)([kl.Reshape((-1, 4))(o) for o in loc])
    ldm_regressions = kl.Concatenate(1)([kl.Reshape((-1, 10))(o) for o in landm])
    classifications = kl.Concatenate(1)([kl.Reshape((-1, 2))(o) for o in conf])
    infer_model = k.Model(inputs, [bbox_regressions, ldm_regressions, classifications])
    train_model = k.Model(inputs, kl.Concatenate(-1)([bbox_regressions, ldm_regressions, classifications]))
    return infer_model, train_model


def ullfd_rfb(input_shape: list, class_num: int, anchor: List[Tuple]) -> k.Model:
    inputs = k.Input(input_shape)

    x1 = conv_bn(16, 2)(inputs)
    x2 = conv_dw(32, 1)(x1)
    x3 = conv_dw(32, 2)(x2)
    x4 = conv_dw(32, 1)(x3)
    x5 = conv_dw(64, 2)(x4)
    x6 = conv_dw(64, 1)(x5)
    x7 = conv_dw(64, 1)(x6)
    x8 = basicrfb(x7.shape.as_list()[-1], 64, 1, 1.)(x7)
    x9 = conv_dw(128, 2)(x8)
    x10 = conv_dw(128, 1)(x9)
    x11 = conv_dw(128, 1)(x10)
    x12 = conv_dw(256, 2)(x11)
    x13 = conv_dw(256, 1)(x12)

    x14 = compose(kl.Conv2D(64, 1),
                  kl.ReLU(),
                  depth_conv2d(256, 3, 2, 'same'),
                  kl.ReLU())(x13)
    detections = [x8, x11, x13, x14]

    region_layers = [depth_conv2d(len(anchor[0]) * (4 + 1 + class_num), 3, padding='same'),
                     depth_conv2d(len(anchor[1]) * (4 + 1 + class_num), 3, padding='same'),
                     depth_conv2d(len(anchor[2]) * (4 + 1 + class_num), 3, padding='same'),
                     kl.Conv2D(len(anchor[3]) * (4 + 1 + class_num), 3, padding='same')]
    detections = [x8, x11, x13, x14]
    regressions = []
    for (x, region) in zip(detections, region_layers):
        regressions.append(region(x))

    regressions = kl.Concatenate(1)([kl.Reshape((-1, 4 + 1 + class_num))(regress) for regress in regressions])
    infer_model = k.Model(inputs, regressions)
    train_model = infer_model
    return infer_model, train_model