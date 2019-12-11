import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as kl
from models.darknet import compose


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


def retinaface_slim(input_shape: list, num_classes=2) -> k.Model:
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
                  kl.ReLU())(x13)
    x14 = depth_conv2d(256, 3, 2, 'same')(x14)
    x14 = kl.ReLU()(x14)

    loc_layers = [depth_conv2d(3 * 4, 3, padding='same'),
                  depth_conv2d(2 * 4, 3, padding='same'),
                  depth_conv2d(2 * 4, 3, padding='same'),
                  kl.Conv2D(3 * 4, 3, padding='same')]

    conf_layers = [depth_conv2d(3 * num_classes, 3, padding='same'),
                   depth_conv2d(2 * num_classes, 3, padding='same'),
                   depth_conv2d(2 * num_classes, 3, padding='same'),
                   kl.Conv2D(3 * num_classes, 3, padding='same')]

    landm_layers = [depth_conv2d(3 * 10, 3, padding='same'),
                    depth_conv2d(2 * 10, 3, padding='same'),
                    depth_conv2d(2 * 10, 3, padding='same'),
                    kl.Conv2D(3 * 10, 3, padding='same')]
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
