import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as kl
from models.darknet import compose


def UltraLightFastGenericFaceBaseNet(inputs: tf.Tensor, base_filters=16) -> k.Model:
    def conv_bn(filters, strides, number):
        channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
        if strides == 2:
            l = [kl.ZeroPadding2D(),
                 kl.Conv2D(filters, 3, strides, 'valid', use_bias=False,
                           kernel_regularizer=k.regularizers.l2(5e-4),
                           name=f'conv_bn_{number}_conv')]
        else:
            l = [kl.Conv2D(filters, 3, strides, 'valid', use_bias=False,
                           kernel_regularizer=k.regularizers.l2(5e-4),
                           name=f'conv_bn_{number}_conv')]
        return l + [kl.BatchNormalization(channel_axis, name=f'conv_bn_{number}_bn'),
                    kl.ReLU(name=f'conv_bn_{number}_relu')]

    def conv_dw(filters, strides, number):
        channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
        if strides == 2:
            l = [kl.ZeroPadding2D(),
                 kl.DepthwiseConv2D(3, strides, padding='valid', use_bias=False,
                                    name=f'conv_dw_{number}_dw'), ]
        else:
            l = [kl.DepthwiseConv2D(3, strides, padding='same', use_bias=False,
                                    name=f'conv_dw_{number}_dw'), ]

        return l + [kl.BatchNormalization(channel_axis, name=f'conv_dw_{number}_bn_1'),
                    kl.ReLU(name=f'conv_dw_{number}_relu_1'),
                    kl.Conv2D(filters, 1, 1, use_bias=False,
                              kernel_regularizer=k.regularizers.l2(5e-4),
                              name=f'conv_dw_{number}_conv'),
                    kl.BatchNormalization(channel_axis, name=f'conv_dw_{number}_bn_2'),
                    kl.ReLU(name=f'conv_dw_{number}_relu_2')]
    l = (
        # 120*160
        conv_bn(base_filters, 2, 0) +
        conv_dw(base_filters * 2, 1, 1) +
        # 60*80
        conv_dw(base_filters * 2, 2, 2) +
        conv_dw(base_filters * 2, 1, 3) +
        # 30*40
        conv_dw(base_filters * 4, 2, 4) +
        conv_dw(base_filters * 4, 1, 5) +
        conv_dw(base_filters * 4, 1, 6) +
        conv_dw(base_filters * 4, 1, 7) +
        # 15*20
        conv_dw(base_filters * 8, 2, 8) +
        conv_dw(base_filters * 8, 1, 9) +
        conv_dw(base_filters * 8, 1, 10) +
        # 8*10
        conv_dw(base_filters * 16, 2, 11) +
        conv_dw(base_filters * 16, 1, 12))

    return k.Model(inputs, compose(*l)(inputs))


def SeperableConv2d(filters, kernel_size=1, strides=1, padding='valid'):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    if kernel_size == 3 and strides == 2 and padding == 'same':
        l = [kl.ZeroPadding2D(),
             kl.DepthwiseConv2D(kernel_size, strides, padding='valid', use_bias=False)]
    else:
        l = [kl.DepthwiseConv2D(kernel_size, strides, padding, use_bias=False)]

    return compose(*l,
                   kl.ReLU(),
                   kl.Conv2D(filters, kernel_size=1))


def UltraLightFastGenericFaceNet_slim(inputs: k.Input, branch_index=[7, 10, 12],
                                      base_filters=16, num_classes=1) -> k.Model:
    base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters)
    extras = compose(
        kl.Conv2D(base_filters * 4, 1),
        kl.ReLU(),
        kl.ZeroPadding2D(),
        kl.DepthwiseConv2D(3, 2, 'valid'),
        kl.ReLU(),
        kl.Conv2D(base_filters * 16, 1),
        kl.ReLU())

    regression_headers = [SeperableConv2d(3 * 4, 3, padding='same'),
                          SeperableConv2d(2 * 4, 3, padding='same'),
                          SeperableConv2d(2 * 4, 3, padding='same'),
                          kl.Conv2D(3 * 4, 3, padding='same')]

    classification_headers = [SeperableConv2d(3 * num_classes, 3, padding='same'),
                              SeperableConv2d(2 * num_classes, 3, padding='same'),
                              SeperableConv2d(2 * num_classes, 3, padding='same'),
                              kl.Conv2D(3 * num_classes, 3, padding='same')]

    def compute_header(i, x):
        confidence = classification_headers[i](x)
        confidence = kl.Reshape((-1, num_classes))(confidence)

        location = regression_headers[i](x)
        location = kl.Reshape((-1, 4))(location)
        return confidence, location

    y_out = []
    for index in branch_index:
        y_out.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)

    if extras != None:
        y_out.append(extras(base_model.output))

    confidences = []
    locations = []
    for i, y in enumerate(y_out):
        confidence, location = compute_header(i, y)
        confidences.append(confidence)
        locations.append(location)

    confidences = kl.Concatenate(1)(confidences)
    locations = kl.Concatenate(1)(locations)

    model = k.Model(inputs, [locations, confidences])
    return model
