import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
from models.darknet import compose


class Shuffle(kl.Layer):
    def __init__(self, groups: int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.groups = groups

    def build(self, input_shape: tf.TensorShape):
        _, self.h, self.w, self.c = input_shape.as_list()

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.reshape(inputs, [-1, self.h, self.w, self.groups, self.c // self.groups])
        inputs = tf.transpose(inputs, [0, 1, 2, 4, 3])
        inputs = tf.reshape(inputs, [-1, self.h, self.w, self.c])
        return inputs


def conv_bn_relu(filters, kernel_size, strides: int = 1, dilation: int = 1):
    """ Convolution2D followed by BatchNormalization and ReLU."""
    return compose(kl.Conv2D(filters, kernel_size, strides, padding='same', dilation_rate=dilation),
                   kl.BatchNormalization(),
                   kl.ReLU())


def depthwise_conv_bn(kernel_size, strides: int = 1, dilation: int = 1):
    return compose(kl.DepthwiseConv2D(kernel_size, strides, padding='same', dilation_rate=dilation),
                   kl.BatchNormalization())


def deconv_bn_relu(filters: int, kernel_size: int = 4, strides: int = 2):
    return compose(
        kl.Conv2DTranspose(filters, kernel_size, strides, padding='same'),
        kl.BatchNormalization(),
        kl.ReLU())


def shufflenet_block(inputs, filers: int, kernel_size: int, stride=1, dilation=1, shuffle_group=2) -> tf.Tensor:
    if stride == 1:
        top, bottom = kl.Lambda(lambda x: tf.split(x, 2, -1))(inputs)
        half_channel = filers // 2

        top = compose(conv_bn_relu(half_channel, 1),
                      depthwise_conv_bn(kernel_size, stride, dilation),
                      conv_bn_relu(half_channel, 1))(top)

        out = kl.Concatenate()([top, bottom])
        out = Shuffle(shuffle_group)(out)

    else:
        half_channel = filers // 2

        b0 = compose(conv_bn_relu(half_channel, 1),
                     depthwise_conv_bn(kernel_size, stride, dilation), conv_bn_relu(half_channel, 1))(inputs)

        b1 = compose(depthwise_conv_bn(kernel_size, stride, dilation),
                     conv_bn_relu(half_channel, 1))(inputs)

        out = kl.Concatenate()([b0, b1])
        out = Shuffle(shuffle_group)(out)
    return out
