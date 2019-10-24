import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.regularizers as kr
from functools import reduce, wraps


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = kl.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = kl.Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(kl.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': kr.l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return kl.Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(),
        kl.LeakyReLU(alpha=0.1))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)
    return compose(kl.Conv2D(last_block_filters, kernel, padding='same', use_bias=False),
                   kl.BatchNormalization(),
                   kl.LeakyReLU())


def MobilenetSeparableConv2D(filters, kernel_size, strides=(1, 1),
                             padding='valid', use_bias=True):
    return compose(kl.DepthwiseConv2D(kernel_size, padding=padding,
                                      use_bias=use_bias, strides=strides),
                   kl.BatchNormalization(),
                   kl.LeakyReLU(),
                   kl.Conv2D(filters, 1, padding='same',
                             use_bias=use_bias, strides=1),
                   kl.BatchNormalization(),
                   kl.LeakyReLU())


def make_last_layers_mobilenet(x, id, num_filters, out_filters):
    x = compose(kl.Conv2D(num_filters, kernel_size=1, padding='same',
                          use_bias=False, name='block_' + str(id) + '_conv'),
                kl.BatchNormalization(momentum=0.9, name='block_' + str(id) + '_BN'),
                kl.LeakyReLU(name='block_' + str(id) + '_relu6'),
                MobilenetSeparableConv2D(2 * num_filters, kernel_size=(3, 3),
                                         use_bias=False, padding='same'),
                kl.Conv2D(num_filters, kernel_size=1,
                          padding='same', use_bias=False,
                          name='block_' + str(id + 1) + '_conv'),
                kl.BatchNormalization(momentum=0.9, name='block_' + str(id + 1) + '_BN'),
                kl.LeakyReLU(name='block_' + str(id + 1) + '_relu6'),
                MobilenetSeparableConv2D(2 * num_filters, kernel_size=(3, 3),
                                         use_bias=False, padding='same'),
                kl.Conv2D(num_filters, kernel_size=1,
                          padding='same', use_bias=False,
                          name='block_' + str(id + 2) + '_conv'),
                kl.BatchNormalization(momentum=0.9, name='block_' + str(id + 2) + '_BN'),
                kl.LeakyReLU(name='block_' + str(id + 2) + '_relu6'))(x)

    y = compose(
        MobilenetSeparableConv2D(2 * num_filters, kernel_size=(3, 3),
                                 use_bias=False, padding='same'),
        kl.Conv2D(out_filters, kernel_size=1,
                  padding='same', use_bias=False))(x)
    return x, y
