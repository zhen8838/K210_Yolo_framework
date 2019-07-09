import tensorflow.python as tf
from tensorflow.contrib import slim
from models.mobilenet_v2 import training_scope, mobilenet_base
from tensorflow.python import keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import *
from models.keras_mobilenet_v2 import MobileNetV2
from models.keras_mobilenet import MobileNet
from functools import reduce, wraps


def yolo_mobilev1(input_shape: list, anchor_num: int, class_num: int, **kwargs) -> [keras.Model, keras.Model]:
    inputs = keras.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape, include_top=False, weights=None, alpha=kwargs['alpha'])  # type: keras.Model

    if kwargs['alpha'] == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif kwargs['alpha'] == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif kwargs['alpha'] == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x1 = base_model.get_layer('conv_pw_11_relu').output

    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128 if kwargs['alpha'] > 0.8 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        keras.layers.UpSampling2D(2))(x2)

    y2 = compose(
        keras.layers.Concatenate(),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = keras.Model(inputs, [y1, y2])
    yolo_model_warpper = keras.Model(inputs, [y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yolo_mobilev2(input_shape: list, anchor_num: int, class_num: int, **kwargs) -> [keras.Model, keras.Model]:
    """ build keras mobilenet v2 yolo model, will return two keras model (yolo_model,yolo_model_warpper)
        use yolo_model_warpper training can avoid mismatch error, final use yolo_model to save.

    Parameters
    ----------
    input_shape : list

    anchor_num : int

    class_num : int


    Returns
    -------
    [keras.Model, keras.Model]
        yolo_model,yolo_model_warpper
    """
    input_tensor = keras.Input(input_shape)
    base_model = MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=input_tensor,
        alpha=kwargs['alpha'],
        input_shape=input_shape,
        pooling=None)  # type: keras.Model

    if kwargs['alpha'] == .5:
        base_model.load_weights('data/mobilenet_v2_base_5.h5')
    elif kwargs['alpha'] == .75:
        base_model.load_weights('data/mobilenet_v2_base_7.h5')
    elif kwargs['alpha'] == 1.:
        base_model.load_weights('data/mobilenet_v2_base_10.h5')

    x1 = base_model.get_layer('block_13_expand_relu').output
    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128 if kwargs['alpha'] > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(128 if kwargs['alpha'] > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = keras.Model(inputs=input_tensor, outputs=[y1, y2])
    yolo_model_warpper = keras.Model(inputs=input_tensor, outputs=[y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def tiny_yolo(input_shape, anchor_num, class_num, **kwargs) -> [keras.Model, keras.Model]:
    inputs = keras.Input(input_shape)
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)

    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)

    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = keras.Model(inputs, [y1, y2])
    yolo_model_warpper = keras.Model(inputs, [y1_reshape, y2_reshape])

    yolo_weight = keras.models.load_model('data/tiny_yolo_weights.h5').get_weights()
    for i, w in enumerate(yolo_weight):
        if w.shape == (1, 1, 1024, 255):
            yolo_weight[i] = w[..., :anchor_num * (class_num + 5)]
        if w.shape == (1, 1, 512, 255):
            yolo_weight[i] = w[..., :anchor_num * (class_num + 5)]
        if w.shape == (1, 1, 256, 255):
            yolo_weight[i] = w[..., :anchor_num * (class_num + 5)]
        if w.shape == (255,):
            yolo_weight[i] = w[:anchor_num * (class_num + 5)]
    yolo_model.set_weights(yolo_weight)

    return yolo_model, yolo_model_warpper


def yolo(input_shape, anchor_num, class_num, **kwargs) -> [keras.Model, keras.Model]:
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = keras.Input(input_shape)
    darknet = keras.Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, anchor_num * (class_num + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, anchor_num * (class_num + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, anchor_num * (class_num + 5))

    y1_reshape = Reshape((13, 13, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = Reshape((26, 26, anchor_num, 5 + class_num), name='l2')(y2)
    y3_reshape = Reshape((52, 52, anchor_num, 5 + class_num), name='l3')(y3)

    yolo_model = keras.Model(inputs, [y1, y2, y3])
    yolo_model_warpper = keras.Model(inputs=inputs, outputs=[y1_reshape, y2_reshape, y3_reshape])

    yolo_weight = keras.models.load_model('data/yolo_weights.h5').get_weights()
    new_weights = yolo_model.get_weights()
    for i in range(len(yolo_weight)):
        minshape = [min(new_weights[i].shape[j], yolo_weight[i].shape[j]) for j in range(len(yolo_weight[i].shape))]
        newshape = tuple([slice(0, s) for s in minshape])
        new_weights[i][newshape] = yolo_weight[i][newshape]

    yolo_model.set_weights(new_weights)

    return yolo_model, yolo_model_warpper


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
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


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))
