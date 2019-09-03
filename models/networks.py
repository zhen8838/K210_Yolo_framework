import tensorflow as tf
from tensorflow.contrib import slim
from models.mobilenet_v2 import training_scope, mobilenet_base
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as kl
from models.keras_mobilenet_v2 import MobileNetV2
from models.keras_mobilenet import MobileNet
from functools import reduce, wraps
from toolz import pipe


def yolo_mobilev1(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape, include_top=False, weights=None, alpha=alpha)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x1 = base_model.get_layer('conv_pw_11_relu').output

    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128 if alpha > 0.8 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        k.layers.UpSampling2D(2))(x2)

    y2 = compose(
        k.layers.Concatenate(),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs, [y1, y2])
    yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yolo_mobilev2(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
    """ build keras mobilenet v2 yolo model, will return two keras model (yolo_model,yolo_model_warpper)
        use yolo_model_warpper training can avoid mismatch error, final use yolo_model to save.

    Parameters
    ----------
    input_shape : list

    anchor_num : int

    class_num : int


    Returns
    -------
    [k.Model, k.Model]
        yolo_model,yolo_model_warpper
    """
    input_tensor = k.Input(input_shape)
    base_model = MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=input_tensor,
        alpha=alpha,
        input_shape=input_shape,
        pooling=None)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v2_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v2_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v2_base_10.h5')

    x1 = base_model.get_layer('block_13_expand_relu').output
    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128 if alpha > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        kl.UpSampling2D(2))(x2)
    y2 = compose(
        kl.Concatenate(),
        DarknetConv2D_BN_Leaky(128 if alpha > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs=input_tensor, outputs=[y1, y2])
    yolo_model_warpper = k.Model(inputs=input_tensor, outputs=[y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yolov2algin_mobilev1(input_shape: list, anchor_num: int, class_num: int, landmark_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape, include_top=False, weights=None, alpha=alpha)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x = base_model.output

    if alpha == .5:
        y = compose(
            DarknetConv2D_BN_Leaky(192, (3, 3)),
            DarknetConv2D_BN_Leaky(192, (3, 3)))(x)
    elif alpha == .75:
        y = compose(
            DarknetConv2D_BN_Leaky(192, (3, 3)),
            DarknetConv2D_BN_Leaky(128, (3, 3)))(x)
    elif alpha == 1.:
        y = compose(
            DarknetConv2D_BN_Leaky(128, (3, 3)),
            DarknetConv2D_BN_Leaky(128, (3, 3)))(x)

    y = DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1))(y)

    y_reshape = kl.Reshape((7, 10, anchor_num, 5 + landmark_num * 2 + class_num), name='l1')(y)

    yolo_model = k.Model(inputs, [y])
    yolo_model_warpper = k.Model(inputs, [y_reshape])

    return yolo_model, yolo_model_warpper


def yoloalgin_mobilev1(input_shape: list, anchor_num: int, class_num: int, landmark_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape, include_top=False, weights=None, alpha=alpha)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x1 = base_model.get_layer('conv_pw_11_relu').output

    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        k.layers.UpSampling2D(2))(x2)

    y2 = compose(
        k.layers.Concatenate(),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + landmark_num * 2 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + landmark_num * 2 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs, [y1, y2])
    yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yoloalgin_mobilev2(input_shape: list, anchor_num: int, class_num: int, landmark_num: int, alpha: float) -> [k.Model, k.Model]:
    """ build keras mobilenet v2 yolo model, will return two keras model (yolo_model,yolo_model_warpper)
        use yolo_model_warpper training can avoid mismatch error, final use yolo_model to save.

    Parameters
    ----------
    input_shape : list

    anchor_num : int

    class_num : int

    landmark_num : int


    Returns
    -------
    [k.Model, k.Model]
        yolo_model,yolo_model_warpper
    """
    input_tensor = k.Input(input_shape)
    base_model = MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=input_tensor,
        alpha=alpha,
        input_shape=input_shape,
        pooling=None)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v2_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v2_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v2_base_10.h5')

    x1 = base_model.get_layer('block_13_expand_relu').output
    x2 = base_model.output

    y1 = compose(
        DarknetConv2D_BN_Leaky(128 if alpha > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        kl.UpSampling2D(2))(x2)
    y2 = compose(
        kl.Concatenate(),
        DarknetConv2D_BN_Leaky(128 if alpha > 0.7 else 192, (3, 3)),
        DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + landmark_num * 2 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + landmark_num * 2 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs=input_tensor, outputs=[y1, y2])
    yolo_model_warpper = k.Model(inputs=input_tensor, outputs=[y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def pfld(input_shape: list, landmark_num: int, alpha=1., weight_decay=5e-5) -> [k.Model, k.Model]:
    """ pfld landmark model

    Parameters
    ----------
    input_shape : list

    landmark_num : int

    alpha : float , optional

        by default 1.

    weight_decay : float , optional

        by default 5e-5

    Returns
    -------
    [k.Model, k.Model]

        infer_model, train_model
    """
    bn_kwargs = {
        'momentum': 0.995,
        'epsilon': 0.001
    }
    conv_kwargs = {
        'kernel_initializer': k.initializers.TruncatedNormal(stddev=0.01),
        'bias_initializer': k.initializers.Zeros(),
        'kernel_regularizer': k.regularizers.l2(weight_decay),
        'bias_regularizer': k.regularizers.l2(weight_decay),
        'padding': 'same'}

    depthwise_kwargs = {
        'depthwise_initializer': k.initializers.TruncatedNormal(stddev=0.01),
        'bias_initializer': k.initializers.Zeros(),
        'depthwise_regularizer': k.regularizers.l2(weight_decay),
        'bias_regularizer': k.regularizers.l2(weight_decay),
        'padding': 'same'}
    # pfld_inference

    # 112*112*3
    inputs = k.Input(input_shape)
    conv1 = pipe(inputs, *[kl.Conv2D(int(64 * alpha), [3, 3], 2, **conv_kwargs),
                           kl.BatchNormalization(**bn_kwargs),
                           kl.ReLU(6)])

    # 56*56*64
    conv2 = pipe(conv1, *[kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                          kl.BatchNormalization(),
                          kl.ReLU(6)])
    # 56*56*64
    conv3_1 = pipe(conv2, *[kl.Conv2D(128, 1, 2, **conv_kwargs),
                            kl.BatchNormalization(),
                            kl.ReLU(6),
                            kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                            kl.BatchNormalization(),
                            kl.ReLU(6),
                            kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                            kl.BatchNormalization(),
                            kl.ReLU(6)])

    conv3_2 = pipe(conv3_1, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    block3_2 = kl.Add()([conv3_1, conv3_2])

    conv3_3 = pipe(block3_2, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_3 = kl.Add()([block3_2, conv3_3])

    conv3_4 = pipe(block3_3, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_4 = kl.Add()([block3_3, conv3_4])

    conv3_5 = pipe(block3_4, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_5 = kl.Add()([block3_4, conv3_5])
    auxiliary_input = block3_5

    conv4_1 = pipe(block3_5, *[kl.Conv2D(128, 1, 2, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])
    # 14*14*128
    conv5_1 = pipe(conv4_1, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    conv5_2 = pipe(conv5_1, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    block5_2 = kl.Add()([conv5_1, conv5_2])

    conv5_3 = pipe(block5_2, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_3 = kl.Add()([block5_2, conv5_3])

    conv5_4 = pipe(block5_3, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_4 = kl.Add()([block5_3, conv5_4])

    conv5_5 = pipe(block5_4, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_5 = kl.Add()([block5_4, conv5_5])

    conv5_6 = pipe(block5_5, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_6 = kl.Add()([block5_5, conv5_6])

    # 14*14*128
    conv6_1 = pipe(block5_6, *[kl.Conv2D(256, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(16 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])
    # 14*14*16
    conv7 = pipe(conv6_1, *[kl.Conv2D(int(32 * alpha), 3, 2, **conv_kwargs),
                            kl.BatchNormalization(),
                            kl.ReLU(6)])

    conv_kwargs['padding'] = 'valid'
    conv8 = pipe(conv7, *[kl.Conv2D(int(128 * alpha), 7, 1, **conv_kwargs),
                          kl.BatchNormalization(),
                          kl.ReLU(6)])

    avg_pool1 = kl.AvgPool2D(conv6_1.shape[1:3], 1)(conv6_1)

    avg_pool2 = kl.AvgPool2D(conv7.shape[1:3], 1)(conv7)

    s1 = kl.Flatten()(avg_pool1)
    s2 = kl.Flatten()(avg_pool2)
    # 1*1*128
    s3 = kl.Flatten()(conv8)

    multi_scale = kl.Concatenate(1)([s1, s2, s3])
    landmark_pre = kl.Dense(landmark_num * 2)(multi_scale)

    pflp_infer_model = k.Model(inputs, landmark_pre)

    conv_kwargs['padding'] = 'same'  # ! 重要
    euler_angles_pre = pipe(auxiliary_input,
                            *[kl.Conv2D(128, 3, 2, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(128, 3, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(32, 3, 2, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(128, 7, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.MaxPool2D(3, 1, 'same'),
                              kl.Flatten(),
                              kl.Dense(32),
                              kl.BatchNormalization(),
                              kl.Dense(3)])

    # NOTE avoid keras loss shape check error
    y_pred = kl.Concatenate(1)([landmark_pre, euler_angles_pre])
    train_model = k.Model(inputs, y_pred)

    return pflp_infer_model, train_model


def pfld_optimized(input_shape: list, landmark_num: int,
                   alpha=1., weight_decay=5e-5) -> [k.Model, k.Model]:
    """ pfld landmark model optimized to fit k210 chip.

    Parameters
    ----------
    input_shape : list

    landmark_num : int

    alpha : float , optional

        by default 1.

    weight_decay : float , optional

        by default 5e-5

    Returns
    -------
    [k.Model, k.Model]

        pflp_infer_model, train_model
    """
    bn_kwargs = {
        'momentum': 0.995,
        'epsilon': 0.001
    }
    conv_kwargs = {
        'kernel_initializer': k.initializers.TruncatedNormal(stddev=0.01),
        'bias_initializer': k.initializers.Zeros(),
        'kernel_regularizer': k.regularizers.l2(weight_decay),
        'bias_regularizer': k.regularizers.l2(weight_decay),
        'padding': 'same'}

    depthwise_kwargs = {
        'depthwise_initializer': k.initializers.TruncatedNormal(stddev=0.01),
        'bias_initializer': k.initializers.Zeros(),
        'depthwise_regularizer': k.regularizers.l2(weight_decay),
        'bias_regularizer': k.regularizers.l2(weight_decay),
        'padding': 'same'}

    conv_kwargs_copy = conv_kwargs.copy()
    depthwise_kwargs_copy = depthwise_kwargs.copy()
    conv_kwargs_copy.pop('padding')
    depthwise_kwargs_copy.pop('padding')

    # 112*112*3
    inputs = k.Input(input_shape)
    conv1 = pipe(inputs, *[kl.ZeroPadding2D(),
                           kl.Conv2D(int(64 * alpha), 3, 2, **conv_kwargs_copy),
                           kl.BatchNormalization(**bn_kwargs),
                           kl.ReLU(6)])

    # 56*56*64
    conv2 = pipe(conv1, *[kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                          kl.BatchNormalization(),
                          kl.ReLU(6)])
    # 28*28*64
    conv3_1 = pipe(conv2, *[
        kl.ZeroPadding2D(),
        kl.Conv2D(128, 3, 2, **conv_kwargs_copy),
        kl.BatchNormalization(),
        kl.ReLU(6),
        kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
        kl.BatchNormalization(),
        kl.ReLU(6),
        kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
        kl.BatchNormalization(),
        kl.ReLU(6)
    ])

    conv3_2 = pipe(conv3_1, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    block3_2 = kl.Add()([conv3_1, conv3_2])

    conv3_3 = pipe(block3_2, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_3 = kl.Add()([block3_2, conv3_3])

    conv3_4 = pipe(block3_3, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_4 = kl.Add()([block3_3, conv3_4])

    conv3_5 = pipe(block3_4, *[kl.Conv2D(128, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block3_5 = kl.Add()([block3_4, conv3_5])
    auxiliary_input = block3_5

    conv4_1 = pipe(block3_5, *[kl.ZeroPadding2D(),
                               kl.Conv2D(128, 3, 2, **conv_kwargs_copy),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    conv5_1 = pipe(conv4_1, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    conv5_2 = pipe(conv5_1, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                              kl.BatchNormalization()])

    block5_2 = kl.Add()([conv5_1, conv5_2])

    conv5_3 = pipe(block5_2, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_3 = kl.Add()([block5_2, conv5_3])

    conv5_4 = pipe(block5_3, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_4 = kl.Add()([block5_3, conv5_4])

    conv5_5 = pipe(block5_4, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_5 = kl.Add()([block5_4, conv5_5])

    conv5_6 = pipe(block5_5, *[kl.Conv2D(512, 1, 1, **conv_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
                               kl.BatchNormalization(),
                               kl.ReLU(6),
                               kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
                               kl.BatchNormalization()])

    block5_6 = kl.Add()([block5_5, conv5_6])  # 14,14,128

    # 7,7,16
    conv6_1 = pipe(block5_6, *[kl.Conv2D(256, 1, 1, **conv_kwargs_copy),
                               kl.BatchNormalization(),
                               kl.ReLU(6),  # 14,14,256
                               kl.ZeroPadding2D(),
                               # can be modify filters
                               kl.Conv2D(int(16 * alpha), 3, 2, **conv_kwargs_copy),
                               kl.BatchNormalization(),
                               kl.ReLU(6),  # 7,7,16
                               ])

    # 7,7,32
    conv7 = pipe(conv6_1, *[
        # can be modify filters
        kl.Conv2D(int(32 * alpha), 3, 1, **conv_kwargs),
        kl.BatchNormalization(),
        kl.ReLU(6)])

    # 7,7,64
    conv8 = pipe(conv7, *[
        # can be modify filters
        kl.Conv2D(int(128 * alpha), 3, 1, **conv_kwargs),
        kl.BatchNormalization(),
        kl.ReLU(6)])
    # 7,7,112
    multi_scale = kl.Concatenate()([conv6_1, conv7, conv8])
    # 7,7,4 = 196  can be modify kernel size
    landmark_pre = kl.Conv2D(landmark_num * 2 // (7 * 7), 3, 1, 'same')(multi_scale)

    pflp_infer_model = k.Model(inputs, landmark_pre)

    euler_angles_pre = pipe(auxiliary_input,
                            *[kl.Conv2D(128, 3, 2, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(128, 3, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(32, 3, 2, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.Conv2D(128, 7, 1, **conv_kwargs),
                              kl.BatchNormalization(),
                              kl.ReLU(6),
                              kl.MaxPool2D(3, 1, 'same'),
                              kl.Flatten(),
                              kl.Dense(32),
                              kl.BatchNormalization(),
                              kl.Dense(3)])

    flatten_landmark_pre = pipe(landmark_pre, *[
        kl.Permute((3, 1, 2)),
        kl.Flatten()
    ])

    y_pred = kl.Concatenate()([flatten_landmark_pre, euler_angles_pre])
    train_model = k.Model(inputs, y_pred)

    return pflp_infer_model, train_model


def tiny_yolo(input_shape, anchor_num, class_num) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    '''Create Tiny YOLO_v3 model CNN body in k.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)

    x2 = compose(
        kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        kl.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)

    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        kl.UpSampling2D(2))(x2)
    y2 = compose(
        kl.Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs, [y1, y2])
    yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

    yolo_weight = k.models.load_model('data/tiny_yolo_weights.h5').get_weights()
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


def yolo(input_shape, anchor_num, class_num) -> [k.Model, k.Model]:
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = k.Input(input_shape)
    darknet = k.Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, anchor_num * (class_num + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), kl.UpSampling2D(2))(x)
    x = kl.Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, anchor_num * (class_num + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), kl.UpSampling2D(2))(x)
    x = kl.Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, anchor_num * (class_num + 5))

    y1_reshape = kl.Reshape((13, 13, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((26, 26, anchor_num, 5 + class_num), name='l2')(y2)
    y3_reshape = kl.Reshape((52, 52, anchor_num, 5 + class_num), name='l3')(y3)

    yolo_model = k.Model(inputs, [y1, y2, y3])
    yolo_model_warpper = k.Model(inputs=inputs, outputs=[y1_reshape, y2_reshape, y3_reshape])

    yolo_weight = k.models.load_model('data/yolo_weights.h5').get_weights()
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
    darknet_conv_kwargs = {'kernel_regularizer': k.regularizers.l2(5e-4)}
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
