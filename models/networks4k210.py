import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as kl
from models.keras_mobilenet_v2 import MobileNetV2
from models.keras_mobilenet import MobileNet
from models.mobilenet_v2 import training_scope, mobilenet_base
from models.darknet import darknet_body, DarknetConv2D, DarknetConv2D_BN_Leaky, compose, resblock_body
from toolz import pipe


def yolo_mbv1_k210(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights=None, alpha=alpha)  # type: k.Model

    if alpha == 0.25:
        base_model.load_weights('data/mobilenet_v1_base_2.h5')
    elif alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x1 = base_model.get_layer('conv_pw_11_relu').output

    x2 = base_model.output

    if alpha == 0.25:
        filters = 192
    elif alpha == 0.5:
        filters = 128
    elif alpha == 0.75:
        filters = 128
    elif alpha == 1.0:
        filters = 128

    y1 = compose(
        DarknetConv2D_BN_Leaky(filters, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        k.layers.UpSampling2D(2))(x2)

    y2 = compose(
        k.layers.Concatenate(),
        DarknetConv2D_BN_Leaky(filters, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs, [y1, y2])
    yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yolo_mbv2_k210(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
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

    if alpha == .35:
        base_model.load_weights('data/mobilenet_v2_base_3.h5')
    elif alpha == .5:
        base_model.load_weights('data/mobilenet_v2_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v2_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v2_base_10.h5')

    x1 = base_model.get_layer('block_13_expand_relu').output
    x2 = base_model.output

    if alpha == 0.35:
        filters = 128
    elif alpha == 0.5:
        filters = 128
    elif alpha == 0.75:
        filters = 128
    elif alpha == 1.0:
        filters = 128

    y1 = compose(
        DarknetConv2D_BN_Leaky(filters, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(x2)
    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        kl.UpSampling2D(2))(x2)
    y2 = compose(
        kl.Concatenate(),
        DarknetConv2D_BN_Leaky(filters, (3, 3)),
        DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

    y1_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((14, 20, anchor_num, 5 + class_num), name='l2')(y2)

    yolo_model = k.Model(inputs=input_tensor, outputs=[y1, y2])
    yolo_model_warpper = k.Model(inputs=input_tensor, outputs=[y1_reshape, y2_reshape])

    return yolo_model, yolo_model_warpper


def yolo2_mbv1_k210(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
    """ build keras mobilenet v1 yolo v2 model, will return two keras model (yolo_model,yolo_model_warpper)
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
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights=None, alpha=alpha)  # type: keras.Model

    if alpha == .25:
        base_model.load_weights('data/mobilenet_v1_base_2.h5')
    elif alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x = base_model.output

    if alpha == 0.25:
        filters_1 = 256
        filters_2 = 128
    elif alpha == 0.5:
        filters_1 = 256
        filters_2 = 128
    elif alpha == 0.75:
        filters_1 = 192
        filters_2 = 128
    elif alpha == 1.0:
        filters_1 = 128
        filters_2 = 128

    y = compose(
        DarknetConv2D_BN_Leaky(filters_1, (3, 3)),
        DarknetConv2D_BN_Leaky(filters_2, (3, 3)))(x)

    y = DarknetConv2D(anchor_num * (class_num + 5), (1, 1))(y)

    y_reshape = kl.Reshape((7, 10, anchor_num, 5 + class_num), name='l1')(y)

    yolo_model = k.Model(inputs, [y])
    yolo_model_warpper = k.Model(inputs, [y_reshape])

    return yolo_model, yolo_model_warpper


def yolov2algin_mbv1_k210(input_shape: list, anchor_num: int,
                          class_num: int, landmark_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights=None, alpha=alpha)  # type: k.Model

    if alpha == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif alpha == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif alpha == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    x1 = base_model.get_layer('conv_pw_11_relu').output  # [14,20,256]

    x2 = base_model.output  # [7,10,512]

    x1 = resblock_body(x1, 128, 2)  # 7,10,128

    if alpha == .5:
        filter_num = 192
    else:
        filter_num = 128

    y = compose(
        k.layers.Concatenate(),
        DarknetConv2D_BN_Leaky(filter_num, (3, 3)),
        DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))([x1, x2])  # [7,10,48]

    y_reshape = kl.Reshape((7, 10, anchor_num, 5 + landmark_num * 2 + class_num), name='l1')(y)

    yolo_model = k.Model(inputs, [y])
    yolo_model_warpper = k.Model(inputs, [y_reshape])

    return yolo_model, yolo_model_warpper


def pfld_k210(input_shape: list, landmark_num: int,
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


def mbv1_triplet_facerec_k210(input_shape: list, embedding_size: int,
                              depth_multiplier: float = 1.0) -> [k.Model, k.Model]:
    in_a = k.Input(input_shape, name='in_a')
    in_p = k.Input(input_shape, name='in_p')
    in_n = k.Input(input_shape, name='in_n')

    """ build dummy model body """

    base_model = MobileNet(input_tensor=in_a, input_shape=input_shape,
                           include_top=False, alpha=depth_multiplier)  # type: keras.Model
    if depth_multiplier == .25:
        base_model.load_weights('data/mobilenet_v1_base_2.h5')
    elif depth_multiplier == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif depth_multiplier == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif depth_multiplier == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    w = base_model.output.shape.as_list()[1]

    embedd = kl.Conv2D(128 // (w * w), 1, use_bias=False)(base_model.output)
    out_a = compose(kl.Permute([3, 1, 2]),
                    kl.Flatten())(embedd)

    encoder_model = k.Model(in_a, embedd)
    infer_model = k.Model(in_a, out_a)

    out_p = infer_model(in_p)
    out_n = infer_model(in_n)

    """ build train model """
    train_model = k.Model([in_a, in_p, in_n], kl.Concatenate()([out_a, out_p, out_n]))

    return encoder_model, train_model


def mbv1_softmax_facerec_k210(input_shape: list, class_num: int,
                              embedding_size: int, depth_multiplier: float = 1.0) -> [k.Model, k.Model]:
    """ mobilenet v1 face recognition model for softmax loss

    Parameters
    ----------
    input_shape : list

    class_num : int

        all class num

    embedding_size : int

    depth_multiplier : float, optional

        by default 1.0

    Returns
    -------

    [k.Model, k.Model]

       encoder,train_model 

    """
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights=None,
                           alpha=depth_multiplier)  # type: keras.Model

    if depth_multiplier == .25:
        base_model.load_weights('data/mobilenet_v1_base_2.h5')
    elif depth_multiplier == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif depth_multiplier == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif depth_multiplier == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    w = base_model.output.shape.as_list()[1]

    embedds = kl.Conv2D(128 // (w * w), 1, use_bias=False)(base_model.output)
    outputs = compose(
        kl.Permute([3, 1, 2]),
        kl.Flatten(),
        kl.Dense(class_num, use_bias=False))(embedds)

    infer_model = k.Model(inputs, embedds)  # encoder to infer
    train_model = k.Model(inputs, outputs)  # full model to train
    return infer_model, train_model


def mbv1_amsoftmax_facerec_k210(input_shape: list, class_num: int,
                                embedding_size: int, depth_multiplier: float = 1.0) -> [k.Model, k.Model]:
    """ mobilenet v1 face recognition model for Additve Margin Softmax loss

    Parameters
    ----------
    input_shape : list

    class_num : int

        all class num

    embedding_size : int

    depth_multiplier : float, optional

        by default 1.0

    Returns
    -------

    [k.Model, k.Model]

       encoder,train_model 

    """
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights=None,
                           alpha=depth_multiplier)  # type: keras.Model

    if depth_multiplier == .25:
        base_model.load_weights('data/mobilenet_v1_base_2.h5')
    elif depth_multiplier == .5:
        base_model.load_weights('data/mobilenet_v1_base_5.h5')
    elif depth_multiplier == .75:
        base_model.load_weights('data/mobilenet_v1_base_7.h5')
    elif depth_multiplier == 1.:
        base_model.load_weights('data/mobilenet_v1_base_10.h5')

    w = base_model.output.shape.as_list()[1]
    embedds = kl.Conv2D(128 // (w * w), 1, use_bias=False)(base_model.output)

    outputs = compose(
        kl.Permute([3, 1, 2]),
        kl.Flatten(),
        # normalize Classification vector len = 1
        kl.Lambda(lambda x: tf.math.l2_normalize(x, 1)),
        kl.Dense(class_num, use_bias=False,
                 # normalize Classification Matrix len = 1
                 # f·W = (f·W)/(‖f‖×‖W‖) = (f·W)/(1×1) = cos(θ)
                 kernel_constraint=k.constraints.unit_norm()))(embedds)

    infer_model = k.Model(inputs, embedds)  # encoder to infer
    train_model = k.Model(inputs, outputs)  # full model to train
    return infer_model, train_model
