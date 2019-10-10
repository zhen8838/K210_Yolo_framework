import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.regularizers as kr
from models.darknet import compose, DarknetConv2D_BN_Leaky


def PEP(inputs: tf.Tensor, projection_channel: int, expand_channel=None,
        pointwise_channel=None, expansion: int = .5) -> tf.Tensor:
    in_channels = inputs.shape.as_list()[-1]
    if pointwise_channel is None:
        pointwise_channel = in_channels
    x = inputs
    x = DarknetConv2D_BN_Leaky(projection_channel, 1, 1)(inputs)

    # Expand
    x = DarknetConv2D_BN_Leaky(expand_channel if expand_channel else int(expansion * in_channels),
                               1, 1)(x)
    # Depthwise
    x = compose(kl.DepthwiseConv2D(3, 1, 'same', use_bias=False),
                kl.BatchNormalization(epsilon=1e-3, momentum=0.999),
                kl.LeakyReLU())(x)

    # Project
    x = kl.Conv2D(pointwise_channel, 1, 1, 'same', use_bias=False)(x)
    x = kl.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    if in_channels == pointwise_channel:
        return kl.Add()([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def EP(inputs: tf.Tensor, filters: int, stride: int = 2, expansion: int = 0.5,
       expand_channel=None, pointwise_channel=None) -> tf.Tensor:
    in_channels = inputs.shape.as_list()[-1]
    pointwise_filters = _make_divisible(filters, 8)
    x = inputs
    # Expand
    x = compose(kl.Conv2D(expand_channel if expand_channel else int(expansion * in_channels), 1, 1, 'same', use_bias=False),
                kl.BatchNormalization(epsilon=1e-3, momentum=0.999),
                kl.LeakyReLU())(x)

    # Depthwise
    if stride == 2:
        # x = tf.space_to_batch(x, [[1, 1], [1, 1]], 1, name=prefix + 'pad')
        x = kl.ZeroPadding2D(padding=[[1, 1], [1, 1]])(x)

    x = compose(kl.DepthwiseConv2D(3, stride,
                                   'same' if stride == 1 else 'valid',
                                   use_bias=False),
                kl.BatchNormalization(epsilon=1e-3, momentum=0.999),
                kl.LeakyReLU())(x)

    # Project
    x = kl.Conv2D(pointwise_channel if pointwise_channel else pointwise_filters,
                  1, 1, 'same', use_bias=False)(x)
    x = kl.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    if (in_channels == pointwise_filters and pointwise_channel is None) or in_channels == pointwise_channel:
        return kl.Add()([inputs, x])
    return x


def transform_layer(inputs: tf.Tensor, stride: int, filters: int = 64):
    """ transform to conv bn leakyrelu """
    x = DarknetConv2D_BN_Leaky(filters, 1, 1)(inputs)
    x = DarknetConv2D_BN_Leaky(filters, 3, stride)(x)
    return x


def split_layer(inputs: tf.Tensor, stride: int, num_split: int = 8):
    splitted_branches = list()
    for i in range(num_split):
        branch = transform_layer(inputs, stride)
        splitted_branches.append(branch)

    return kl.Concatenate(-1)(splitted_branches)


def SE(inputs, filters: int, reduction_ratio: int = 4) -> tf.Tensor:
    squeeze = kl.GlobalAveragePooling2D()(inputs)
    excitation = kl.Dense(units=filters // reduction_ratio)(squeeze)
    excitation = kl.LeakyReLU()(excitation)
    excitation = kl.Dense(units=filters)(excitation)
    excitation = kl.Activation('sigmoid')(excitation)
    excitation = kl.Reshape((1, 1, filters))(excitation)

    scale = kl.Multiply()([inputs, excitation])
    return scale


def FCA(inputs: tf.Tensor, reduction_ratio: int):
    in_channels = inputs.shape.as_list()[-1]
    x = split_layer(inputs, 1)
    x = DarknetConv2D_BN_Leaky(in_channels, 1, 1)(x)
    x = SE(x, in_channels, 8)

    x = kl.Add()([inputs, x])
    x = kl.LeakyReLU()(x)
    return x


def yolo3_nano(input_shape: list, anchor_num: int, class_num: int) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    x_1 = DarknetConv2D_BN_Leaky(12, (3, 3))(inputs)  # 416,416,12
    x_2 = DarknetConv2D_BN_Leaky(24, (3, 3), 2)(x_1)  # 208,208,24
    x_3 = PEP(x_2, 7)  # 208, 208, 24
    x_4 = EP(x_3, 24, 2, pointwise_channel=70)  # 104,104,70
    x_5 = PEP(x_4, 25)  # 104, 104, 70
    x_6 = PEP(x_5, 24)  # 104, 104, 70
    x_7 = EP(x_6, 24, 2, pointwise_channel=150)  # 52,52,150
    x_8 = PEP(x_7, 56)  # 52,52,150
    x_9 = DarknetConv2D_BN_Leaky(150, 1, 1)(x_8)  # 52, 52, 150
    x_10 = FCA(x_9, 150)  # 52, 52, 150
    x_11 = PEP(x_10, 73)  # 52, 52, 150
    x_12 = PEP(x_11, 71)  # 52, 52, 150
    x_13 = PEP(x_12, 75)  # ! 52, 52, 150 要给到 34
    x_14 = EP(x_13, 32, 2, pointwise_channel=325)  # 26,26,325
    x_15 = PEP(x_14, 132)  # 26, 26, 325
    x_16 = PEP(x_15, 132)  # 26, 26, 325
    x_17 = PEP(x_16, 132)  # 26, 26, 325
    x_18 = PEP(x_17, 132)  # 26, 26, 325
    x_19 = PEP(x_18, 132)  # 26, 26, 325
    x_20 = PEP(x_19, 132)  # 26, 26, 325
    x_21 = PEP(x_20, 132)  # 26, 26, 325
    x_22 = PEP(x_21, 132)  # ! 26, 26, 325 要给到 30
    x_23 = EP(x_22, 48, 2, pointwise_channel=545)  # 13, 13, 545
    x_24 = PEP(x_23, 276)  # 13, 13, 545
    x_25 = DarknetConv2D_BN_Leaky(230, 1, 1)(x_24)  # 13, 13, 230
    x_26 = EP(x_25, 48, 1, pointwise_channel=489)  # 13, 13, 489
    x_27 = PEP(x_26, 213)  # 13, 13, 489
    x_28 = DarknetConv2D_BN_Leaky(189, 1, 1)(x_27)  # ! 13, 13, 189  给到 40
    x_29 = DarknetConv2D_BN_Leaky(105, 1, 1)(x_28)  # 13, 13, 105
    """ output3 """
    x_30 = PEP(kl.Concatenate(-1)([kl.UpSampling2D()(x_29), x_22]), 113, pointwise_channel=325)  # 26,26,325
    x_31 = PEP(x_30, 99, pointwise_channel=207)  # 26,26,207
    x_32 = DarknetConv2D_BN_Leaky(98, 1, 1)(x_31)  # !  26,26,98 要给到 38
    x_33 = DarknetConv2D_BN_Leaky(47, 1, 1)(x_32)  # 26,26,47
    x_34 = PEP(kl.Concatenate(-1)([kl.UpSampling2D()(x_33), x_13]), 58, pointwise_channel=122)  # 52,52,122
    x_35 = PEP(x_34, 52, pointwise_channel=87)  # 52,52,87
    x_36 = PEP(x_35, 47, pointwise_channel=93)  # 52,52,93
    y3 = kl.Conv2D(anchor_num * (5 + class_num), 1, 1, 'same', use_bias=False)(x_36)  # (52,52,anchor_num* (5 + class_num))
    """ output2 """
    x_37 = EP(x_32, 64, 1, pointwise_channel=183)  # 26, 26, 183
    y2 = kl.Conv2D(anchor_num * (5 + class_num), 1, 1, 'same', use_bias=False)(x_37)  # (26,26,anchor_num* (5 + class_num))
    """ output1 """
    x_38 = EP(x_28, 64, 1, pointwise_channel=462)  # 13, 13, 462
    y1 = kl.Conv2D(anchor_num * (5 + class_num), 1, 1, 'same', use_bias=False)(x_38)  # (13,13,anchor_num* (5 + class_num))

    y1_reshape = kl.Reshape((13, 13, anchor_num, 5 + class_num), name='l1')(y1)
    y2_reshape = kl.Reshape((26, 26, anchor_num, 5 + class_num), name='l2')(y2)
    y3_reshape = kl.Reshape((52, 52, anchor_num, 5 + class_num), name='l3')(y3)

    infer_model = k.Model(inputs, [y1, y2, y3])
    train_model = k.Model(inputs=inputs, outputs=[y1_reshape, y2_reshape, y3_reshape])
    return infer_model, train_model
