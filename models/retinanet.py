import tensorflow as tf
from models.darknet import compose, DarknetConv2D
from typing import List
k = tf.keras
kl = tf.keras.layers


def Conv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis),
        kl.LeakyReLU(alpha=0.1))


def Conv2D_BN_Relu(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and ReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis),
        kl.ReLU())


def Conv2D_BN(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis))


def SSH(inputs: tf.Tensor, filters: int, depth: int = 3) -> tf.Tensor:
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
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


def retinafacenet(input_shape: list, anchor_num: int,
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

    bbox_out = [kl.Conv2D(anchor_num * 4, 1, 1)(feat) for feat in features]  # BboxHead
    class_out = [kl.Conv2D(anchor_num * 2, 1, 1)(feat) for feat in features]  # ClassHead
    landm_out = [kl.Conv2D(anchor_num * 10, 1, 1)(feat) for feat in features]  # LandmarkHead

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
