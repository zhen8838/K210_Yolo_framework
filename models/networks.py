import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from tensorflow.keras.applications import MobileNet, MobileNetV2
from models.dcn import DCN, DeconvLayer
from toolz import pipe
from models.darknet import DarknetConv2D, darknet_body, DarknetConv2D_BN_Leaky, compose, \
    make_last_layers, make_last_layers_mobilenet, MobilenetConv2D
from models.shufflenet import conv_bn_relu, shufflenet_block, deconv_bn_relu
from models.yolo_nano import yolo3_nano
from models.retinanet import retinafacenet, retinaface_rfb, retinaface_slim, ullfd_slim
from models.facenet import mbv1_facerec


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

    conv_kwargs['padding'] = 'same'
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

    y1_reshape = kl.Lambda(lambda x: x, name='l1')(y1)
    y2_reshape = kl.Lambda(lambda x: x, name='l2')(y2)

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

    y1_reshape = kl.Lambda(lambda x: x, name='l1')(y1)
    y2_reshape = kl.Lambda(lambda x: x, name='l2')(y2)
    y3_reshape = kl.Lambda(lambda x: x, name='l3')(y3)

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


def mbv2_ctdet(input_shape: list, class_num: int, filter_num: list, kernel_num: list, alpha: float = 1.0):
    input_tensor = k.Input(input_shape)
    body_model = MobileNetV2(include_top=False,
                             input_tensor=input_tensor,
                             alpha=alpha,
                             input_shape=input_shape,
                             pooling=None)  # type:k.Model

    out = DeconvLayer(len(filter_num), filter_num, kernel_num)(body_model.output)  # shape=(?, 96, 96, 64)

    out_heatmap = pipe(out, *[kl.Conv2D(64, 3, 1, 'same'),
                              kl.LeakyReLU(),
                              kl.Conv2D(class_num, 1, 1)])

    out_wh = pipe(out, *[kl.Conv2D(64, 3, 1, 'same'),
                         kl.LeakyReLU(),
                         kl.Conv2D(2, 1, 1)])

    out_reg = pipe(out, *[kl.Conv2D(64, 3, 1, 'same'),
                          kl.LeakyReLU(),
                          kl.Conv2D(2, 1, 1)])

    infer_model = k.Model(input_tensor, [out_heatmap, out_wh, out_reg])

    train_model = k.Model(input_tensor, kl.Concatenate()([out_heatmap, out_wh, out_reg]))

    return infer_model, train_model


def shuffle_ctdet(input_shape: list, class_num: int,
                  depth_multiplier: float = 1.0,
                  first_filters: int = 24, feature_filters: int = 256,
                  shuffle_group: int = 2) -> [k.Model, k.Model]:
    if depth_multiplier == 0.5:
        channel_sizes = [(48, 4), (96, 8), (192, 4), (1024, 1)]
    elif depth_multiplier == 1.0:
        channel_sizes = [(116, 4), (232, 8), (464, 4), (1024, 1)]
    elif depth_multiplier == 1.5:
        channel_sizes = [(176, 4), (352, 8), (704, 4), (1024, 1)]
    elif depth_multiplier == 2.0:
        channel_sizes = [(244, 4), (488, 8), (976, 4), (2048, 1)]
    else:
        raise ValueError(f'depth_multiplier is not in [0.5,1.0,1.5,2.0]')

    inputs = k.Input(input_shape)
    """ stage 4 """
    out_2 = conv_bn_relu(first_filters, 3, 2)(inputs)
    out_4 = conv_bn_relu(first_filters, 3, 2)(out_2)
    """ stage_8  """
    out_channel, repeat = channel_sizes[0]
    out_8 = shufflenet_block(out_4, out_channel, 3, 2,
                             shuffle_group=shuffle_group)
    for i in range(repeat - 1):
        out_8 = shufflenet_block(out_8, out_channel, 3,
                                 shuffle_group=shuffle_group)
    """ stage 16 """
    out_channel, repeat = channel_sizes[1]
    out_16 = shufflenet_block(out_8, out_channel, 3, 2,
                              shuffle_group=shuffle_group)  # /16
    for i in range(repeat - 1):
        out_16 = shufflenet_block(out_16, out_channel, 3,
                                  shuffle_group=shuffle_group)
    """ stage 32 """
    out_channel, repeat = channel_sizes[2]
    # First block is downsampling
    out_32 = shufflenet_block(out_16, out_channel, 3, 2,
                              shuffle_group=shuffle_group)  # /32
    for i in range(repeat - 1):
        out_32 = shufflenet_block(out_32, out_channel, 3,
                                  shuffle_group=shuffle_group)
    """ feature map fuse """
    deconv1 = deconv_bn_relu(feature_filters)(out_32)
    out_16 = conv_bn_relu(feature_filters, 1)(out_16)
    fuse1 = kl.Add()([deconv1, out_16])

    deconv2 = deconv_bn_relu(feature_filters)(fuse1)
    out_8 = conv_bn_relu(feature_filters, 1)(out_8)
    fuse2 = kl.Add()([out_8, deconv2])

    deconv3 = deconv_bn_relu(feature_filters)(fuse2)
    out_4 = conv_bn_relu(feature_filters, 1)(out_4)
    fuse3 = kl.Add()([out_4, deconv3])  # 96 96 256

    """ detector """
    out_heatmap = compose(kl.Conv2D(feature_filters, 3, 1, 'same'),
                          kl.ReLU(),
                          kl.Conv2D(class_num, 1, 1, 'same'))(fuse3)
    out_wh = compose(kl.Conv2D(feature_filters, 3, 1, 'same'),
                     kl.ReLU(),
                     kl.Conv2D(2, 1, 1, 'same'))(fuse3)

    out_offset = compose(kl.Conv2D(feature_filters, 3, 1, 'same'),
                         kl.ReLU(),
                         kl.Conv2D(2, 1, 1, 'same'))(fuse3)

    infer_model = k.Model(inputs, [out_heatmap, out_wh, out_offset])

    outputs = kl.Concatenate()([out_heatmap, out_wh, out_offset])

    train_model = k.Model(inputs, outputs)

    return infer_model, train_model


def yolo_mbv1(input_shape: list, anchor_num: int, class_num: int, alpha: float) -> [k.Model, k.Model]:
    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights='imagenet', alpha=alpha)  # type: k.Model

    x, y1 = make_last_layers_mobilenet(base_model.output, 17, 512, anchor_num * (class_num + 5))

    x = compose(kl.Conv2D(256, kernel_size=1,
                          padding='same', use_bias=False,
                          name='block_20_conv'),
                kl.BatchNormalization(momentum=0.9, name='block_20_BN'),
                kl.ReLU(6., name='block_20_relu6'),
                kl.UpSampling2D(2))(x)

    x = kl.Concatenate()([x, MobilenetConv2D((1, 1), alpha, 384)(base_model.get_layer('conv_pw_11_relu').output)])

    x, y2 = make_last_layers_mobilenet(x, 21, 256, anchor_num * (class_num + 5))

    x = compose(
        kl.Conv2D(128, kernel_size=1, padding='same', use_bias=False, name='block_24_conv'),
        kl.BatchNormalization(momentum=0.9, name='block_24_BN'),
        kl.ReLU(6., name='block_24_relu6'),
        kl.UpSampling2D(2))(x)

    x = kl.Concatenate()([x, MobilenetConv2D((1, 1), alpha, 128)(base_model.get_layer('conv_pw_5_relu').output)])
    x, y3 = make_last_layers_mobilenet(x, 25, 128, anchor_num * (class_num + 5))

    y1_reshape = kl.Lambda(lambda x: x, name='y1')(y1)
    y2_reshape = kl.Lambda(lambda x: x, name='y2')(y2)
    y3_reshape = kl.Lambda(lambda x: x, name='y3')(y3)

    infer_model = k.Model(inputs, [y1, y2, y3])
    train_model = k.Model(inputs=inputs, outputs=[y1_reshape, y2_reshape, y3_reshape])
    return infer_model, train_model


def mbv1_imgnet(input_shape: list, class_num: int, depth_multiplier: float = 1.0, weights=None):

    inputs = k.Input(input_shape)
    model = MobileNet(input_tensor=inputs, input_shape=tuple(input_shape),
                      include_top=True, weights=weights,
                      alpha=depth_multiplier, classes=class_num)  # type: keras.Model

    return model, model


def mbv2_imgnet(input_shape: list, class_num: int, depth_multiplier: float = 1.0, weights=None):

    inputs = k.Input(input_shape)
    model = MobileNetV2(input_tensor=inputs, input_shape=tuple(input_shape),
                        include_top=True, weights=weights,
                        alpha=depth_multiplier, classes=class_num)  # type: keras.Model

    return model, model
