import tensorflow as tf
from models.darknet import compose, DarknetConv2D
from typing import List
k = tf.keras
kl = tf.keras.layers


def Conv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis),
        kl.LeakyReLU(alpha=0.1))


def Conv2D_BN(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        kl.BatchNormalization(channel_axis))


def SSH(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    conv3X3 = Conv2D_BN(filters // 2, 3, 1, padding='same')(inputs)
    conv5X5_1 = Conv2D_BN_Leaky(filters // 4, 3, 1, padding='same')(inputs)
    conv5X5 = Conv2D_BN(filters // 4, 3, 1, padding='same')(conv5X5_1)
    conv7X7_2 = Conv2D_BN_Leaky(filters // 4, 3, 1, padding='same')(conv5X5_1)
    conv7X7 = Conv2D_BN(filters // 4, 3, 1, padding='same')(conv7X7_2)
    out = kl.Concatenate(1)([conv3X3, conv5X5, conv7X7])
    out = kl.ReLU()(out)
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


def retinafacenet(input_shape: list, anchor_num: int, filters: int, alpha=0.25
                  ) -> [k.Model, k.Model]:
    k.backend.set_image_data_format('channels_first')
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

    infer_model = k.Model(inputs, sum([[b, l, c] for b, l, c in zip(bbox_out, landm_out, class_out)], []))

    bbox_out = [kl.Reshape((-1, 4))(kl.Permute((2, 3, 1))(b)) for b in bbox_out]
    landm_out = [kl.Reshape((-1, 10))(kl.Permute((2, 3, 1))(b)) for b in landm_out]
    class_out = [kl.Reshape((-1, 2))(kl.Permute((2, 3, 1))(b)) for b in class_out]

    bbox_out = kl.Concatenate(1)(bbox_out)
    landm_out = kl.Concatenate(1)(landm_out)
    class_out = kl.Concatenate(1)(class_out)
    out = kl.Concatenate()([bbox_out, landm_out, class_out])

    train_model = k.Model(inputs, out)

    return infer_model, train_model
