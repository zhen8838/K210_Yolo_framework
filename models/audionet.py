import tensorflow as tf
from typing import List
from models.darknet import compose

k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


def dualmbv2net(input_shape: list, class_num: int, depth_multiplier: float = 1.0) -> [k.Model, k.Model, k.Model]:
    inputs = k.Input(input_shape)
    input_label = k.Input(input_shape)
    input_unlabel = k.Input(input_shape)

    conv1 = kl.Conv2D(3, kernel_size=7, strides=2, padding='same', use_bias=False, name='pre_conv')(inputs)

    base_model: k.Model = k.applications.MobileNetV2(input_tensor=conv1,
                                                     include_top=False, weights=None,
                                                     alpha=depth_multiplier)
    base_model.load_weights(f'/home/zqh/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{str(depth_multiplier)}_224_no_top.h5',
                            by_name=True)

    tmp = kl.GlobalMaxPooling2D()(base_model.output)
    base_model = k.Model(inputs, tmp)

    label_out = compose(kl.Dense(1024),
                        kl.LeakyReLU(),
                        kl.Dropout(0.2),
                        kl.Dense(1024),
                        kl.LeakyReLU(),
                        kl.Dropout(0.1),
                        kl.Dense(class_num))(base_model(input_label))

    unlabel_out = compose(kl.Dense(1024),
                          kl.LeakyReLU(),
                          kl.Dropout(0.2),
                          kl.Dense(1024),
                          kl.LeakyReLU(),
                          kl.Dropout(0.1),
                          kl.Dense(class_num))(base_model(input_unlabel))
    outputs = kl.Concatenate()([label_out, unlabel_out])  # [batch,calss_num*2]

    train_model = k.Model([input_label, input_unlabel], outputs)
    val_model = k.Model(input_label, label_out)
    infer_model = val_model

    return infer_model, val_model, train_model


def sslmbv2net(input_shape: list, class_num: int,
               depth_multiplier: float = 1.0) -> k.Model:
    inputs = k.Input(input_shape)

    conv1 = kl.Conv2D(3, kernel_size=7, strides=2, padding='same', use_bias=False, name='pre_conv')(inputs)

    base_model: k.Model = k.applications.MobileNetV2(input_tensor=conv1,
                                                     include_top=False, weights=None,
                                                     alpha=depth_multiplier)
    base_model.load_weights(f'/home/zqh/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{str(depth_multiplier)}_224_no_top.h5',
                            by_name=True)

    tmp = kl.GlobalMaxPooling2D()(base_model.output)

    # [batch,calss_num]
    outputs = compose(kl.Dense(1024),
                      kl.LeakyReLU(),
                      kl.Dropout(0.2),
                      kl.Dense(1024),
                      kl.LeakyReLU(),
                      kl.Dropout(0.1),
                      kl.Dense(class_num))(tmp)

    model = k.Model(inputs, outputs)

    return model
