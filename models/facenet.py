import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from typing import List, Callable
from tensorflow.python.keras.utils.tf_utils import smart_cond
from tensorflow.python.keras.applications import MobileNet, MobileNetV2
from models.darknet import compose


def mbv1_facerec(input_shape: list, class_num: int,
                 embedding_size: int, depth_multiplier: float = 1.0,
                 loss='amsoftmax') -> [k.Model, k.Model]:
    """ mobilenet v1 face recognition model for Additve Margin Softmax loss

    Parameters
    ----------
    input_shape : list

    class_num : int

        all class num

    embedding_size : int

    depth_multiplier : float, optional

        by default 1.0

    loss : str

        loss in ['softmax','amsoftmax','triplet']

    Returns
    -------

    [k.Model, k.Model]

       encoder,train_model

    """
    loss_list = ['softmax', 'asoftmax', 'amsoftmax', 'triplet']
    if loss not in loss_list:
        raise ValueError(f"loss not valid! must in {' '.join(loss_list)}")

    inputs = k.Input(input_shape)
    base_model = MobileNet(input_tensor=inputs, input_shape=input_shape,
                           include_top=False, weights='imagenet',
                           alpha=depth_multiplier)  # type: keras.Model
    embedds = compose(
        kl.Flatten(),
        kl.Dense(2048),
        kl.AlphaDropout(0.2),
        kl.LeakyReLU(),
        kl.Dense(512),
        kl.AlphaDropout(0.2),
        kl.Dense(embedding_size),
    )(base_model.output)

    if 'softmax' in loss:
        if loss in ['amsoftmax', 'asoftmax']:
            # normalize Classification vector len = 1
            embedds = kl.Lambda(lambda x: tf.math.l2_normalize(x, 1))(embedds)
            outputs = kl.Dense(class_num, use_bias=False,
                               # normalize Classification Matrix len = 1
                               # f·W = (f·W)/(‖f‖×‖W‖) = (f·W)/(1×1) = cos(θ)
                               kernel_constraint=k.constraints.unit_norm())(embedds)
        elif loss in ['softmax']:
            outputs = kl.Dense(class_num, use_bias=False)(embedds)

        infer_model = k.Model(inputs, embedds)  # encoder to infer
        train_model = k.Model(inputs, outputs)    # full model to train

    elif 'triplet' in loss:

        infer_model = k.Model(inputs, embedds)  # encoder to infer
        in_a = k.Input(input_shape)
        in_p = k.Input(input_shape)
        in_n = k.Input(input_shape)
        embedd_a = infer_model(in_a)
        embedd_p = infer_model(in_p)
        embedd_n = infer_model(in_n)
        embedd_merge = kl.Concatenate()([embedd_a, embedd_p, embedd_n])
        # full model to train
        train_model = k.Model([in_a, in_p, in_n], embedd_merge)

    input_a = k.Input(input_shape)
    input_b = k.Input(input_shape)
    val_model = k.Model([input_a, input_b], [infer_model(input_a), infer_model(input_b)])

    return infer_model, val_model, train_model
