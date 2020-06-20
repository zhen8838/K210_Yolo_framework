import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as kl
from models.keras_mobilenet_v2 import MobileNetV2
from models.keras_mobilenet import MobileNet
from models.darknet import darknet_body, DarknetConv2D, DarknetConv2D_BN_Leaky, compose, resblock_body
from models.ultralffdnet import UltraLightFastGenericFaceBaseNet, SeperableConv2d
from models.retinanet import SSH, Conv2D_BN_Relu
from toolz import pipe
from typing import List, Tuple
import numpy as np


def yolo_mbv1_k210(input_shape: list, anchor_num: int, class_num: int,
                   alpha: float) -> [k.Model, k.Model]:
  inputs = k.Input(input_shape)
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights=None,
      alpha=alpha)  # type: k.Model

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
      DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(
          x2)

  x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), k.layers.UpSampling2D(2))(x2)

  y2 = compose(k.layers.Concatenate(), DarknetConv2D_BN_Leaky(filters, (3, 3)),
               DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

  y1_reshape = kl.Activation('linear', name='l1')(y1)
  y2_reshape = kl.Activation('linear', name='l2')(y2)

  yolo_model = k.Model(inputs, [y1, y2])
  yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

  return yolo_model, yolo_model_warpper


def yolo_mbv2_k210(input_shape: list, anchor_num: int, class_num: int,
                   alpha: float) -> [k.Model, k.Model]:
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
      DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))(
          x2)
  x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), kl.UpSampling2D(2))(x2)
  y2 = compose(kl.Concatenate(), DarknetConv2D_BN_Leaky(filters, (3, 3)),
               DarknetConv2D(anchor_num * (class_num + 5), (1, 1)))([x2, x1])

  y1_reshape = kl.Activation('linear', name='l1')(y1)
  y2_reshape = kl.Activation('linear', name='l2')(y2)

  yolo_model = k.Model(inputs=input_tensor, outputs=[y1, y2])
  yolo_model_warpper = k.Model(
      inputs=input_tensor, outputs=[y1_reshape, y2_reshape])

  return yolo_model, yolo_model_warpper


def yolo2_mbv1_k210(input_shape: list, anchor_num: int, class_num: int,
                    alpha: float) -> [k.Model, k.Model]:
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
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights=None,
      alpha=alpha)  # type: keras.Model

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
      DarknetConv2D_BN_Leaky(filters_2, (3, 3)))(
          x)

  y = DarknetConv2D(anchor_num * (class_num + 5), (1, 1))(y)

  y_reshape = kl.Activation('linear', name='l1')(y)

  yolo_model = k.Model(inputs, [y])
  yolo_model_warpper = k.Model(inputs, [y_reshape])

  return yolo_model, yolo_model_warpper


def yolov2algin_mbv1_k210(input_shape: list, anchor_num: int, class_num: int,
                          landmark_num: int, alpha: float) -> [k.Model, k.Model]:
  inputs = k.Input(input_shape)
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights=None,
      alpha=alpha)  # type: k.Model

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
      k.layers.Concatenate(), DarknetConv2D_BN_Leaky(filter_num, (3, 3)),
      DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num),
                    (1, 1)))([x1, x2])  # [7,10,48]

  y_reshape = kl.Reshape((int(input_shape[0] / 32), int(
      input_shape[1] / 32), anchor_num, 5 + landmark_num * 2 + class_num),
      name='l1')(
      y)

  yolo_model = k.Model(inputs, [y])
  yolo_model_warpper = k.Model(inputs, [y_reshape])

  return yolo_model, yolo_model_warpper


def yoloalgin_mbv1_k210(input_shape: list, anchor_num: int, class_num: int,
                        landmark_num: int, alpha: float) -> [k.Model, k.Model]:
  inputs = k.Input(input_shape)
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights=None,
      alpha=alpha)  # type: k.Model

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
      DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num), (1, 1)))(
          x2)

  x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), k.layers.UpSampling2D(2))(x2)

  y2 = compose(
      k.layers.Concatenate(), DarknetConv2D_BN_Leaky(filters, (3, 3)),
      DarknetConv2D(anchor_num * (5 + landmark_num * 2 + class_num),
                    (1, 1)))([x2, x1])

  y1_reshape = kl.Activation('linear', name='l1')(y1)
  y2_reshape = kl.Activation('linear', name='l2')(y2)

  yolo_model = k.Model(inputs, [y1, y2])
  yolo_model_warpper = k.Model(inputs, [y1_reshape, y2_reshape])

  return yolo_model, yolo_model_warpper


def pfld_k210(input_shape: list, landmark_num: int, alpha=1.,
              weight_decay=5e-5) -> [k.Model, k.Model]:
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
  bn_kwargs = {'momentum': 0.995, 'epsilon': 0.001}
  conv_kwargs = {
      'kernel_initializer': k.initializers.TruncatedNormal(stddev=0.01),
      'bias_initializer': k.initializers.Zeros(),
      'kernel_regularizer': k.regularizers.l2(weight_decay),
      'bias_regularizer': k.regularizers.l2(weight_decay),
      'padding': 'same'
  }

  depthwise_kwargs = {
      'depthwise_initializer': k.initializers.TruncatedNormal(stddev=0.01),
      'bias_initializer': k.initializers.Zeros(),
      'depthwise_regularizer': k.regularizers.l2(weight_decay),
      'bias_regularizer': k.regularizers.l2(weight_decay),
      'padding': 'same'
  }

  conv_kwargs_copy = conv_kwargs.copy()
  depthwise_kwargs_copy = depthwise_kwargs.copy()
  conv_kwargs_copy.pop('padding')
  depthwise_kwargs_copy.pop('padding')

  # 112*112*3
  inputs = k.Input(input_shape)
  conv1 = pipe(
      inputs, *[
          kl.ZeroPadding2D(),
          kl.Conv2D(int(64 * alpha), 3, 2, **conv_kwargs_copy),
          kl.BatchNormalization(**bn_kwargs),
          kl.ReLU(6)
      ])

  # 56*56*64
  conv2 = pipe(
      conv1, *[
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6)
      ])
  # 28*28*64
  conv3_1 = pipe(
      conv2, *[
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

  conv3_2 = pipe(
      conv3_1, *[
          kl.Conv2D(128, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block3_2 = kl.Add()([conv3_1, conv3_2])

  conv3_3 = pipe(
      block3_2, *[
          kl.Conv2D(128, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block3_3 = kl.Add()([block3_2, conv3_3])

  conv3_4 = pipe(
      block3_3, *[
          kl.Conv2D(128, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block3_4 = kl.Add()([block3_3, conv3_4])

  conv3_5 = pipe(
      block3_4, *[
          kl.Conv2D(128, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(64 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block3_5 = kl.Add()([block3_4, conv3_5])
  auxiliary_input = block3_5

  conv4_1 = pipe(
      block3_5, *[
          kl.ZeroPadding2D(),
          kl.Conv2D(128, 3, 2, **conv_kwargs_copy),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  conv5_1 = pipe(
      conv4_1, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  conv5_2 = pipe(
      conv5_1, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block5_2 = kl.Add()([conv5_1, conv5_2])

  conv5_3 = pipe(
      block5_2, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block5_3 = kl.Add()([block5_2, conv5_3])

  conv5_4 = pipe(
      block5_3, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block5_4 = kl.Add()([block5_3, conv5_4])

  conv5_5 = pipe(
      block5_4, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block5_5 = kl.Add()([block5_4, conv5_5])

  conv5_6 = pipe(
      block5_5, *[
          kl.Conv2D(512, 1, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.DepthwiseConv2D(3, 1, **depthwise_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Conv2D(int(128 * alpha), 1, 1, **conv_kwargs),
          kl.BatchNormalization()
      ])

  block5_6 = kl.Add()([block5_5, conv5_6])  # 14,14,128

  # 7,7,16
  conv6_1 = pipe(
      block5_6,
      *[
          kl.Conv2D(256, 1, 1, **conv_kwargs_copy),
          kl.BatchNormalization(),
          kl.ReLU(6),  # 14,14,256
          kl.ZeroPadding2D(),
          # can be modify filters
          kl.Conv2D(int(16 * alpha), 3, 2, **conv_kwargs_copy),
          kl.BatchNormalization(),
          kl.ReLU(6),  # 7,7,16
      ])

  # 7,7,32
  conv7 = pipe(
      conv6_1,
      *[
          # can be modify filters
          kl.Conv2D(int(32 * alpha), 3, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6)
      ])

  # 7,7,64
  conv8 = pipe(
      conv7,
      *[
          # can be modify filters
          kl.Conv2D(int(128 * alpha), 3, 1, **conv_kwargs),
          kl.BatchNormalization(),
          kl.ReLU(6)
      ])
  # 7,7,112
  multi_scale = kl.Concatenate()([conv6_1, conv7, conv8])
  # 7,7,4 = 196  can be modify kernel size
  landmark_pre = kl.Conv2D(landmark_num * 2 // (7 * 7), 3, 1, 'same')(multi_scale)

  pflp_infer_model = k.Model(inputs, landmark_pre)

  euler_angles_pre = pipe(
      auxiliary_input, *[
          kl.Conv2D(128, 3, 2, **conv_kwargs),
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
          kl.Dense(3)
      ])

  flatten_landmark_pre = pipe(landmark_pre,
                              *[kl.Permute(
                                  (3, 1, 2)), kl.Flatten()])

  y_pred = kl.Concatenate()([flatten_landmark_pre, euler_angles_pre])
  train_model = k.Model(inputs, y_pred)

  return pflp_infer_model, train_model


def mbv1_imgnet_k210(input_shape: list,
                     class_num: int,
                     depth_multiplier: float = 1.0,
                     weights=None):

  inputs = k.Input(input_shape)
  model = MobileNet(
      input_tensor=inputs,
      input_shape=tuple(input_shape),
      include_top=True,
      weights=weights,
      alpha=depth_multiplier,
      classes=class_num)  # type: keras.Model

  return model, model


def mbv2_imgnet_k210(input_shape: list,
                     class_num: int,
                     depth_multiplier: float = 1.0,
                     weights=None):

  inputs = k.Input(input_shape)
  model = MobileNetV2(
      input_tensor=inputs,
      input_shape=tuple(input_shape),
      include_top=True,
      weights=weights,
      alpha=depth_multiplier,
      classes=class_num)  # type: keras.Model

  return model, model


def retinafacenet_k210(input_shape: list,
                       anchor: List[Tuple],
                       branch_index=[7, 10, 12],
                       base_filters=16) -> k.Model:
  inputs = k.Input(input_shape)
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters)

  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)

  bbox_out = [
      SeperableConv2d(len(anchor[i]) * 4, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # BboxHead
  class_out = [
      SeperableConv2d(len(anchor[i]) * 2, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # ClassHead
  landm_out = [
      SeperableConv2d(len(anchor[i]) * 10, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # LandmarkHead

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


def retinafacenet_k210_v1(input_shape: list,
                          anchor: List[Tuple],
                          branch_index=[7, 10, 12],
                          base_filters=16) -> k.Model:
  """ Use SSH block """
  inputs = k.Input(input_shape)
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters)

  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)
  """ SSH block """
  features = [SSH(feat, base_filters * 4, depth=2) for feat in features]

  bbox_out = [
      kl.Conv2D(len(anchor[i]) * 4, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # BboxHead
  class_out = [
      kl.Conv2D(len(anchor[i]) * 2, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # ClassHead
  landm_out = [
      kl.Conv2D(len(anchor[i]) * 10, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # LandmarkHead

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


def retinafacenet_k210_v2(input_shape: list,
                          anchor: List[Tuple],
                          branch_index=[7, 10, 12],
                          base_filters=16) -> k.Model:
  """ Add FPN block for feature merge
    """
  inputs = k.Input(input_shape)
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters)

  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)
  """ FPN only featrues[0] and featrues[1] """
  up2 = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
  features[0] = kl.Concatenate(channel_axis)([features[0], up2])

  bbox_out = [
      SeperableConv2d(len(anchor[i]) * 4, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # BboxHead
  class_out = [
      SeperableConv2d(len(anchor[i]) * 2, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # ClassHead
  landm_out = [
      SeperableConv2d(len(anchor[i]) * 10, 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]  # LandmarkHead

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


def retinafacenet_k210_v3(input_shape: list,
                          anchor: List[Tuple],
                          branch_index=[7, 10, 12],
                          base_filters=16) -> k.Model:
  """ 1.  Add FPN block for feature merge
        2.  Use SSH block for better regerssion
    """
  inputs = k.Input(input_shape)
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters)

  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)
  """ FPN only featrues[0] and featrues[1] """
  up2 = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
  features[0] = kl.Concatenate(channel_axis)([features[0], up2])
  """ SSH block """
  features = [SSH(feat, base_filters * 4, depth=2) for feat in features]

  bbox_out = [
      kl.Conv2D(len(anchor[i]) * 4, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # BboxHead
  class_out = [
      kl.Conv2D(len(anchor[i]) * 2, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # ClassHead
  landm_out = [
      kl.Conv2D(len(anchor[i]) * 10, 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]  # LandmarkHead

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


def ullfd_k210(input_shape: list,
               class_num: int,
               anchor: List[Tuple],
               branch_index=[7, 10, 12],
               base_filters=16) -> k.Model:
  """ K210 Main memory usage: 212520 B """
  inputs = k.Input(input_shape)
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters,
                                                len(branch_index) == 4)
  assert len(anchor) == len(branch_index), 'anchor layer num must == branch num'
  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32),round(in_hw / 64)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)

  out = [
      SeperableConv2d(len(anchor[i]) * (4 + 1 + class_num), 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]
  out = [kl.Reshape((-1, (4 + 1 + class_num)))(o) for o in out]
  out = kl.Concatenate(1)(out)

  infer_model = k.Model(inputs, out)
  train_model = k.Model(inputs, out)

  return infer_model, train_model


def ullfd_k210_v1(input_shape: list,
                  class_num: int,
                  anchor: List[Tuple],
                  branch_index=[7, 10, 12],
                  base_filters=16) -> k.Model:
  """ Use SSH block """
  inputs = k.Input(input_shape)
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters,
                                                len(branch_index) == 4)
  assert len(anchor) == len(branch_index), 'anchor layer num must == branch num'
  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32),round(in_hw / 64)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)
  """ SSH block """
  features = [SSH(feat, base_filters * 4, depth=2) for feat in features]
  out = [
      kl.Conv2D(len(anchor[i]) * (4 + 1 + class_num), 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]
  out = [kl.Reshape((-1, (4 + 1 + class_num)))(o) for o in out]
  out = kl.Concatenate(1)(out)

  infer_model = k.Model(inputs, out)
  train_model = infer_model

  return infer_model, train_model


def ullfd_k210_v2(input_shape: list,
                  class_num: int,
                  anchor: List[Tuple],
                  branch_index=[7, 10, 12],
                  base_filters=16) -> k.Model:
  """ Add FPN block for feature merge
    """
  inputs = k.Input(input_shape)
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters,
                                                len(branch_index) == 4)

  assert len(anchor) == len(branch_index), 'anchor layer num must == branch num'
  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32),round(in_hw / 64)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)

  if len(branch_index) == 4:
    #  FPN in featrues[0]-featrues[1] and featrues[2]-featrues[3]
    up = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
    features[0] = kl.Concatenate(channel_axis)([features[0], up])
    up1 = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[3]))
    features[2] = kl.Concatenate(channel_axis)([features[2], up1])
  else:
    #  FPN only featrues[0] and featrues[1]
    up = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
    features[0] = kl.Concatenate(channel_axis)([features[0], up])

  out = [
      SeperableConv2d(len(anchor[i]) * (4 + 1 + class_num), 3, padding='same')(feat)
      for (i, feat) in enumerate(features)
  ]
  out = [kl.Reshape((-1, (4 + 1 + class_num)))(o) for o in out]
  out = kl.Concatenate(1)(out)

  infer_model = k.Model(inputs, out)
  train_model = infer_model

  return infer_model, train_model


def ullfd_k210_v3(input_shape: list,
                  class_num: int,
                  anchor: List[Tuple],
                  branch_index=[7, 10, 12],
                  base_filters=16) -> k.Model:
  """ 1.  Add FPN block for feature merge
        2.  Use SSH block for better regerssion
    """
  inputs = k.Input(input_shape)
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  base_model = UltraLightFastGenericFaceBaseNet(inputs, base_filters,
                                                len(branch_index) == 4)

  assert len(anchor) == len(branch_index), 'anchor layer num must == branch num'
  features = []
  for index in branch_index:
    # round(in_hw / 8),round(in_hw / 16),round(in_hw / 32),round(in_hw / 64)
    features.append(base_model.get_layer(f'conv_dw_{index}_relu_2').output)

  if len(branch_index) == 4:
    #  FPN in featrues[0]-featrues[1] and featrues[2]-featrues[3]
    up = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
    features[0] = kl.Concatenate(channel_axis)([features[0], up])
    up1 = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[3]))
    features[2] = kl.Concatenate(channel_axis)([features[2], up1])
  else:
    #  FPN only featrues[0] and featrues[1]
    up = kl.UpSampling2D()(Conv2D_BN_Relu(base_filters, 1, 1)(features[1]))
    features[0] = kl.Concatenate(channel_axis)([features[0], up])
  """ SSH block """
  features = [SSH(feat, base_filters * 4, depth=2) for feat in features]
  out = [
      kl.Conv2D(len(anchor[i]) * (4 + 1 + class_num), 1, 1)(feat)
      for (i, feat) in enumerate(features)
  ]
  out = [kl.Reshape((-1, (4 + 1 + class_num)))(o) for o in out]
  out = kl.Concatenate(1)(out)

  infer_model = k.Model(inputs, out)
  train_model = infer_model

  return infer_model, train_model


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
  """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  pointwise_conv_filters = int(pointwise_conv_filters * alpha)

  if strides == (1, 1):
    x = inputs
  else:
    x = kl.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
  x = kl.DepthwiseConv2D((3, 3),
                         padding='same' if strides == (1, 1) else 'valid',
                         depth_multiplier=depth_multiplier,
                         strides=strides,
                         use_bias=False,
                         name='conv_dw_%d' % block_id)(
                             x)
  x = kl.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
  x = kl.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

  x = kl.Conv2D(
      pointwise_conv_filters, (1, 1),
      padding='same',
      use_bias=False,
      strides=(1, 1),
      name='conv_pw_%d' % block_id)(
          x)
  x = kl.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
  return kl.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


def mbv1_facerec_k210(input_shape: list,
                      class_num: int,
                      embedding_size: int,
                      depth_multiplier: float = 1.0,
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
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights='imagenet',
      alpha=depth_multiplier)  # type: k.Model

  finally_shape = base_model.outputs[0].shape.as_list()[1:3]
  assert embedding_size % np.prod(
      finally_shape
  ) == 0, f'embedding_size must be a multiple of {finally_shape[0]}*{finally_shape[1]}'
  finally_filters = embedding_size // np.prod(finally_shape)

  x = _depthwise_conv_block(
      base_model.outputs[0], 512, depth_multiplier, block_id=14)
  x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=15)
  x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=16)
  embedds = compose(
      kl.DepthwiseConv2D((3, 3),
                         padding='same',
                         depth_multiplier=1,
                         strides=(1, 1),
                         use_bias=False), kl.BatchNormalization(), kl.ReLU(6.),
      kl.Conv2D(
          finally_filters, (3, 3), padding='same', use_bias=False,
          strides=(1, 1)), kl.Flatten())(
              x)

  if 'softmax' in loss:
    if loss in ['amsoftmax', 'asoftmax']:
      # normalize Classification vector len = 1
      embedds = kl.Lambda(lambda x: tf.math.l2_normalize(x, 1))(embedds)
      outputs = kl.Dense(
          class_num,
          use_bias=False,
          # normalize Classification Matrix len = 1
          # f·W = (f·W)/(‖f‖×‖W‖) = (f·W)/(1×1) = cos(θ)
          kernel_constraint=k.constraints.unit_norm())(
              embedds)
    elif loss in ['softmax']:
      outputs = kl.Dense(class_num, use_bias=False)(embedds)
    infer_model = k.Model(inputs, embedds)  # encoder to infer
    train_model = k.Model(inputs, outputs)  # full model to train

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
  val_model = k.Model(
      [input_a, input_b],
      [infer_model(input_a), infer_model(input_b)])

  return infer_model, val_model, train_model


def mbv1_facerec_k210_eager(input_shape: list,
                            class_num: int,
                            embedding_size: int,
                            depth_multiplier: float = 1.0,
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
  loss_list = ['softmax', 'asoftmax', 'amsoftmax', 'triplet', 'circleloss']
  if loss not in loss_list:
    raise ValueError(f"loss not valid! must in {' '.join(loss_list)}")

  inputs = k.Input(input_shape)
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights='imagenet',
      alpha=depth_multiplier)  # type: k.Model

  finally_shape = base_model.outputs[0].shape.as_list()[1:3]
  assert embedding_size % np.prod(
      finally_shape
  ) == 0, f'embedding_size must be a multiple of {finally_shape[0]}*{finally_shape[1]}'
  finally_filters = embedding_size // np.prod(finally_shape)

  x = _depthwise_conv_block(
      base_model.outputs[0], 512, depth_multiplier, block_id=14)
  x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=15)
  x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=16)
  embedds = compose(
      kl.DepthwiseConv2D((3, 3),
                         padding='same',
                         depth_multiplier=1,
                         strides=(1, 1),
                         use_bias=False), kl.BatchNormalization(), kl.ReLU(6.),
      kl.Conv2D(
          finally_filters, (3, 3), padding='same', use_bias=False,
          strides=(1, 1)), kl.Flatten())(
              x)

  if 'softmax' in loss:
    if loss in ['amsoftmax', 'asoftmax', 'circleloss']:
      # normalize Classification vector len = 1
      embedds = kl.Lambda(lambda x: tf.math.l2_normalize(x, 1))(embedds)
      outputs = kl.Dense(
          class_num,
          use_bias=False,
          # normalize Classification Matrix len = 1
          # f·W = (f·W)/(‖f‖×‖W‖) = (f·W)/(1×1) = cos(θ)
          kernel_constraint=k.constraints.unit_norm())(
              embedds)
    elif loss in ['softmax']:
      outputs = kl.Dense(class_num, use_bias=False)(embedds)
    infer_model = k.Model(inputs, embedds)  # encoder to infer
    train_model = k.Model(inputs, outputs)  # full model to train
    val_model = train_model

  elif 'triplet' in loss:
    infer_model = k.Model(inputs, embedds)  # encoder to infer
    val_model = train_model = infer_model

  return infer_model, val_model, train_model
