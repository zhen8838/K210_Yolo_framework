import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from typing import List, Callable
from tensorflow.keras.applications import MobileNet, MobileNetV2
from models.darknet import compose


def mbv1_facerec(input_shape: list,
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
  loss_list = ['softmax', 'asoftmax', 'amsoftmax', 'circlesoftmax', 'triplet']
  if loss not in loss_list:
    raise ValueError(f"loss not valid! must in {' '.join(loss_list)}")

  inputs = k.Input(input_shape)
  base_model = MobileNet(
      input_tensor=inputs,
      input_shape=input_shape,
      include_top=False,
      weights='imagenet',
      alpha=depth_multiplier)  # type: keras.Model

  nn = base_model.outputs[0]
  """ GDC """
  nn = kl.Conv2D(512, 1, use_bias=False)(nn)
  nn = kl.BatchNormalization()(nn)
  nn = kl.DepthwiseConv2D(nn.shape[1], depth_multiplier=1, use_bias=False)(nn)
  nn = kl.BatchNormalization()(nn)
  nn = kl.Dropout(0.1)(nn)
  nn = kl.Flatten()(nn)
  nn = kl.Dense(embedding_size, activation=None, use_bias=False,
                kernel_initializer="glorot_normal")(nn)
  embedds = kl.BatchNormalization(name="embedding")(nn)

  if 'softmax' in loss:
    if loss in ['amsoftmax', 'asoftmax', 'circlesoftmax']:
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


def FMobileFaceNet_eager(input_shape: list,
                         class_num: int,
                         embedding_size: int,
                         blocks: list,
                         loss: str,
                         act_type='prelu',
                         bn_mom=0.9):

  def Act(act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
      return kl.PReLU(name=name)
    elif act_type == 'leakyrelu':
      return kl.LeakyReLU(name=name)
    else:
      return kl.Activation(act_type, name=name)

  def Conv(data: tf.Tensor,
           num_filter=1,
           kernel=(1, 1),
           stride=(1, 1),
           pad='valid',
           num_group=1,
           name=None,
           suffix='') -> tf.Tensor:

    func_list = []
    # if stride == (2, 2) and pad == 'same':
    #   func_list.append(kl.ZeroPadding2D(((1, 1), (1, 1))))
    #   pad == 'valid'

    if num_group == data.shape[-1]:
      func_list.append(
          kl.DepthwiseConv2D(
              kernel, stride, pad, use_bias=False, name=f'{name}{suffix}_conv2d'))
    else:
      func_list.append(
          kl.Conv2D(
              num_filter,
              kernel,
              stride,
              pad,
              use_bias=False,
              name=f'{name}{suffix}_conv2d'))
    return compose(
        *func_list,
        kl.BatchNormalization(
            momentum=bn_mom,
            center=False,
            scale=False,
            name=f'{name}{suffix}_batchnorm'),
        Act(act_type, name=f'{name}{suffix}_relu'))(
            data)

  def Linear(data: tf.Tensor,
             num_filter=1,
             kernel=(1, 1),
             stride=(1, 1),
             pad='valid',
             num_group=1,
             name=None,
             suffix='') -> tf.Tensor:

    func_list = []
    # if stride == (2, 2) and pad == 'same':
    #   func_list.append(kl.ZeroPadding2D(((1, 1), (1, 1))))
    #   pad == 'valid'
    if num_group == data.shape[-1]:
      func_list.append(
          kl.DepthwiseConv2D(
              kernel, stride, pad, use_bias=False, name=f'{name}{suffix}_conv2d'))
    else:
      func_list.append(
          kl.Conv2D(
              num_filter,
              kernel,
              stride,
              pad,
              use_bias=False,
              name=f'{name}{suffix}_conv2d'))
    return compose(
        *func_list,
        kl.BatchNormalization(
            momentum=bn_mom,
            center=False,
            scale=False,
            name=f'{name}{suffix}_batchnorm'))(
                data)

  def ConvOnly(data,
               num_filter=1,
               kernel=(1, 1),
               stride=(1, 1),
               pad='valid',
               num_group=1,
               name=None,
               suffix=''):
    func_list = []
    if stride == (2, 2) and pad == 'same':
      func_list.append(kl.ZeroPadding2D(((1, 1), (1, 1))))
      pad == 'valid'
    if num_group == data.shape[1]:
      func_list.append(
          kl.DepthwiseConv2D(
              kernel, stride, pad, use_bias=False, name=f'{name}{suffix}_conv2d'))
    else:
      func_list.append(
          kl.Conv2D(
              num_filter,
              kernel,
              stride,
              pad,
              use_bias=False,
              name=f'{name}{suffix}_conv2d'))
    return compose(*func_list)(data)

  def DResidual(data,
                num_out=1,
                kernel=(3, 3),
                stride=(2, 2),
                pad='same',
                num_group=1,
                name=None,
                suffix=''):
    conv = Conv(
        data=data,
        num_filter=num_group,
        kernel=(1, 1),
        pad='valid',
        stride=(1, 1),
        name='%s%s_conv_sep' % (name, suffix))
    conv_dw = Conv(
        data=conv,
        num_filter=num_group,
        num_group=num_group,
        kernel=kernel,
        pad=pad,
        stride=stride,
        name='%s%s_conv_dw' % (name, suffix))
    proj = Linear(
        data=conv_dw,
        num_filter=num_out,
        kernel=(1, 1),
        pad='valid',
        stride=(1, 1),
        name='%s%s_conv_proj' % (name, suffix))
    return proj

  def Residual(data,
               num_block=1,
               num_out=1,
               kernel=(3, 3),
               stride=(1, 1),
               pad='same',
               num_group=1,
               name=None,
               suffix=''):
    identity = data
    for i in range(num_block):
      shortcut = identity
      conv = DResidual(
          data=identity,
          num_out=num_out,
          kernel=kernel,
          stride=stride,
          pad=pad,
          num_group=num_group,
          name='%s%s_block' % (name, suffix),
          suffix='%d' % i)
      identity = kl.Add()([conv, shortcut])
    return identity

  data = k.Input(input_shape)
  conv_1 = Conv(
      data,
      num_filter=64,
      kernel=(3, 3),
      pad='same',
      stride=(2, 2),
      name="conv_1")
  if blocks[0] == 1:
    conv_2_dw = Conv(
        conv_1,
        num_group=64,
        num_filter=64,
        kernel=(3, 3),
        pad='same',
        stride=(1, 1),
        name="conv_2_dw")
  else:
    conv_2_dw = Residual(
        conv_1,
        num_block=blocks[0],
        num_out=64,
        kernel=(3, 3),
        stride=(1, 1),
        pad='same',
        num_group=64,
        name="res_2")
  conv_23 = DResidual(
      conv_2_dw,
      num_out=64,
      kernel=(3, 3),
      stride=(2, 2),
      pad='same',
      num_group=128,
      name="dconv_23")
  conv_3 = Residual(
      conv_23,
      num_block=blocks[1],
      num_out=64,
      kernel=(3, 3),
      stride=(1, 1),
      pad='same',
      num_group=128,
      name="res_3")
  conv_34 = DResidual(
      conv_3,
      num_out=128,
      kernel=(3, 3),
      stride=(2, 2),
      pad='same',
      num_group=256,
      name="dconv_34")
  conv_4 = Residual(
      conv_34,
      num_block=blocks[2],
      num_out=128,
      kernel=(3, 3),
      stride=(1, 1),
      pad='same',
      num_group=256,
      name="res_4")
  conv_45 = DResidual(
      conv_4,
      num_out=128,
      kernel=(3, 3),
      stride=(2, 2),
      pad='same',
      num_group=512,
      name="dconv_45")
  conv_5 = Residual(
      conv_45,
      num_block=blocks[3],
      num_out=128,
      kernel=(3, 3),
      stride=(1, 1),
      pad='same',
      num_group=256,
      name="res_5")
  conv_6_sep = Conv(
      conv_5,
      num_filter=512,
      kernel=(1, 1),
      pad='valid',
      stride=(1, 1),
      name="conv_6sep")

  conv_6_dw = Linear(
      conv_6_sep,
      num_filter=512,
      num_group=512,
      kernel=(7, 7),
      pad='valid',
      stride=(1, 1),
      name="conv_6dw7_7")
  conv_6_dw = kl.Reshape((512,))(conv_6_dw)
  conv_6_f = kl.Dense(embedding_size, name='pre_fc1')(conv_6_dw)
  embedds = kl.BatchNormalization(
      scale=False, epsilon=2e-5, momentum=bn_mom, name='fc1')(
          conv_6_f)

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
    infer_model = k.Model(data, embedds)  # encoder to infer
    train_model = k.Model(data, outputs)  # full model to train
    val_model = train_model

  elif 'triplet' in loss:
    infer_model = k.Model(data, embedds)  # encoder to infer
    val_model = train_model = infer_model

  return infer_model, val_model, train_model
