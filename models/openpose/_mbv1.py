import tensorflow as tf
from typing import List, Tuple
from models.darknet import compose

k = tf.keras
kl = tf.keras.layers


class depthwise_conv(object):
  def __init__(self, kernel_size, pointwise_conv_filters, use_relu=True, strides=(1, 1), name=None):
    super().__init__()
    self.channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
    self.pointwise_conv_filters = pointwise_conv_filters
    self.name = name
    self.strides = strides
    self.kernel_size = kernel_size
    self.use_relu = use_relu

  def __call__(self, inputs):
    if self.strides == (1, 1):
      x = inputs
    else:
      x = kl.ZeroPadding2D(((0, 1), (0, 1)),
                           name=self.name + '_padding')(inputs)

    x = kl.DepthwiseConv2D(self.kernel_size,
                           padding='same' if self.strides == (1, 1) else 'valid',
                           depth_multiplier=1,
                           strides=self.strides,
                           use_bias=False,
                           name=self.name + '_depthwise')(x)
    # x = kl.BatchNormalization(
    #     axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    # x = kl.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = kl.Conv2D(self.pointwise_conv_filters, (1, 1),
                  padding='same',
                  use_bias=True,
                  strides=(1, 1),
                  name=self.name + '_pointwise')(x)
    x = kl.BatchNormalization(axis=self.channel_axis, momentum=0.999,
                              name=self.name + '_bn')(x)
    if self.use_relu:
      return kl.ReLU(6., name=self.name + '_relu')(x)
    else:
      return x


def MobileNetV1OpenPose(input_shape: List[int], alpha: float = 0.75,
                        alpha2: float = None, num_refine: int = 4
                        ) -> Tuple[k.Model, k.Model, k.Model]:
  """MobileNetV1OpenPose

  Args:
      `input_shape` (List[int]): [height,width,channls]

      `alpha` (float, optional): Defaults to 0.75.

      `alpha2` (float, optional): external branch alpha Defaults to None.

      `num_refine` (int, optional): external branch number. Defaults to 4.


  Returns:
      Tuple[k.Model, k.Model, k.Model]: `infer_model`, `val_model`, `train_model`
      NOTE train model outputs: [`l1_vectmap`,`l1_heatmap`,`l2_vectmap`,`l2_heatmap`, ...]
      NOTE infer model outputs: [`l5_vectmap`,`l5_heatmap`]
  """
  channel_axis = 1 if k.backend.image_data_format() == 'channels_first' else -1
  inputs = k.Input(input_shape)
  basemodel: k.Model = k.applications.MobileNet(input_tensor=inputs, alpha=alpha, include_top=False)

  # basemodel.summary()
  nodes = {}
  nodes['conv_1'] = basemodel.get_layer('conv_pw_1_relu').output
  nodes['conv_1_pool'] = kl.MaxPool2D(2, 2, padding='same')(nodes['conv_1'])

  nodes['conv_5'] = basemodel.get_layer('conv_pw_5_relu').output
  nodes['conv_5_upsample'] = kl.UpSampling2D(interpolation='bilinear')(nodes['conv_5'])

  nodes['conv_8'] = basemodel.get_layer('conv_pw_8_relu').output

  nodes['feature_lv'] = kl.Concatenate(channel_axis)([
      nodes['conv_1_pool'],
      basemodel.get_layer('conv_pw_3_relu').output,
      nodes['conv_5_upsample']])

  def depth2(d): return max(int(d * (alpha2 if alpha2 else alpha)), 8)

  with tf.name_scope('Openpose'):
    prefix = 'MConv_Stage1'
    nodes[prefix + '_L1_5'] = compose(
        depthwise_conv(3, depth2(128), name=prefix + '_L1_1'),
        depthwise_conv(3, depth2(128), name=prefix + '_L1_2'),
        depthwise_conv(3, depth2(128), name=prefix + '_L1_3'),
        depthwise_conv(1, depth2(512), name=prefix + '_L1_4'),
        depthwise_conv(1, 38, use_relu=False, name=prefix + '_L1_5'))(nodes['feature_lv'])

    nodes[prefix + '_L2_5'] = compose(
        depthwise_conv(3, depth2(128), name=prefix + '_L2_1'),
        depthwise_conv(3, depth2(128), name=prefix + '_L2_2'),
        depthwise_conv(3, depth2(128), name=prefix + '_L2_3'),
        depthwise_conv(1, depth2(512), name=prefix + '_L2_4'),
        depthwise_conv(1, 19, use_relu=False, name=prefix + '_L2_5'))(nodes['feature_lv'])

    for stage_id in range(num_refine):
      prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
      prefix = 'MConv_Stage%d' % (stage_id + 2)

      nodes[prefix + '_concat'] = kl.Concatenate(channel_axis, name=prefix + '_concat')(
          [nodes[prefix_prev + '_L1_5'],
           nodes[prefix_prev + '_L2_5'],
           nodes['feature_lv']])

      nodes[prefix + '_L1_5'] = compose(
          depthwise_conv(7, depth2(128), name=prefix + '_L1_1'),
          depthwise_conv(7, depth2(128), name=prefix + '_L1_2'),
          depthwise_conv(7, depth2(128), name=prefix + '_L1_3'),
          depthwise_conv(1, depth2(128), name=prefix + '_L1_4'),
          depthwise_conv(1, 38, use_relu=False, name=prefix + '_L1_5')
      )(nodes[prefix + '_concat'])

      nodes[prefix + '_L2_5'] = compose(
          depthwise_conv(7, depth2(128), name=prefix + '_L2_1'),
          depthwise_conv(7, depth2(128), name=prefix + '_L2_2'),
          depthwise_conv(7, depth2(128), name=prefix + '_L2_3'),
          depthwise_conv(1, depth2(128), name=prefix + '_L2_4'),
          depthwise_conv(1, 19, use_relu=False, name=prefix + '_L2_5')
      )(nodes[prefix + '_concat'])

    l1s = []
    l2s = []

    for key in sorted(nodes.keys()):
      if '_L1_5' in key:
        l1s.append(nodes[key])
      if '_L2_5' in key:
        l2s.append(nodes[key])

    train_model = k.Model(inputs, list(zip(l1s, l2s)))

    infer_model = k.Model(inputs, [nodes[f'MConv_Stage{num_refine+1}_L1_5'],
                                   nodes[f'MConv_Stage{num_refine+1}_L2_5']])
    val_model = infer_model

    return infer_model, val_model, train_model
