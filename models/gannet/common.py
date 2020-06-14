import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from tensorflow.python.ops import nn
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import conv_utils, tf_utils
from tensorflow.python.ops import gen_math_ops, sparse_ops, standard_ops, state_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.eager import context


class InstanceNormalization(kl.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

  def get_config(self):
    config = {
        'epsilon': self.epsilon,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SpectralNormalization(kl.Wrapper):
  def __init__(self, layer, iteration=1, eps=1e-12, **kwargs):
    self.iteration = iteration
    if not isinstance(layer, kl.Layer):
      raise ValueError(
          'Please initialize `TimeDistributed` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))
    super().__init__(layer, **kwargs)

  def build(self, input_shape):
    self.layer.build(input_shape)
    self.kernel = self.layer.kernel
    self.w_shape = self.layer.kernel.shape.as_list()

    self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                             initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                             trainable=False,
                             name='sn_u',
                             dtype=tf.float32)

    super().build()

  def call(self, inputs):
    self.update_weights()
    output = self.layer(inputs)
    return output

  def update_weights(self):
    w = tf.reshape(self.layer.kernel, [-1, self.w_shape[-1]])
    u_hat = self.u
    v_hat = None

    for _ in range(self.iteration):
      v_ = tf.matmul(u_hat, tf.transpose(w))
      v_hat = tf.nn.l2_normalize(v_)

      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([self.u.assign(u_hat)]):
      w_norm = w / sigma
      w_norm = tf.reshape(w_norm, self.w_shape)
      self.layer.kernel.assign(w_norm)


class ReflectionPadding2D(kl.ZeroPadding2D):

  @staticmethod
  def reflect_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
      data_format = k.backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
      raise ValueError('Unknown data_format: ' + str(data_format))

    if data_format == 'channels_first':
      pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
    else:
      pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
    return tf.pad(x, pattern, mode='REFLECT')

  def call(self, inputs):
    return self.reflect_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)


class ConstraintMinMax(k.constraints.Constraint):
  """MinMax weight constraint.

  Constrains the weights incident to each hidden unit
  to have the norm between a lower bound and an upper bound.

  Arguments:
      min_value: the minimum norm for the incoming weights.
      max_value: the maximum norm for the incoming weights.
  """

  def __init__(self, min_value=0.0, max_value=1.0):
    self.min_value = min_value
    self.max_value = max_value
    assert self.min_value < self.max_value

  def __call__(self, w):
    desired = tf.clip_by_value(w, self.min_value, self.max_value)
    return desired

  def get_config(self):
    return {'min_value': self.min_value,
            'max_value': self.max_value}
