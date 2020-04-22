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
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers.convolutional import Conv


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
    normalized = (x-mean) * inv
    return self.scale * normalized + self.offset

  def get_config(self):
    config = {
        'epsilon': self.epsilon,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ConvSpectralNormal(Conv):

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    try:
      # Disable variable partitioning when creating the variable
      if hasattr(self, '_scope') and self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None

      self.u = self.add_weight(
          name='sn_u',
          shape=(1, tf.reduce_prod(kernel_shape[:-1])),
          dtype=self.dtype,
          initializer=tf.keras.initializers.ones,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)

    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_channel})

    self._build_conv_op_input_shape = input_shape
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    self._conv_op_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)
    self.built = True

  def call(self, inputs, training=None):
    # Check if the input_shape in call() is different from that in build().
    # If they are different, recreate the _convolution_op to avoid the stateful
    # behavior.
    if training is None:
      training = K.learning_phase()

    call_input_shape = inputs.get_shape()
    recreate_conv_op = (
        call_input_shape[1:] != self._build_conv_op_input_shape[1:])

    if recreate_conv_op:
      self._convolution_op = nn_ops.Convolution(
          call_input_shape,
          filter_shape=self.kernel.shape,
          dilation_rate=self.dilation_rate,
          strides=self.strides,
          padding=self._padding_op,
          data_format=self._conv_op_data_format)

    # Apply causal padding to inputs for Conv1D.
    if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())

    # Update SpectralNormalization variable
    u, v, w = self.calc_u(self.kernel)

    def u_update():
      # TODO u_update maybe need `training control`
      return tf_utils.smart_cond(training, lambda: self._assign_new_value(
          self.u, u), lambda: array_ops.identity(u))

    # NOTE add update must in call function scope
    self.add_update(u_update)

    sigma = self.calc_sigma(u, v, w)
    new_kernel = tf_utils.smart_cond(
        training, lambda: self.kernel / sigma, lambda: self.kernel)

    outputs = self._convolution_op(inputs, new_kernel)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def calc_u(self, w):
    w = K.reshape(w, (-1, w.shape[-1]))
    v = K.l2_normalize(K.dot(self.u, w))
    u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return u, v, w

  def calc_sigma(self, u, v, w):
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

  def _assign_new_value(self, variable, value):
    with K.name_scope('AssignNewValue') as scope:
      with ops.colocate_with(variable):
        return state_ops.assign(variable, value, name=scope)


class Conv2DSpectralNormal(ConvSpectralNormal):

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=tf.keras.activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.get(kernel_initializer),
        bias_initializer=tf.keras.initializers.get(bias_initializer),
        kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
        bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        kernel_constraint=tf.keras.constraints.get(kernel_constraint),
        bias_constraint=tf.keras.constraints.get(bias_constraint),
        **kwargs)


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