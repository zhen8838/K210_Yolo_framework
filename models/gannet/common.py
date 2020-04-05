import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers


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


class SpectralNormalization(object):
  """
    UseAge:
    
    `x = SpectralNormalization(Dense(100, activation='relu'))(x)`
    
  """

  def __init__(self, layer):
    self.layer: kl.Layer = layer

  def spectral_norm(self, w, r=1):
    w_shape = K.int_shape(w)
    in_dim = tf.reduce_prod(w_shape[:-1])
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
      v = K.l2_normalize(K.dot(u, w))
      u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

  def spectral_normalization(self, w):
    return w / self.spectral_norm(w)

  def __call__(self, inputs):
    with K.name_scope(self.layer.name):
      if not self.layer.built:
        input_shape = K.int_shape(inputs)
        self.layer.build(input_shape)
        self.layer.built = True
        if self.layer._initial_weights is not None:
          self.layer.set_weights(self.layer._initial_weights)
    if not hasattr(self.layer, 'spectral_normalization'):
      if hasattr(self.layer, 'kernel'):
        self.layer.kernel = self.spectral_normalization(self.layer.kernel)
      if hasattr(self.layer, 'gamma'):
        self.layer.gamma = self.spectral_normalization(self.layer.gamma)
      self.layer.spectral_normalization = True
    return self.layer(inputs)
