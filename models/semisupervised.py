import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from models.darknet import compose


class ReSampling(kl.Layer):

  def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
    super().__init__(
        trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

  def call(self, inputs, **kwargs):
    z_μ, z_log_σ = inputs
    u = tf.random.normal(shape=tf.shape(z_μ))
    return z_μ + tf.exp(z_log_σ / 2) * u


class ReShuffle(kl.Layer):

  def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
    """ ReShuffle layer
        
        input = [None, 32, 128]
        return (equal label, output)
        
        NOTE: equal label = (shuffle_idx != orign_idx) = [None, 1, 1]
        output = shuffle(input)
        
    """
    super().__init__(
        trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

  def call(self, inputs, **kwargs):
    x_shape = tf.shape(inputs)
    batch = x_shape[0]
    orig_idxs = tf.range(batch)
    shuf_idxs = tf.random.shuffle(orig_idxs)
    eq_label = tf.cast(tf.not_equal(orig_idxs, shuf_idxs), tf.float32)
    eq_label = tf.reshape(
        eq_label,
        tf.concat([x_shape[0:1], tf.ones_like(x_shape[1:])], 0))
    return eq_label, tf.gather(inputs, shuf_idxs)


def cifar_infomax_ssl_v1(
    input_shape,
    nclasses,
    softmax=False,
    z_dim=256,  # 隐变量维度
    weight_decay=0.0005) -> [k.Model, k.Model, k.Model]:
  """ infomax with ssl first experiment
    1.  only add classify head in last layer  

  """
  conv_args = dict(kernel_regularizer=k.regularizers.l2(weight_decay))
  dense_args = dict(
      kernel_initializer=k.initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=k.regularizers.l2(weight_decay))
  bn_args = dict(axis=-1, momentum=0.999)

  inputs = k.Input(input_shape)
  encoder_stage_1 = compose(
      kl.Conv2D(z_dim // 4, kernel_size=(3, 3), padding='SAME', **conv_args),
      kl.BatchNormalization(**bn_args),
      kl.LeakyReLU(0.2),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(z_dim // 2, kernel_size=(3, 3), padding='SAME', **conv_args),
      kl.BatchNormalization(**bn_args),
      kl.LeakyReLU(0.2),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(z_dim, kernel_size=(3, 3), padding='SAME', **conv_args),
      kl.BatchNormalization(**bn_args),
      kl.LeakyReLU(0.2),
      kl.MaxPooling2D((2, 2)),
  )
  fmap = encoder_stage_1(inputs)
  encoder_stage_2 = compose(
      kl.Conv2D(z_dim, kernel_size=(3, 3), padding='SAME', **conv_args),
      kl.BatchNormalization(**bn_args),
      kl.LeakyReLU(0.2),
      kl.Conv2D(z_dim, kernel_size=(3, 3), padding='SAME', **conv_args),
      kl.BatchNormalization(**bn_args),
      kl.LeakyReLU(0.2),
      kl.GlobalMaxPooling2D(),
  )
  hiddens = encoder_stage_2(fmap)

  # add independ normal prior layer
  z_mean = kl.Dense(z_dim, name='normal_mean')(hiddens)  # 均值
  z_log_sigma = kl.Dense(z_dim, name='normal_std')(hiddens)  # 方差
  z_samples = ReSampling()([z_mean, z_log_sigma])
  # build global sample pair
  (zz_label, z_shuffle) = ReShuffle(name='shuffling_global')(z_samples)
  zz_true = kl.Concatenate()([z_samples, z_samples])  # global true pair
  zz_false = kl.Concatenate()([z_samples, z_shuffle])  # global false pair
  # build local sample pair
  (zf_label, fmap_shuffle) = ReShuffle(name='shuffling_local')(fmap)

  z_samples_map = compose(
      kl.RepeatVector(4 * 4),
      kl.Reshape((4, 4, z_dim)),
  )(
      z_samples)

  zf_true = kl.Concatenate()([z_samples_map, fmap])  # loacl true pair
  zf_false = kl.Concatenate()([z_samples_map, fmap_shuffle])  # loacl false pair

  # get global discriminate result
  globaldiscriminator = compose(
      # kl.InputLayer(input_shape=[2 * z_dim]),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(1, activation='sigmoid'),
  )

  zz_true_scores = globaldiscriminator(zz_true)  # 使用判别器判别正样本对
  zz_false_scores = globaldiscriminator(zz_false)  # 使用判别器判别负样本对

  # get local discriminate result
  localdiscriminator = compose(
      # kl.InputLayer(input_shape=[None, None, z_dim * 2]),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(z_dim, activation='relu', **dense_args),
      kl.Dense(1, activation='sigmoid'),
  )
  zf_true_scores = localdiscriminator(zf_true)
  zf_false_scores = localdiscriminator(zf_false)

  # add classify head
  # can choice hiddens or z_mean
  logits = kl.Dense(
      nclasses,
      activation='softmax' if softmax else None,
      name='logits',
      **dense_args)(
          z_mean)

  train_model = k.Model(inputs, [
      logits, z_mean, z_log_sigma, zz_true_scores, zz_false_scores, zz_label,
      zf_true_scores, zf_false_scores, zf_label
  ])

  val_model = k.Model(inputs, logits)
  infer_model = val_model
  return infer_model, val_model, train_model
