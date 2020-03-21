import tensorflow as tf
from transforms.audio import ops

NAME_TO_FUNC = {
    'Identity': tf.identity,
    'FreqMask': ops.freq_mask,
    'TimeMask': ops.time_mask,
    'FreqRescale': ops.freq_rescale,
    'TimeRescale': ops.time_rescale,
    'FreqWarping': ops.freq_warping,
    'TimeWarping': ops.time_warping,
    'Dropout': ops.mel_dropout,
    'Loudness': ops.mel_loudness,
}


def _ignore_level_to_arg(level):
  del level
  return ()


def _mask_level_to_arg(level):
  # level = [0~1]
  factor = level * 0.5
  times = 1
  return (
      factor,
      times,
  )


def _rescale_level_to_arg(level):
  # level = [0~1]
  factor = level * 0.5
  return (factor,)


def _warping_level_to_arg(level):
  # level = [0~1]
  factor = level * 0.5
  npoints = 1
  return (
      factor,
      npoints,
  )


def _dropout_level_to_arg(level):
  # level = [0~1]
  drop_prob = level * 0.3
  return (drop_prob,)


def _loudness_level_to_arg(level):
  # level = [0~1]
  factor = level * 0.4
  return (factor,)


LEVEL_TO_ARG = {
    'Identity': _ignore_level_to_arg,
    'FreqMask': _mask_level_to_arg,
    'TimeMask': _mask_level_to_arg,
    'FreqRescale': _rescale_level_to_arg,
    'TimeRescale': _rescale_level_to_arg,
    'FreqWarping': _warping_level_to_arg,
    'TimeWarping': _warping_level_to_arg,
    'Dropout': _dropout_level_to_arg,
    'Loudness': _loudness_level_to_arg,
}

AUG_OPS = [
    'Identity',
    'FreqMask',
    'TimeMask',
    'FreqRescale',
    'TimeRescale',
    'FreqWarping',
    'TimeWarping',
    'Dropout',
    'Loudness',
]


def _skip_mirrored_creator(next_creator, *args, **kwargs):
  """Skip mirrored variable creation."""
  kwargs['skip_mirrored_creator'] = True
  return next_creator(*args, **kwargs)


def apply_augmentation_op(data, op_index, op_level, prob_to_apply):
  """Applies one augmentation op to the data."""
  branch_fns = []
  for augment_op_name in AUG_OPS:
    augment_fn = NAME_TO_FUNC[augment_op_name]
    level_to_args_fn = LEVEL_TO_ARG[augment_op_name]

    def _branch_fn(data=data,
                   augment_fn=augment_fn,
                   level_to_args_fn=level_to_args_fn):
      args = [data] + list(level_to_args_fn(op_level))
      return augment_fn(*args)

    branch_fns.append(_branch_fn)
  aug_data = tf.switch_case(op_index, branch_fns, default=lambda: data)
  if prob_to_apply is not None:
    return tf.cond(
        tf.random.uniform(shape=[], dtype=tf.float32) <
        prob_to_apply, lambda: aug_data, lambda: data)
  else:
    return aug_data


class CTAugment(object):
  """Implementation of control theory augment."""

  def __init__(self,
               num_layers=2,
               confidence_threshold=0.85,
               decay=0.99,
               epsilon=0.001,
               prob_to_apply=None,
               num_levels=10):
    """Initialize CT Augment.

    Args:
      num_layers: number of augmentation layers, i.e. how many times to do
        augmentation.
      confidence_threshold: confidence threshold for probabilities
      decay: decay factor for augmentation rates
      epsilon: samll number which is used to avoid numerical instabilities
        while computing probabilities.
      prob_to_apply: probability to apply on each layer.
        If None then always apply.
      num_levels: number of levels for quantization of the magnitude.
    """
    # Augmenter args
    self.num_layers = num_layers
    self.confidence_threshold = float(confidence_threshold)
    self.decay = float(decay)
    self.alpha = 1.0 - self.decay
    self.epsilon = epsilon
    self.num_levels = int(num_levels)
    self.prob_to_apply = prob_to_apply
    # State of the augmenter is defined by rates.
    # 增强器的状态由rates定义。
    # To speed up sampling we also keep separate variable for sampling
    # probabilities (log_probs) which are deterministically computed from rates.
    # 为了加快采样速度，我们还为采样概率(log_probs)保留单独的变量，这些概率是根据rates确定计算的。
    self.state_shape = [len(AUG_OPS), self.num_levels]
    # NOTE 这些以下的副本更新方式都是用于分布式训练的,在单卡训练无关紧要
    # rates are updated using assign_add and averaged across all replicas.
    # 使用assign_add更新rates，并对所有更新进行平均。
    self.rates = tf.Variable(
        tf.ones(self.state_shape, dtype=tf.float32),
        trainable=False,
        name='cta_rates',
        aggregation=tf.VariableAggregation.MEAN,
        synchronization=tf.VariableSynchronization.ON_WRITE)
    # log_probs is deterministically computed from rates and value should
    # be the same on all replicas, thus we use ONLY_FIRST_REPLICA aggregation
    # log_probs是根据rates确定计算的，所有副本上的值应该相同，因此我们只使用第一个副本聚合
    self.probs = tf.Variable(
        tf.ones(self.state_shape, dtype=tf.float32) / self.num_levels,
        trainable=False,
        name='cta_probs',
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        synchronization=tf.VariableSynchronization.ON_WRITE)
    # list of log probs variables for each data pipeline
    # NOTE 为每个数据管道设置的log probs变量列表
    self.log_probs = []

  def update(self, tensor_dict, probe_probs):
    """Update augmenter state to classification of probe images."""
    # shape of probe_probs is (batch_size, num_classes)
    op_idx = tensor_dict['probe_op_indices']  # shape=(batch_size, num_layers)
    op_arg = tensor_dict['probe_op_args']  # shape=(batch_size, num_layers)
    label = tf.expand_dims(tensor_dict['label'], 1)  # shape=(batch_size, 1)

    # Compute proximity metric as softmax(model(probe_data))[correct_label]
    # Tile proximity, so its shape will be (batch_size, num_layers)
    proximity = tf.gather(probe_probs, label, axis=1, batch_dims=1)
    proximity = tf.tile(proximity, [1, self.num_layers])
    # Quantize op_arg to obtain levels of the ops.
    # NOTE: computed level should be always less than num_levels,
    #       nevertherless use minimum operation to enforce the range.
    level_idx = tf.cast(op_arg * self.num_levels, tf.int32)
    level_idx = tf.minimum(level_idx, self.num_levels)

    # Update rates.
    # For each (op_index, level_index, proximity) in the list of selected ops
    # update rate using following formula:
    #   rate[op_idx, level_idx] = rate[op_idx, level_idx] * decay
    #                             + proximity * (1 - decay)
    # which is equivalent to:
    #   alpha = 1 - decay
    #   rate[op_idx, level_idx] += (proximity - rate[op_idx, level_idx]) * alpha
    #
    # So update is performed using assign_add operation. If several updates
    # correpond to the same (op_idx, level_idx) then they are averaged.
    op_level_idx = tf.concat(
        [tf.reshape(op_idx, [-1, 1]),
         tf.reshape(level_idx, [-1, 1])], axis=1)
    flat_proximity = tf.reshape(proximity, [-1])
    sparse_update = ((flat_proximity - tf.gather_nd(self.rates, op_level_idx)) *
                     self.alpha)
    # Dense matrix with updates is computed in dense_update_numerator.
    # tf.scatter_nd adds up all updates which correspond to the same index,
    # however we need to compute mean. Thus we compute number of
    # updates corresponding to each index and divide by this number.
    dense_update_numerator = tf.scatter_nd(
        op_level_idx, sparse_update, shape=self.state_shape)
    dense_update_denominator = tf.scatter_nd(
        op_level_idx, tf.ones_like(sparse_update), shape=self.state_shape)
    dense_update_denominator = tf.maximum(dense_update_denominator, 1.0)
    self.rates.assign_add(dense_update_numerator / dense_update_denominator)

    # Convert rates to log probabilities
    probs = tf.maximum(self.rates, self.epsilon)
    probs = probs / tf.reduce_max(probs, axis=1, keepdims=True)
    probs = tf.where(probs < self.confidence_threshold, tf.zeros_like(probs),
                     probs)
    probs = probs + self.epsilon
    probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
    self.probs.assign(probs)

  def sync_state(self):
    log_prob_value = tf.math.log(self.probs)
    for v in self.log_probs:
      v.assign(log_prob_value)

  def get_state(self):
    """Returns augmenter state to save in checkpoint or for debugging."""
    return {
        'ct_augment_rates': self.rates,
        'ct_augment_probs': self.probs,
    }

  def _sample_ops_uniformly(self) -> [tf.Tensor, tf.Tensor]:
    """Uniformly samples sequence of augmentation ops."""
    op_indices = tf.random.uniform(
        shape=[self.num_layers], maxval=len(AUG_OPS), dtype=tf.int32)
    op_args = tf.random.uniform(shape=[self.num_layers], dtype=tf.float32)
    return op_indices, op_args

  def _sample_ops(self, local_log_prob):
    """Samples sequence of augmentation ops using current probabilities."""
    # choose operations
    op_indices = tf.random.uniform(
        shape=[self.num_layers], maxval=len(AUG_OPS), dtype=tf.int32)
    # sample arguments for each selected operation
    selected_ops_log_probs = tf.gather(local_log_prob, op_indices, axis=0)
    op_args = tf.random.categorical(selected_ops_log_probs, num_samples=1)
    op_args = tf.cast(tf.squeeze(op_args, axis=1), tf.float32)
    op_args = (op_args + tf.random.uniform([self.num_layers])) / self.num_levels
    return op_indices, op_args

  def _apply_ops(self, data, op_indices, op_args, prob_to_apply=None):
    org_shape = data.shape
    for idx in range(self.num_layers):
      data = apply_augmentation_op(data, op_indices[idx], op_args[idx],
                                   prob_to_apply)
      data.set_shape(org_shape)
    return data

  def __call__(self, data: tf.Tensor, probe: bool = True,
               aug_key: str = 'data') -> dict:
    """
      When training labeled data, use `probe=True` to update weights
      
      When training unlabeled data, use `probe=False aug_key=aug_data` to augment data
    
    Args:
        data (tf.Tensor): 
        probe (bool, optional): Defaults to True.
        aug_key (str, optional): Defaults to 'data'.
    
    Returns:
        dict: data_dict
    """
    # creating local variable which will store copy of CTA log probabilities
    with tf.variable_creator_scope(_skip_mirrored_creator):
      local_log_prob = tf.Variable(
          lambda: tf.ones(self.state_shape, dtype=tf.float32),
          trainable=False,
          name='cta_log_probs')
    self.log_probs.append(local_log_prob)

    output_dict = {}
    if probe:
      # 采样 [num_layers] 个 op_indices 和 op_args
      probe_op_indices, probe_op_args = self._sample_ops_uniformly()
      probe_data = self._apply_ops(data, probe_op_indices, probe_op_args)
      output_dict['probe_op_indices'] = probe_op_indices
      output_dict['probe_op_args'] = probe_op_args
      output_dict['probe_data'] = probe_data

    if aug_key is not None:
      op_indices, op_args = self._sample_ops(local_log_prob)
      aug_data = self._apply_ops(
          data, op_indices, op_args, prob_to_apply=self.prob_to_apply)
      output_dict[aug_key] = aug_data

    if aug_key != 'data':
      output_dict['data'] = data

    return output_dict
