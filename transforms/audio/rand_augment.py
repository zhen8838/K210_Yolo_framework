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
  # Note factor loop in [0. ~ 0.2]
  limit = tf.constant(0.2, tf.float32)
  factor = tf.math.mod(level, limit)
  factor = tf.cond(tf.equal(factor, 0.), lambda: limit, lambda: factor)
  times = tf.cast(tf.math.floordiv(level, limit), tf.int32) + 1
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
  # Note factor loop in [0. ~ 0.2]
  factor = tf.math.mod(level, 0.2)
  factor = tf.cond(tf.equal(factor, 0.), lambda: 0.2, lambda: factor)

  npoints = tf.cast(tf.math.floordiv(level, 0.2), tf.int32) + 1
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


class RandAugment(object):
  """Random augment with fixed magnitude."""

  def __init__(self,
               num_layers: int = 2,
               prob_to_apply: float = None,
               num_levels: int = 10):
    """Initialized rand augment.
    
    Args:
        num_layers (int, optional): how many times to do augmentation. Defaults to 2.
        prob_to_apply (float, optional): probability to apply on each layer.
        If None then always apply. Defaults to None.
        num_levels (int, optional): number of levels for quantization of the magnitude. Defaults to 10.
    """
    self.num_layers = num_layers
    self.prob_to_apply = (
        float(prob_to_apply) if prob_to_apply is not None else None)
    self.num_levels = int(num_levels) if num_levels else None

  def _get_level(self):
    level = tf.random.uniform([], 1, self.num_levels + 1, tf.int32)
    return (tf.cast(level, tf.float32) / self.num_levels)

  def _apply_one_layer(self, data):
    """Applies one level of augmentation to the data."""
    level = self._get_level()
    branch_fns = []
    for augment_op_name in AUG_OPS:
      augment_fn = NAME_TO_FUNC[augment_op_name]
      level_to_args_fn = LEVEL_TO_ARG[augment_op_name]

      def _branch_fn(data=data,
                     augment_fn=augment_fn,
                     level_to_args_fn=level_to_args_fn):
        args = [data] + list(level_to_args_fn(level))
        return augment_fn(*args)

      branch_fns.append(_branch_fn)

    branch_index = tf.random.uniform(
        shape=[], maxval=len(branch_fns), dtype=tf.int32)
    aug_data = tf.switch_case(branch_index, branch_fns, default=lambda: data)
    if self.prob_to_apply is not None:
      return tf.cond(
          tf.random.uniform(shape=[], dtype=tf.float32) <
          self.prob_to_apply, lambda: aug_data, lambda: data)
    else:
      return aug_data

  def __call__(self, data: tf.Tensor, aug_key='data') -> tf.Tensor:
    output_dict = {}
    org_shape = data.shape

    if aug_key is not None:
      aug_data = data
      for _ in range(self.num_layers):
        aug_data = self._apply_one_layer(aug_data)
        # NOTE must set shape for while_loop !
        aug_data.set_shape(org_shape)
      output_dict[aug_key] = aug_data

    if aug_key != 'data':
      output_dict['data'] = data

    return output_dict
