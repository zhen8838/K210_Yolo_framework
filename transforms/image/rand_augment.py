import tensorflow as tf
from transforms.image import ops
import inspect
NAME_TO_FUNC = {
    'Identity': tf.identity,
    'AutoContrast': ops.autocontrast,
    'AutoContrastBlend': ops.autocontrast_blend,
    'Equalize': ops.equalize,
    'EqualizeBlend': ops.equalize_blend,
    'Invert': ops.invert,
    'InvertBlend': ops.invert_blend,
    'Rotate': ops.rotate,
    'Posterize': ops.posterize,
    'Solarize': ops.solarize,
    'SolarizeAdd': ops.solarize_add,
    'Color': ops.color,
    'Contrast': ops.contrast,
    'Brightness': ops.brightness,
    'Sharpness': ops.sharpness,
    'ShearX': ops.shear_x,
    'ShearY': ops.shear_y,
    'TranslateX': ops.translate_x,
    'TranslateY': ops.translate_y,
    'Blur': ops.blur,
    'Smooth': ops.smooth,
    'Rescale': ops.rescale,
}

# Reference for Imagenet:
# https://cs.corp.google.com/piper///depot/google3/learning/brain/research/meta_architect/data/data_processing.py?rcl=275474938&l=2950

IMAGENET_AUG_OPS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'SolarizeAdd',
    'Identity',
]

# Levels in this file are assumed to be floats in [0, 1] range
# If you need quantization or integer levels, this should be controlled
# in client code.
MAX_LEVEL = 1.

# Constant which is used when computing translation argument from level
TRANSLATE_CONST = 100.


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level/MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
  level = (level/MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level):
  level = (level/MAX_LEVEL) * TRANSLATE_CONST
  # Flip level to negative with 50% chance
  level = _randomly_negate_tensor(level)
  return (level,)


def _posterize_level_to_arg(level):
  return (tf.cast((level/MAX_LEVEL) * 4, tf.uint8),)


def _solarize_level_to_arg(level):
  return (tf.cast((level/MAX_LEVEL) * 256, tf.int32),)


def _solarize_add_level_to_arg(level):
  return (tf.cast((level/MAX_LEVEL) * 110, tf.int32),)


def _ignore_level_to_arg(level):
  del level
  return ()


def _divide_level_by_max_level_arg(level):
  return (level / MAX_LEVEL,)


LEVEL_TO_ARG = {
    'AutoContrast': _ignore_level_to_arg,
    'Equalize': _ignore_level_to_arg,
    'Invert': _ignore_level_to_arg,
    'Rotate': _rotate_level_to_arg,
    'Posterize': _posterize_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_level_to_arg,
    'TranslateY': _translate_level_to_arg,
    'Identity': _ignore_level_to_arg,
    'Blur': _divide_level_by_max_level_arg,
    'Smooth': _divide_level_by_max_level_arg,
    'Rescale': _divide_level_by_max_level_arg,
}


class RandAugment(object):
  """Random augment with fixed magnitude."""

  def __init__(self,
               num_layers=2,
               prob_to_apply=None,
               magnitude=None,
               num_levels=10,
               replace=[128, 128, 128]):
    """Initialized rand augment.

    Args:
      num_layers: number of augmentation layers, i.e. how many times to do
        augmentation.
      prob_to_apply: probability to apply on each layer.
        If None then always apply.
      magnitude: default magnitude in range [0, 1],
        if None then magnitude will be chosen randomly.
      num_levels: number of levels for quantization of the magnitude.
    """
    self.num_layers = num_layers
    self.prob_to_apply = (
        float(prob_to_apply) if prob_to_apply is not None else None)
    self.num_levels = int(num_levels) if num_levels else None
    self.level = float(magnitude) if magnitude is not None else None
    self.replace = replace

  def _get_level(self):
    if self.level is not None:
      return tf.convert_to_tensor(self.level)
    if self.num_levels is None:
      return tf.random.uniform(shape=[], dtype=tf.float32)
    else:
      level = tf.random.uniform(
          shape=[], maxval=self.num_levels + 1, dtype=tf.int32)
      return tf.cast(level, tf.float32) / self.num_levels

  def _apply_one_layer(self, data):
    """Applies one level of augmentation to the data."""
    level = self._get_level()
    branch_fns = []
    for augment_op_name in IMAGENET_AUG_OPS:
      augment_fn = NAME_TO_FUNC[augment_op_name]
      level_to_args_fn = LEVEL_TO_ARG[augment_op_name]

      def _branch_fn(data=data,
                     augment_fn=augment_fn,
                     level_to_args_fn=level_to_args_fn):

        args = [data] + list(level_to_args_fn(level))
        fuc_args = inspect.getfullargspec(augment_fn).args
        if 'replace' in fuc_args and 'replace' == fuc_args[-1]:
          # Make sure replace is the final argument
          args.append(self.replace)
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

  def __call__(self, data, aug_key='data'):
    output_dict = {}

    if aug_key is not None:
      aug_data = data
      for _ in range(self.num_layers):
        aug_data = self._apply_one_layer(aug_data)
      output_dict[aug_key] = aug_data

    if aug_key != 'data':
      output_dict['data'] = data

    return output_dict
