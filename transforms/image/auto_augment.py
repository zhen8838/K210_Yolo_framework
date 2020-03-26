from transforms.image import ops
import inspect
from easydict import EasyDict
import tensorflow as tf
# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def policy_v0():
  """Autoaugment policy that was used in AutoAugment Detection Paper."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
      [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
      [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
      [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
      [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
  ]
  return policy


def policy_v1():
  """Autoaugment policy that was used in AutoAugment Detection Paper."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
      [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
      [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
      [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
      [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
      [('Color', 0.0, 0), ('ShearX_Only_BBoxes', 0.8, 4)],
      [('ShearY_Only_BBoxes', 0.8, 2), ('Flip_Only_BBoxes', 0.0, 10)],
      [('Equalize', 0.6, 10), ('TranslateX_BBox', 0.2, 2)],
      [('Color', 1.0, 10), ('TranslateY_Only_BBoxes', 0.4, 6)],
      [('Rotate_BBox', 0.8, 10), ('Contrast', 0.0, 10)],
      [('Cutout', 0.2, 2), ('Brightness', 0.8, 10)],
      [('Color', 1.0, 6), ('Equalize', 1.0, 2)],
      [('Cutout_Only_BBoxes', 0.4, 6), ('TranslateY_Only_BBoxes', 0.8, 2)],
      [('Color', 0.2, 8), ('Rotate_BBox', 0.8, 10)],
      [('Sharpness', 0.4, 4), ('TranslateY_Only_BBoxes', 0.0, 4)],
      [('Sharpness', 1.0, 4), ('SolarizeAdd', 0.4, 4)],
      [('Rotate_BBox', 1.0, 8), ('Sharpness', 0.2, 8)],
      [('ShearY_BBox', 0.6, 10), ('Equalize_Only_BBoxes', 0.6, 8)],
      [('ShearX_BBox', 0.2, 6), ('TranslateY_Only_BBoxes', 0.2, 10)],
      [('SolarizeAdd', 0.6, 8), ('Brightness', 0.8, 10)],
  ]
  return policy


def policy_vtest():
  """Autoaugment test policy for debugging."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('TranslateX_BBox', 1.0, 4), ('Equalize', 1.0, 10)],
  ]
  return policy


def policy_v2():
  """Additional policy that performs well on object detection."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
      [('Rotate_BBox', 0.4, 8), ('Sharpness', 0.4, 2), ('Rotate_BBox', 0.8, 10)],
      [('TranslateY_BBox', 1.0, 8), ('AutoContrast', 0.8, 2)],
      [('AutoContrast', 0.4, 6), ('ShearX_BBox', 0.8, 8),
       ('Brightness', 0.0, 10)],
      [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10), ('AutoContrast', 0.6, 0)],
      [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
      [('TranslateY_BBox', 0.0, 4), ('Equalize', 0.6, 8), ('Solarize', 0.0, 10)],
      [('TranslateY_BBox', 0.2, 2), ('ShearY_BBox', 0.8, 8),
       ('Rotate_BBox', 0.8, 8)],
      [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
      [('Color', 0.8, 4), ('TranslateY_BBox', 1.0, 6), ('Rotate_BBox', 0.6, 6)],
      [('Rotate_BBox', 0.6, 10), ('BBox_Cutout', 1.0, 4), ('Cutout', 0.2, 8)],
      [('Rotate_BBox', 0.0, 0), ('Equalize', 0.6, 6), ('ShearY_BBox', 0.6, 8)],
      [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2), ('Brightness', 0.2, 2)],
      [('TranslateY_BBox', 0.4, 8), ('Solarize', 0.4, 6),
       ('SolarizeAdd', 0.2, 10)],
      [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
  ]
  return policy


def policy_v3():
  """"Additional policy that performs well on object detection."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('Posterize', 0.8, 2), ('TranslateX_BBox', 1.0, 8)],
      [('BBox_Cutout', 0.2, 10), ('Sharpness', 1.0, 8)],
      [('Rotate_BBox', 0.6, 8), ('Rotate_BBox', 0.8, 10)],
      [('Equalize', 0.8, 10), ('AutoContrast', 0.2, 10)],
      [('SolarizeAdd', 0.2, 2), ('TranslateY_BBox', 0.2, 8)],
      [('Sharpness', 0.0, 2), ('Color', 0.4, 8)],
      [('Equalize', 1.0, 8), ('TranslateY_BBox', 1.0, 8)],
      [('Posterize', 0.6, 2), ('Rotate_BBox', 0.0, 10)],
      [('AutoContrast', 0.6, 0), ('Rotate_BBox', 1.0, 6)],
      [('Equalize', 0.0, 4), ('Cutout', 0.8, 10)],
      [('Brightness', 1.0, 2), ('TranslateY_BBox', 1.0, 6)],
      [('Contrast', 0.0, 2), ('ShearY_BBox', 0.8, 0)],
      [('AutoContrast', 0.8, 10), ('Contrast', 0.2, 10)],
      [('Rotate_BBox', 1.0, 10), ('Cutout', 1.0, 10)],
      [('SolarizeAdd', 0.8, 6), ('Equalize', 0.8, 8)],
  ]
  return policy


NAME_TO_FUNC = {
    'AutoContrast':
        ops.autocontrast,
    'Equalize':
        ops.equalize,
    'Posterize':
        ops.posterize,
    'Solarize':
        ops.solarize,
    'SolarizeAdd':
        ops.solarize_add,
    'Color':
        ops.color,
    'Contrast':
        ops.contrast,
    'Brightness':
        ops.brightness,
    'Sharpness':
        ops.sharpness,
    'Cutout':
        ops.cutout,
    'BBox_Cutout':
        ops.bbox_cutout,
    'Rotate_BBox':
        ops.rotate_with_bboxes,
    # pylint:disable=g-long-lambda
    'TranslateX_BBox':
        lambda image, bboxes, pixels, replace: ops.translate_bbox(
            image, bboxes, pixels, replace, shift_horizontal=True),
    'TranslateY_BBox':
        lambda image, bboxes, pixels, replace: ops.translate_bbox(
            image, bboxes, pixels, replace, shift_horizontal=False),
    'ShearX_BBox':
        lambda image, bboxes, level, replace: ops.shear_with_bboxes(
            image, bboxes, level, replace, shear_horizontal=True),
    'ShearY_BBox':
        lambda image, bboxes, level, replace: ops.shear_with_bboxes(
            image, bboxes, level, replace, shear_horizontal=False),
    # pylint:enable=g-long-lambda
    'Rotate_Only_BBoxes':
        ops.rotate_only_bboxes,
    'ShearX_Only_BBoxes':
        ops.shear_x_only_bboxes,
    'ShearY_Only_BBoxes':
        ops.shear_y_only_bboxes,
    'TranslateX_Only_BBoxes':
        ops.translate_x_only_bboxes,
    'TranslateY_Only_BBoxes':
        ops.translate_y_only_bboxes,
    'Flip_Only_BBoxes':
        ops.flip_only_bboxes,
    'Solarize_Only_BBoxes':
        ops.solarize_only_bboxes,
    'Equalize_Only_BBoxes':
        ops.equalize_only_bboxes,
    'Cutout_Only_BBoxes':
        ops.cutout_only_bboxes,
}


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL/level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _bbox_cutout_level_to_arg(level, hparams):
  cutout_pad_fraction = (level/_MAX_LEVEL) * hparams.cutout_max_pad_fraction
  return (cutout_pad_fraction, hparams.cutout_bbox_replace_with_mean)


def level_to_arg(hparams):
  return {
      'AutoContrast':
          lambda level: (),
      'Equalize':
          lambda level: (),
      'Posterize':
          lambda level: (int((level/_MAX_LEVEL) * 4),),
      'Solarize':
          lambda level: (int((level/_MAX_LEVEL) * 256),),
      'SolarizeAdd':
          lambda level: (int((level/_MAX_LEVEL) * 110),),
      'Color':
          _enhance_level_to_arg,
      'Contrast':
          _enhance_level_to_arg,
      'Brightness':
          _enhance_level_to_arg,
      'Sharpness':
          _enhance_level_to_arg,
      'Cutout':
          lambda level: (int((level/_MAX_LEVEL) * hparams.cutout_const),),
      # pylint:disable=g-long-lambda
      'BBox_Cutout':
          lambda level: _bbox_cutout_level_to_arg(level, hparams),
      'TranslateX_BBox':
          lambda level: _translate_level_to_arg(level, hparams.translate_const),
      'TranslateY_BBox':
          lambda level: _translate_level_to_arg(level, hparams.translate_const),
      # pylint:enable=g-long-lambda
      'ShearX_BBox':
          _shear_level_to_arg,
      'ShearY_BBox':
          _shear_level_to_arg,
      'Rotate_BBox':
          _rotate_level_to_arg,
      'Rotate_Only_BBoxes':
          _rotate_level_to_arg,
      'ShearX_Only_BBoxes':
          _shear_level_to_arg,
      'ShearY_Only_BBoxes':
          _shear_level_to_arg,
      # pylint:disable=g-long-lambda
      'TranslateX_Only_BBoxes':
          lambda level: _translate_level_to_arg(level, hparams.
                                                translate_bbox_const),
      'TranslateY_Only_BBoxes':
          lambda level: _translate_level_to_arg(level, hparams.
                                                translate_bbox_const),
      # pylint:enable=g-long-lambda
      'Flip_Only_BBoxes':
          lambda level: (),
      'Solarize_Only_BBoxes':
          lambda level: (int((level/_MAX_LEVEL) * 256),),
      'Equalize_Only_BBoxes':
          lambda level: (),
      # pylint:disable=g-long-lambda
      'Cutout_Only_BBoxes':
          lambda level: (int((level/_MAX_LEVEL) * hparams.cutout_bbox_const),),
      # pylint:enable=g-long-lambda
  }


def bbox_wrapper(func):
  """Adds a bboxes function argument to func and returns unchanged bboxes."""

  def wrapper(images, bboxes, *args, **kwargs):
    return (func(images, *args, **kwargs), bboxes)

  return wrapper


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(augmentation_hparams)[name](level)

  # Check to see if prob is passed into function. This is used for operations
  # where we alter bboxes independently.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getfullargspec(func)[0]:
    args = tuple([prob] + list(args))
  # pytype:enable=wrong-arg-types

  # Add in replace arg if it is required for the function that is being called.
  if 'replace' in inspect.getfullargspec(func)[0]:
    # Make sure replace is the final argument
    assert 'replace' == inspect.getfullargspec(func)[0][-1]
    args = tuple(list(args) + [replace_value])

  # Add bboxes as the second positional argument for the function if it does
  # not already exist.
  if 'bboxes' not in inspect.getfullargspec(func)[0]:
    func = bbox_wrapper(func)
  return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob, bboxes):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)
  assert 'bboxes' == inspect.getargspec(func)[0][1]

  # If prob is a function argument, then this randomness is being handled
  # inside the function, so make sure it is always called.
  if 'prob' in inspect.getargspec(func)[0]:
    prob = 1.0

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image, augmented_bboxes = tf.cond(should_apply_op, lambda: func(
      image, bboxes, *args), lambda: (image, bboxes))
  return augmented_image, augmented_bboxes


def select_and_apply_random_policy(policies, image, bboxes):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image, bboxes = tf.compat.v1.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image, bboxes),
        lambda: (image, bboxes))
  return (image, bboxes)


def build_and_apply_nas_policy(policies, image, bboxes, augmentation_hparams):
  """Build a policy from the given policies passed in and apply to image.

    Args:
      policies: list of lists of tuples in the form `(func, prob, level)`, `func`
        is a string name of the augmentation function, `prob` is the probability
        of applying the `func` operation, `level` is the input argument for
        `func`.
      image: tf.Tensor that the resulting policy will be applied to.
      bboxes:
      augmentation_hparams: Hparams associated with the NAS learned policy.

    Returns:
      A version of image that now has data augmentation applied to it based on
      the `policies` pass into the function. Additionally, returns bboxes if
      a value for them is passed in that is not None
    """
  replace_value = [128, 128, 128]

  # func is the string name of the augmentation function, prob is the
  # probability of applying the operation and level is the parameter associated
  # with the tf op.

  # tf_policies are functions that take in an image and return an augmented
  # image.
  tf_policies = []
  for policy in policies:
    tf_policy = []
    # Link string name to the correct python function and make sure the correct
    # argument is passed into that function.
    for policy_info in policy:
      policy_info = list(policy_info) + [replace_value, augmentation_hparams]

      tf_policy.append(_parse_policy_info(*policy_info))
    # Now build the tf policy that will apply the augmentation procedue
    # on image.

    def make_final_policy(tf_policy_):

      def final_policy(image_, bboxes_):
        for func, prob, args in tf_policy_:
          image_, bboxes_ = _apply_func_with_prob(func, image_, args, prob,
                                                  bboxes_)
        return image_, bboxes_

      return final_policy

    tf_policies.append(make_final_policy(tf_policy))

  augmented_images, augmented_bboxes = select_and_apply_random_policy(
      tf_policies, image, bboxes)
  # If no bounding boxes were specified, then just return the images.
  return (augmented_images, augmented_bboxes)


# TODO(barretzoph): Add in ArXiv link once paper is out.
def distort_image_with_autoaugment(image, bboxes, augmentation_name):
  """Applies the AutoAugment policy to `image` and `bboxes`.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      bboxes: `Tensor` of shape [N, 4] representing ground truth boxes that are
        normalized between [0, 1].
      augmentation_name: The name of the AutoAugment policy to use. The available
        options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
        all of the results in the paper and was found to achieve the best results
        on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
        found on the COCO dataset that have slight variation in what operations
        were used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3).

    Returns:
      A tuple containing the augmented versions of `image` and `bboxes`.
    """
  available_policies = {
      'v0': policy_v0,
      'v1': policy_v1,
      'v2': policy_v2,
      'v3': policy_v3,
      'test': policy_vtest
  }
  if augmentation_name not in available_policies:
    raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

  policy = available_policies[augmentation_name]()
  # Hparams that will be used for AutoAugment.
  augmentation_hparams = EasyDict(
      cutout_max_pad_fraction=0.75,
      cutout_bbox_replace_with_mean=False,
      cutout_const=100,
      translate_const=250,
      cutout_bbox_const=50,
      translate_bbox_const=120)

  return build_and_apply_nas_policy(policy, image, bboxes, augmentation_hparams)
