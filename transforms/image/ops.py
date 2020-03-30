import tensorflow as tf
import tensorflow_addons as tfa
from transforms.image.box_utils import clip_boxes, clip_keypoints
import math
# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]


def letterbox_resize(image: tf.Tensor,
                     desired_size: tf.Tensor,
                     method: str = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                    ) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
  """letter box resize image function

    Args:
        image (tf.Tensor): dtype : tf.float32 tor tf.uint8
        desired_size (tf.Tensor): target image size : `[heihgt,width]`
        method (str, optional): resize method. Defaults to `tf.image.ResizeMethod.NEAREST_NEIGHBOR`

    Returns:
        img (tf.Tensor): image
        scale (tf.Tensor): image resize scale, shape: [2]
        size (tf.Tensor): image target size, shape: [2]
        offset (tf.Tensor): image resize offset, shape: [2]

        NOTE using `NEAREST_NEIGHBOR` img dtype same as before, using other img dtype is float32
        scale , offset use for bbox transfrom
    """
  img_hw = tf.shape(image)[:2]

  def _resize(img, img_hw):
    img_hw = tf.cast(img_hw, tf.float32)
    hw = tf.cast(desired_size, tf.float32)
    scale = tf.tile(tf.reduce_min(hw / img_hw, keepdims=True), [2])
    new_hw = img_hw * scale
    offset = (hw-new_hw) / 2
    img = tf.image.resize(img, tf.cast(new_hw, tf.int32), method=method)
    img = tf.image.pad_to_bounding_box(img, tf.cast(offset[0], tf.int32),
                                       tf.cast(offset[1], tf.int32),
                                       tf.cast(hw[0], tf.int32),
                                       tf.cast(hw[1], tf.int32))
    return img, scale, offset

  img, scale, offset = tf.cond(
      tf.reduce_all(tf.equal(size, img_hw)), lambda: (tf.cast(
          image, image.dtype if method == tf.image.ResizeMethod.NEAREST_NEIGHBOR
          else tf.float32), tf.ones([2]), tf.zeros([2])), lambda: _resize(
              image, img_hw))
  return img, scale, desired_size, offset


def compute_padded_size(desired_size, stride):
  """Compute the padded size given the desired size and the stride.

    The padded size will be the smallest rectangle, such that each dimension is
    the smallest multiple of the stride which is larger than the desired
    dimension. For example, if desired_size = (100, 200) and stride = 32,
    the output padded_size = (128, 224).

    Args:
      desired_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the target output image size.
      stride: an integer, the stride of the backbone network.

    Returns:
      padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size.
    """
  if isinstance(desired_size, list) or isinstance(desired_size, tuple):
    padded_size = [
        int(math.ceil(d * 1.0 / stride) * stride) for d in desired_size
    ]
  else:
    padded_size = tf.cast(
        tf.math.ceil(tf.cast(desired_size, dtype=tf.float32) / stride) * stride,
        tf.int32)
  return padded_size


def resize_clip_boxes(boxes: tf.Tensor, image_scale: tf.Tensor,
                      output_size: tf.Tensor, offset: tf.Tensor) -> tf.Tensor:
  """Resizes and clip boxes to output size with scale and offset.

    Args:
        boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
        image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
        output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
        offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.

    Returns:
        boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
  # Adjusts box coordinates based on image_scale and offset.
  boxes = boxes * tf.tile(image_scale, [2]) + tf.tile(offset, [2])
  boxes = clip_boxes(boxes, output_size)
  return boxes


def retinanet_resize(image,
                     desired_size,
                     padded_size,
                     aug_scale_min=1.0,
                     aug_scale_max=1.0,
                     method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size (RetinaNet style). 
        NOTE will clip image
    Resize and pad images given the desired output size of the image and
    stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
       the largest rectangle to be bounded by the rectangle specified by the
       `desired_size`.
    2. Pad the rescaled image to the padded_size.

    Args:
      image: a `Tensor` of shape [height, width, 3] representing an image.
      desired_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the desired actual output image size.
      padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size. Padding will be applied
        after scaling the image to the desired_size.
      aug_scale_min: a `float` with range between [0, 1.0] representing minimum
        random scale applied to desired_size for training scale jittering.
      aug_scale_max: a `float` with range between [1.0, inf] representing maximum
        random scale applied to desired_size for training scale jittering.
      method: function to resize input image to scaled image.

    Returns:
      output_image: `Tensor` of shape [height, width, 3] where [height, width]
        equals to `output_size`.
      image_scale: [y_scale, x_scale]
      desired_size: [desired_height, desired_width],
      offset: [y_offset, x_offset]]
    """
  with tf.name_scope('retinanet_resize'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      scaled_size = desired_size

    scale = tf.minimum(scaled_size[0] / image_size[0],
                       scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(
          tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
      offset = max_offset * tf.random.uniform([
          2,
      ], 0, 1)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] +
                                  desired_size[0], offset[1]:offset[1] +
                                  desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                padded_size[0], padded_size[1])

    return output_image, image_scale, desired_size, -tf.cast(offset, tf.float32)


def fastrcnn_resize(image,
                    short_side,
                    long_side,
                    padded_size,
                    aug_scale_min=1.0,
                    aug_scale_max=1.0,
                    method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size (Faster R-CNN style).

    Resize and pad images given the specified short / long side length and the
    stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and first try to rescale the short
       side of the original image to `short_side`.
    2. If the scaled image after 1 has a long side that exceeds `long_side`, keep
       the aspect ratio and rescal the long side of the image to `long_side`.
    2. Pad the rescaled image to the padded_size.

    Args:
      image: a `Tensor` of shape [height, width, 3] representing an image.
      short_side: a scalar `Tensor` or `int` representing the desired short side
        to be rescaled to.
      long_side: a scalar `Tensor` or `int` representing the desired long side to
        be rescaled to.
      padded_size: a `Tensor` or `int` list/tuple of two elements representing
        [height, width] of the padded output image size. Padding will be applied
        after scaling the image to the desired_size.
      aug_scale_min: a `float` with range between [0, 1.0] representing minimum
        random scale applied to desired_size for training scale jittering.
      aug_scale_max: a `float` with range between [1.0, inf] representing maximum
        random scale applied to desired_size for training scale jittering.
      method: function to resize input image to scaled image.

    Returns:
      output_image: `Tensor` of shape [height, width, 3] where [height, width]
        equals to `output_size`.
      image_scale: [y_scale, x_scale]
      desired_size: [desired_height, desired_width],
      offset: [y_offset, x_offset]]
    """
  with tf.name_scope('resize_and_crop_image_v2'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    scale_using_short_side = (
        short_side / tf.minimum(image_size[0], image_size[1]))
    scale_using_long_side = (long_side / tf.maximum(image_size[0], image_size[1]))

    scaled_size = tf.round(image_size * scale_using_short_side)
    scaled_size = tf.where(
        tf.greater(tf.maximum(scaled_size[0], scaled_size[1]), long_side),
        tf.round(image_size * scale_using_long_side), scaled_size)
    desired_size = scaled_size

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([],
                                       aug_scale_min,
                                       aug_scale_max,
                                       seed=seed)
      scaled_size = tf.round(random_scale * scaled_size)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(
          tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
      offset = max_offset * tf.random.uniform([
          2,
      ], 0, 1)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] +
                                  desired_size[0], offset[1]:offset[1] +
                                  desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                padded_size[0], padded_size[1])

    return output_image, image_scale, desired_size, -tf.cast(offset, tf.float32)


def resize_keypoints(keypoints: tf.Tensor, image_scale: tf.Tensor,
                     output_size: tf.Tensor, offset: tf.Tensor) -> tf.Tensor:
  """Resizes boxes to output size with scale and offset.

    Args:
        keypoints: `Tensor` of shape [N, M, 2] representing ground truth boxes.
        image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
        output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
        offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.

    Returns:
        keypoints: `Tensor` of shape [N, M, 4] representing the scaled boxes.
    """
  # Adjusts box coordinates based on image_scale and offset.
  keypoints = keypoints*image_scale + offset
  # Clips the keypoints.
  keypoints = clip_keypoints(keypoints, output_size)
  return keypoints


def rescale(image, level):
  """Rescales image and enlarged cornet."""
  # TODO(kurakin): should we do center crop instead?
  # TODO(kurakin): add support of other resize methods
  # See tf.image.ResizeMethod for full list
  size = image.shape[:2]
  scale = level * 0.25
  scale_height = tf.cast(scale * size[0], tf.int32)
  scale_width = tf.cast(scale * size[1], tf.int32)
  cropped_image = tf.image.crop_to_bounding_box(
      image,
      offset_height=scale_height,
      offset_width=scale_width,
      target_height=size[0] - scale_height,
      target_width=size[1] - scale_width)
  rescaled = tf.image.resize(cropped_image, size, tf.image.ResizeMethod.BICUBIC)
  return tf.saturate_cast(rescaled, tf.uint8)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
  """Blend image1 and image2 using 'factor'.

  A value of factor 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor.
    image2: An image Tensor.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor.
  """
  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)
  return tf.saturate_cast(image1 + factor * (image2-image1), tf.uint8)


def cutout(image: tf.Tensor, pad_size: float, replace: float = 0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.

    Returns:
      An image Tensor that is of type uint8.
    """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [
      image_height - (lower_pad+upper_pad), image_width - (left_pad+right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace, image)
  return image


def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  threshold = tf.saturate_cast(threshold, image.dtype)
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor, addition=0, threshold=128) -> tf.Tensor:
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  threshold = tf.saturate_cast(threshold, image.dtype)
  added_im = tf.cast(image, tf.int32) + tf.cast(addition, tf.int32)
  added_im = tf.saturate_cast(added_im, tf.uint8)
  return tf.where(image < threshold, added_im, image)


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  return 255 - tf.convert_to_tensor(image)


def invert_blend(image: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
  """Implements blend of invert with original image."""
  return blend(invert(image), image, factor)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
  """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = tfa.image.rotate(wrap(image), radians)
  return unwrap(image, replace)


def random_shift_bbox(image,
                      bbox,
                      pixel_scaling,
                      replace,
                      new_min_bbox_coords=None):
  """Move the bbox and the image content to a slightly new random location.

    Args:
      image: 3D uint8 Tensor.
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
        The potential values for the new min corner of the bbox will be between
        [old_min - pixel_scaling * bbox_height/2,
         old_min - pixel_scaling * bbox_height/2].
      pixel_scaling: A float between 0 and 1 that specifies the pixel range
        that the new bbox location will be sampled from.
      replace: A one or three value 1D tensor to fill empty pixels.
      new_min_bbox_coords: If not None, then this is a tuple that specifies the
        (min_y, min_x) coordinates of the new bbox. Normally this is randomly
        specified, but this allows it to be manually set. The coordinates are
        the absolute coordinates between 0 and image height/width and are int32.

    Returns:
      The new image that will have the shifted bbox location in it along with
      the new bbox that contains the new coordinates.
    """
  # Obtains image height and width and create helper clip functions.
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)

  def clip_y(val):
    return tf.clip_by_value(val, 0, tf.cast(image_height, tf.int32) - 1)

  def clip_x(val):
    return tf.clip_by_value(val, 0, tf.cast(image_width, tf.int32) - 1)

  # Convert bbox to pixel coordinates.
  # min_y = tf.cast(image_height * bbox[0], tf.int32)
  # min_x = tf.cast(image_width * bbox[1], tf.int32)
  # max_y = clip_y(tf.cast(image_height * bbox[2], tf.int32))
  # max_x = clip_x(tf.cast(image_width * bbox[3], tf.int32))
  # todo change
  min_y = tf.cast(bbox[0], tf.int32)
  min_x = tf.cast(bbox[1], tf.int32)
  max_y = clip_y(tf.cast(bbox[2], tf.int32))
  max_x = clip_x(tf.cast(bbox[3], tf.int32))
  bbox_height, bbox_width = (max_y - min_y + 1, max_x - min_x + 1)
  image_height = tf.cast(image_height, tf.int32)
  image_width = tf.cast(image_width, tf.int32)

  # Select the new min/max bbox ranges that are used for sampling the
  # new min x/y coordinates of the shifted bbox.
  minval_y = clip_y(
      min_y -
      tf.cast(pixel_scaling * tf.cast(bbox_height, tf.float32) / 2.0, tf.int32))
  maxval_y = clip_y(
      min_y +
      tf.cast(pixel_scaling * tf.cast(bbox_height, tf.float32) / 2.0, tf.int32))
  minval_x = clip_x(min_x -
                    tf.cast(pixel_scaling * tf.cast(bbox_width, tf.float32) /
                            2.0, tf.int32))
  maxval_x = clip_x(min_x +
                    tf.cast(pixel_scaling * tf.cast(bbox_width, tf.float32) /
                            2.0, tf.int32))

  # Sample and calculate the new unclipped min/max coordinates of the new bbox.
  if new_min_bbox_coords is None:
    unclipped_new_min_y = tf.random.uniform(
        shape=[], minval=minval_y, maxval=maxval_y, dtype=tf.int32)
    unclipped_new_min_x = tf.random.uniform(
        shape=[], minval=minval_x, maxval=maxval_x, dtype=tf.int32)
  else:
    unclipped_new_min_y, unclipped_new_min_x = (clip_y(new_min_bbox_coords[0]),
                                                clip_x(new_min_bbox_coords[1]))
  unclipped_new_max_y = unclipped_new_min_y + bbox_height - 1
  unclipped_new_max_x = unclipped_new_min_x + bbox_width - 1

  # Determine if any of the new bbox was shifted outside the current image.
  # This is used for determining if any of the original bbox content should be
  # discarded.
  new_min_y, new_min_x, new_max_y, new_max_x = (clip_y(unclipped_new_min_y),
                                                clip_x(unclipped_new_min_x),
                                                clip_y(unclipped_new_max_y),
                                                clip_x(unclipped_new_max_x))
  shifted_min_y = (new_min_y-unclipped_new_min_y) + min_y
  shifted_max_y = max_y - (unclipped_new_max_y-new_max_y)
  shifted_min_x = (new_min_x-unclipped_new_min_x) + min_x
  shifted_max_x = max_x - (unclipped_new_max_x-new_max_x)

  # Create the new bbox tensor by converting pixel integer values to floats.
  new_bbox = tf.stack([
      tf.cast(new_min_y, tf.float32),
      tf.cast(new_min_x, tf.float32),
      tf.cast(new_max_y, tf.float32),
      tf.cast(new_max_x, tf.float32)
  ])

  # Copy the contents in the bbox and fill the old bbox location
  # with gray (128).
  bbox_content = image[shifted_min_y:shifted_max_y +
                       1, shifted_min_x:shifted_max_x + 1, :]

  def mask_and_add_image(min_y_, min_x_, max_y_, max_x_, mask, content_tensor,
                         image_):
    """Applies mask to bbox region in image then adds content_tensor to it."""
    mask = tf.pad(
        mask, [[min_y_,
                (image_height-1) - max_y_], [min_x_,
                                             (image_width-1) - max_x_], [0, 0]],
        constant_values=1)
    content_tensor = tf.pad(
        content_tensor,
        [[min_y_, (image_height-1) - max_y_], [min_x_,
                                               (image_width-1) - max_x_], [0, 0]],
        constant_values=0)
    return image_*mask + content_tensor

  # Zero out original bbox location.
  mask = tf.zeros_like(image)[min_y:max_y + 1, min_x:max_x + 1, :]
  grey_tensor = tf.zeros_like(mask) + replace[0]
  image = mask_and_add_image(min_y, min_x, max_y, max_x, mask, grey_tensor, image)

  # Fill in bbox content to new bbox location.
  mask = tf.zeros_like(bbox_content)
  image = mask_and_add_image(new_min_y, new_min_x, new_max_y, new_max_x, mask,
                             bbox_content, image)

  return image, new_bbox


def _clip_bbox(min_y, min_x, max_y, max_x, image_height, image_width):
  """Clip bounding box coordinates between 0 and image_height, image_width.

    Args:
      min_y: Normalized bbox coordinate of type float between 0 and image_height
      min_x: Normalized bbox coordinate of type float between 0 and image_width
      max_y: Normalized bbox coordinate of type float between 0 and image_height
      max_x: Normalized bbox coordinate of type float between 0 and image_width

    Returns:
      Clipped coordinate values between 0 and image_height, image_width.
    """
  min_y = tf.clip_by_value(min_y, 0.0, image_height)
  min_x = tf.clip_by_value(min_x, 0.0, image_width)
  max_y = tf.clip_by_value(max_y, 0.0, image_height)
  max_x = tf.clip_by_value(max_x, 0.0, image_width)
  return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y,
                     min_x,
                     max_y,
                     max_x,
                     image_height,
                     image_width,
                     delta=1):
  """Adjusts bbox coordinates to make sure the area is > 0.

    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
      delta: Float, this is used to create a gap of size 2 * delta between
        bbox min/max coordinates that are the same on the boundary.
        This prevents the bbox from having an area of zero.

    Returns:
      Tuple of new bbox coordinates between 0 and 1 that will now have a
      guaranteed area > 0.
    """
  height = max_y - min_y
  width = max_x - min_x

  def _adjust_bbox_boundaries(min_coord, max_coord, bound):
    # Make sure max is never 0 and min is never 1.
    max_coord = tf.maximum(max_coord, 0.0 + delta)
    min_coord = tf.minimum(min_coord, bound - delta)
    return min_coord, max_coord

  min_y, max_y = tf.cond(
      tf.equal(height, 0.0), lambda: _adjust_bbox_boundaries(
          min_y, max_y, image_height), lambda: (min_y, max_y))
  min_x, max_x = tf.cond(
      tf.equal(width, 0.0), lambda: _adjust_bbox_boundaries(
          min_x, max_x, image_width), lambda: (min_x, max_x))
  return min_y, min_x, max_y, max_x


def _scale_bbox_only_op_probability(prob):
  """Reduce the probability of the bbox-only operation.

    Probability is reduced so that we do not distort the content of too many
    bounding boxes that are close to each other. The value of 3.0 was a chosen
    hyper parameter when designing the autoaugment algorithm that we found
    empirically to work well.

    Args:
      prob: Float that is the probability of applying the bbox-only operation.

    Returns:
      Reduced probability.
    """
  return prob / 3.0


def _apply_bbox_augmentation(image, bbox, augmentation_func, *args):
  """Applies augmentation_func to the subsection of image indicated by bbox.

    Args:
      image: 3D uint8 Tensor.
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      augmentation_func: Augmentation function that will be applied to the
        subsection of image.
      *args: Additional parameters that will be passed into augmentation_func
        when it is called.

    Returns:
      A modified version of image, where the bbox location in the image will
      have `ugmentation_func applied to it.
    """
  # image_height = tf.cast(tf.shape(image)[0], tf.float32)
  # image_width = tf.cast(tf.shape(image)[1], tf.float32)
  # # todo
  # min_y = tf.cast(image_height * bbox[0], tf.int32)
  # min_x = tf.cast(image_width * bbox[1], tf.int32)
  # max_y = tf.cast(image_height * bbox[2], tf.int32)
  # max_x = tf.cast(image_width * bbox[3], tf.int32)
  # image_height = tf.cast(image_height, tf.int32)
  # image_width = tf.cast(image_width, tf.int32)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # todo
  min_y = tf.cast(bbox[0], tf.int32)
  min_x = tf.cast(bbox[1], tf.int32)
  max_y = tf.cast(bbox[2], tf.int32)
  max_x = tf.cast(bbox[3], tf.int32)

  # Clip to be sure the max values do not fall out of range.
  max_y = tf.minimum(max_y, image_height - 1)
  max_x = tf.minimum(max_x, image_width - 1)

  # Get the sub-tensor that is the image within the bounding box region.
  bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

  # Apply the augmentation function to the bbox portion of the image.
  augmented_bbox_content = augmentation_func(bbox_content, *args)

  # Pad the augmented_bbox_content and the mask to match the shape of original
  # image.
  augmented_bbox_content = tf.pad(
      augmented_bbox_content,
      [[min_y, (image_height-1) - max_y], [min_x,
                                           (image_width-1) - max_x], [0, 0]])

  # Create a mask that will be used to zero out a part of the original image.
  mask_tensor = tf.zeros_like(bbox_content)

  mask_tensor = tf.pad(
      mask_tensor,
      [[min_y, (image_height-1) - max_y], [min_x,
                                           (image_width-1) - max_x], [0, 0]],
      constant_values=1)
  # Replace the old bbox content with the new augmented content.
  image = image*mask_tensor + augmented_bbox_content
  return image


def _concat_bbox(bbox, bboxes):
  """Helper function that concates bbox to bboxes along the first dimension."""

  # Note if all elements in bboxes are -1 (_INVALID_BOX), then this means
  # we discard bboxes and start the bboxes Tensor with the current bbox.
  bboxes_sum_check = tf.reduce_sum(bboxes)
  bbox = tf.expand_dims(bbox, 0)
  # This check will be true when it is an _INVALID_BOX
  bboxes = tf.cond(
      tf.equal(bboxes_sum_check,
               -4.0), lambda: bbox, lambda: tf.concat([bboxes, bbox], 0))
  return bboxes


def _apply_bbox_augmentation_wrapper(image, bbox, new_bboxes, prob,
                                     augmentation_func, func_changes_bbox, *args):
  """Applies _apply_bbox_augmentation with probability prob.

    Args:
      image: 3D uint8 Tensor.
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      new_bboxes: 2D Tensor that is a list of the bboxes in the image after they
        have been altered by aug_func. These will only be changed when
        func_changes_bbox is set to true. Each bbox has 4 elements
        (min_y, min_x, max_y, max_x) of type float that are the normalized
        bbox coordinates between 0 and 1.
      prob: Float that is the probability of applying _apply_bbox_augmentation.
      augmentation_func: Augmentation function that will be applied to the
        subsection of image.
      func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
        to image.
      *args: Additional parameters that will be passed into augmentation_func
        when it is called.

    Returns:
      A tuple. Fist element is a modified version of image, where the bbox
      location in the image will have augmentation_func applied to it if it is
      chosen to be called with probability `prob`. The second element is a
      Tensor of Tensors of length 4 that will contain the altered bbox after
      applying augmentation_func.
    """
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  if func_changes_bbox:
    augmented_image, bbox = tf.cond(should_apply_op, lambda: augmentation_func(
        image, bbox, *args), lambda: (image, bbox))
  else:
    augmented_image = tf.cond(
        should_apply_op, lambda: _apply_bbox_augmentation(
            image, bbox, augmentation_func, *args), lambda: image)
  new_bboxes = _concat_bbox(bbox, new_bboxes)
  return augmented_image, new_bboxes


def _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                   func_changes_bbox, *args):
  """Applies aug_func to the image for each bbox in bboxes.

    Args:
      image: 3D uint8 Tensor.
      bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
        has 4 elements (min_y, min_x, max_y, max_x) of type float.
      prob: Float that is the probability of applying aug_func to a specific
        bounding box within the image.
      aug_func: Augmentation function that will be applied to the
        subsections of image indicated by the bbox values in bboxes.
      func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
        to image.
      *args: Additional parameters that will be passed into augmentation_func
        when it is called.

    Returns:
      A modified version of image, where each bbox location in the image will
      have augmentation_func applied to it if it is chosen to be called with
      probability prob independently across all bboxes. Also the final
      bboxes are returned that will be unchanged if func_changes_bbox is set to
      false and if true, the new altered ones will be returned.
    """
  # Will keep track of the new altered bboxes after aug_func is repeatedly
  # applied. The -1 values are a dummy value and this first Tensor will be
  # removed upon appending the first real bbox.
  new_bboxes = tf.constant(_INVALID_BOX)

  # If the bboxes are empty, then just give it _INVALID_BOX. The result
  # will be thrown away.
  bboxes = tf.cond(
      tf.equal(tf.size(bboxes),
               0), lambda: tf.constant(_INVALID_BOX), lambda: bboxes)

  bboxes = tf.ensure_shape(bboxes, (None, 4))

  # pylint:disable=g-long-lambda
  # pylint:disable=line-too-long
  def wrapped_aug_func(_image, bbox, _new_bboxes):
    return _apply_bbox_augmentation_wrapper(_image, bbox, _new_bboxes, prob,
                                            aug_func, func_changes_bbox, *args)

  # pylint:enable=g-long-lambda
  # pylint:enable=line-too-long

  # Setup the while_loop.
  num_bboxes = tf.shape(bboxes)[0]  # We loop until we go over all bboxes.
  idx = tf.constant(0)  # Counter for the while loop.

  # Conditional function when to end the loop once we go over all bboxes
  # images_and_bboxes contain (_image, _new_bboxes)
  def cond(_idx, _images_and_bboxes):
    return tf.less(_idx, num_bboxes)

  # Shuffle the bboxes so that the augmentation order is not deterministic if
  # we are not changing the bboxes with aug_func.
  if not func_changes_bbox:
    loop_bboxes = tf.random.shuffle(bboxes)
  else:
    loop_bboxes = bboxes

  # Main function of while_loop where we repeatedly apply augmentation on the
  # bboxes in the image.
  # pylint:disable=g-long-lambda
  def body(_idx, _images_and_bboxes):
    return [
        _idx + 1,
        wrapped_aug_func(_images_and_bboxes[0], loop_bboxes[_idx],
                         _images_and_bboxes[1])
    ]

  # pylint:enable=g-long-lambda

  _, (image, new_bboxes) = tf.while_loop(
      cond,
      body, [idx, (image, new_bboxes)],
      shape_invariants=[
          idx.get_shape(), (image.get_shape(), tf.TensorShape([None, 4]))
      ])

  # Either return the altered bboxes or the original ones depending on if
  # we altered them in anyway.
  if func_changes_bbox:
    final_bboxes = new_bboxes
  else:
    final_bboxes = bboxes
  return image, final_bboxes


def _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, aug_func,
                                           func_changes_bbox, *args):
  """Checks to be sure num bboxes > 0 before calling inner function."""
  num_bboxes = tf.shape(bboxes)[0]
  image, bboxes = tf.cond(
      tf.equal(num_bboxes, 0),
      lambda: (image, bboxes),
      # pylint:disable=g-long-lambda
      lambda: _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                             func_changes_bbox, *args))
  # pylint:enable=g-long-lambda
  return image, bboxes


def rotate_only_bboxes(image, bboxes, prob, degrees, replace):
  """Apply rotate to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, rotate,
                                                func_changes_bbox, degrees,
                                                replace)


def shear_x_only_bboxes(image, bboxes, prob, level, replace):
  """Apply shear_x to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, shear_x,
                                                func_changes_bbox, level, replace)


def shear_y_only_bboxes(image, bboxes, prob, level, replace):
  """Apply shear_y to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, shear_y,
                                                func_changes_bbox, level, replace)


def translate_x_only_bboxes(image, bboxes, prob, pixels, replace):
  """Apply translate_x to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, translate_x,
                                                func_changes_bbox, pixels,
                                                replace)


def translate_y_only_bboxes(image, bboxes, prob, pixels, replace):
  """Apply translate_y to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, translate_y,
                                                func_changes_bbox, pixels,
                                                replace)


def flip_only_bboxes(image, bboxes, prob):
  """Apply flip_lr to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob,
                                                tf.image.flip_left_right,
                                                func_changes_bbox)


def solarize_only_bboxes(image, bboxes, prob, threshold):
  """Apply solarize to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, solarize,
                                                func_changes_bbox, threshold)


def equalize_only_bboxes(image, bboxes, prob):
  """Apply equalize to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, equalize,
                                                func_changes_bbox)


def cutout_only_bboxes(image, bboxes, prob, pad_size, replace):
  """Apply cutout to each bbox in the image with probability prob."""
  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, cutout,
                                                func_changes_bbox, pad_size,
                                                replace)


def _rotate_bbox(bbox, image_height, image_width, degrees):
  """Rotates the bbox coordinated by degrees.

    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: Int, height of the image.
      image_width: Int, height of the image.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.

    Returns:
      A tensor of the same shape as bbox, but now with the rotated coordinates.
    """
  image_height, image_width = (tf.cast(image_height, tf.float32),
                               tf.cast(image_width, tf.float32))

  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # Translate the bbox to the center of the image and turn the normalized 0-1
  # coordinates to absolute pixel locations.
  # Y coordinates are made negative as the y axis of images goes down with
  # increasing pixel values, so we negate to make sure x axis and y axis points
  # are in the traditionally positive direction.
  min_y = -tf.cast(bbox[0] - 0.5*image_height, tf.int32)
  min_x = tf.cast(bbox[1] - 0.5*image_width, tf.int32)
  max_y = -tf.cast(bbox[2] - 0.5*image_height, tf.int32)
  max_x = tf.cast(bbox[3] - 0.5*image_width, tf.int32)
  coordinates = tf.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                          [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)
  # Rotate the coordinates according to the rotation matrix clockwise if
  # radians is positive, else negative
  rotation_matrix = tf.stack([[tf.cos(radians), tf.sin(radians)],
                              [-tf.sin(radians),
                               tf.cos(radians)]])
  new_coords = tf.cast(
      tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to normalized 0-1 floats.
  min_y = -(
      tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) - 0.5*image_height)
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) + 0.5*image_width
  max_y = -(
      tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) - 0.5*image_height)
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) + 0.5*image_width

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x,
                                          image_height, image_width)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x,
                                                image_height, image_width)
  return tf.stack([min_y, min_x, max_y, max_x])


def rotate_with_bboxes(image, bboxes, degrees, replace):
  """Equivalent of PIL Rotate that rotates the image and bbox.

    Args:
      image: 3D uint8 Tensor.
      bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
        has 4 elements (min_y, min_x, max_y, max_x) of type float.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
      A tuple containing a 3D uint8 Tensor that will be the result of rotating
      image by degrees. The second element of the tuple is bboxes, where now
      the coordinates will be shifted to reflect the rotated image.
    """
  # Rotate the image.
  image = rotate(image, degrees, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # pylint:disable=g-long-lambda
  def wrapped_rotate_bbox(bbox):
    return _rotate_bbox(bbox, image_height, image_width, degrees)

  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
  return image, bboxes


def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  image = tfa.image.translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = tfa.image.translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
  """Shifts the bbox coordinates by pixels.

    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: float, height of the image.
      image_width: float, width of the image.
      pixels: An int. How many pixels to shift the bbox.
      shift_horizontal: Boolean. If true then shift in X dimension else shift in
        Y dimension.

    Returns:
      A tensor of the same shape as bbox, but now with the shifted coordinates.
    """
  pixels = tf.cast(pixels, tf.float32)
  # Convert bbox to integer pixel locations.

  min_y = bbox[0]
  min_x = bbox[1]
  max_y = bbox[2]
  max_x = bbox[3]

  if shift_horizontal:
    min_x = tf.maximum(0., min_x - pixels)
    max_x = tf.minimum(image_width, max_x - pixels)
  else:
    min_y = tf.maximum(0., min_y - pixels)
    max_y = tf.minimum(image_height, max_y - pixels)

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x,
                                          image_height, image_width)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x,
                                                image_height, image_width)
  return tf.stack([min_y, min_x, max_y, max_x])


def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
  """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.

    Args:
      image: 3D uint8 Tensor.
      bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
        has 4 elements (min_y, min_x, max_y, max_x) of type float with values
        between [0, 1].
      pixels: An int. How many pixels to shift the image and bboxes
      replace: A one or three value 1D tensor to fill empty pixels.
      shift_horizontal: Boolean. If true then shift in X dimension else shift in
        Y dimension.

    Returns:
      A tuple containing a 3D uint8 Tensor that will be the result of translating
      image by pixels. The second element of the tuple is bboxes, where now
      the coordinates will be shifted to reflect the shifted image.
    """
  if shift_horizontal:
    image = translate_x(image, pixels, replace)
  else:
    image = translate_y(image, pixels, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)

  # pylint:disable=g-long-lambda
  def wrapped_shift_bbox(bbox):
    return _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)

  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shift_bbox, bboxes)
  return image, bboxes


def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = tfa.image.transform(wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = tfa.image.transform(wrap(image), [1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
  """Shifts the bbox according to how the image was sheared.

    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: Int, height of the image.
      image_width: Int, height of the image.
      level: Float. How much to shear the image.
      shear_horizontal: If true then shear in X dimension else shear in
        the Y dimension.

    Returns:
      A tensor of the same shape as bbox, but now with the shifted coordinates.
    """
  # todo
  min_y = tf.cast(bbox[0], tf.int32)
  min_x = tf.cast(bbox[1], tf.int32)
  max_y = tf.cast(bbox[2], tf.int32)
  max_x = tf.cast(bbox[3], tf.int32)
  coordinates = tf.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                          [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)

  # Shear the coordinates according to the translation matrix.
  if shear_horizontal:
    translation_matrix = tf.stack([[1, 0], [-level, 1]])
  else:
    translation_matrix = tf.stack([[1, -level], [0, 1]])
  translation_matrix = tf.cast(translation_matrix, tf.float32)
  new_coords = tf.cast(
      tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.float32)

  # Find min/max values and convert them back to floats.
  min_y = tf.reduce_min(new_coords[0, :])
  min_x = tf.reduce_min(new_coords[1, :])
  max_y = tf.reduce_max(new_coords[0, :])
  max_x = tf.reduce_max(new_coords[1, :])

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x,
                                          image_height, image_width)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x,
                                                image_height, image_width)
  return tf.stack([min_y, min_x, max_y, max_x])


def shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
  """Applies Shear Transformation to the image and shifts the bboxes.

    Args:
      image: 3D uint8 Tensor.
      bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
        has 4 elements (min_y, min_x, max_y, max_x) of type float with values
        between [0, 1].
      level: Float. How much to shear the image. This value will be between
        -0.3 to 0.3.
      replace: A one or three value 1D tensor to fill empty pixels.
      shear_horizontal: Boolean. If true then shear in X dimension else shear in
        the Y dimension.

    Returns:
      A tuple containing a 3D uint8 Tensor that will be the result of shearing
      image by level. The second element of the tuple is bboxes, where now
      the coordinates will be shifted to reflect the sheared image.
    """
  if shear_horizontal:
    image = shear_x(image, level, replace)
  else:
    image = shear_y(image, level, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)

  # pylint:disable=g-long-lambda
  def wrapped_shear_bbox(bbox):
    return _shear_bbox(bbox, image_height, image_width, level, shear_horizontal)

  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shear_bbox, bboxes)
  return image, bboxes


def autocontrast(image: tf.Tensor) -> tf.Tensor:
  """Implements Autocontrast function from PIL using TF ops."""

  def scale_channel(channel):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(channel), tf.float32)
    hi = tf.cast(tf.reduce_max(channel), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi-lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      return tf.saturate_cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(channel), lambda: channel)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def autocontrast_blend(image: tf.Tensor, factor) -> tf.Tensor:
  """Implements blend of autocontrast with original image."""
  return blend(autocontrast(image), image, factor)


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                       dtype=tf.float32,
                       shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]

  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step//2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step,
                 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def equalize_blend(image, factor):
  """Implements blend of equalize with original image."""
  return blend(equalize(image), image, factor)


def _convolve_image_with_kernel(image, kernel):
  num_channels = tf.shape(image)[-1]
  kernel = tf.tile(kernel, [1, 1, num_channels, 1])
  image = tf.expand_dims(image, axis=0)
  convolved_im = tf.nn.depthwise_conv2d(
      tf.cast(image, tf.float32), kernel, strides=[1, 1, 1, 1], padding='SAME')
  # adding 0.5 for future rounding, same as in PIL:
  # https://github.com/python-pillow/Pillow/blob/555e305a60d7fcefd1ad4aa6c8fd879e2f474192/src/libImaging/Filter.c#L101  # pylint: disable=line-too-long
  convolved_im = convolved_im + 0.5
  return tf.squeeze(convolved_im, axis=0)


def blur(image, factor):
  """Blur with the same kernel as ImageFilter.BLUR."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class BLUR(BuiltinFilter):
  #     name = "Blur"
  #     # fmt: off
  #     filterargs = (5, 5), 16, 0, (
  #         1, 1, 1, 1, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 1, 1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  blur_kernel = tf.constant(
      [[1., 1., 1., 1., 1.], [1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.],
       [1., 0., 0., 0., 1.], [1., 1., 1., 1., 1.]],
      dtype=tf.float32,
      shape=[5, 5, 1, 1]) / 16.0
  blurred_im = _convolve_image_with_kernel(image, blur_kernel)
  return blend(image, blurred_im, factor)


def smooth(image, factor):
  """Smooth with the same kernel as ImageFilter.SMOOTH."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class SMOOTH(BuiltinFilter):
  #     name = "Smooth"
  #     # fmt: off
  #     filterargs = (3, 3), 13, 0, (
  #         1, 1, 1,
  #         1, 5, 1,
  #         1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  smooth_kernel = tf.constant([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]],
                              dtype=tf.float32,
                              shape=[3, 3, 1, 1]) / 13.0
  smoothed_im = _convolve_image_with_kernel(image, smooth_kernel)
  return blend(image, smoothed_im, factor)


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image: tf.Tensor, replace=[128, 128, 128]) -> tf.Tensor:
  """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
      image: A 3D Image Tensor with 4 channels.
      replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
      image: A 3D image Tensor with 3 channels.
    """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = flattened_image[:, 3]

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0)[..., None],
      tf.ones_like(flattened_image, dtype=image.dtype) * replace, flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _cutout_inside_bbox(image, bbox, pad_fraction):
  """Generates cutout mask and the mean pixel value of the bbox.

    First a location is randomly chosen within the image as the center where the
    cutout mask will be applied. Note this can be towards the boundaries of the
    image, so the full cutout mask may not be applied.

    Args:
      image: 3D uint8 Tensor.
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      pad_fraction: Float that specifies how large the cutout mask should be in
        in reference to the size of the original bbox. If pad_fraction is 0.25,
        then the cutout mask will be of shape
        (0.25 * bbox height, 0.25 * bbox width).

    Returns:
      A tuple. Fist element is a tensor of the same shape as image where each
      element is either a 1 or 0 that is used to determine where the image
      will have cutout applied. The second element is the mean of the pixels
      in the image where the bbox is located.
    """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # Transform from shape [1, 4] to [4].
  bbox = tf.squeeze(bbox)
  min_y = tf.cast(bbox[0], tf.int32)
  min_x = tf.cast(bbox[1], tf.int32)
  max_y = tf.cast(bbox[2], tf.int32)
  max_x = tf.cast(bbox[3], tf.int32)

  # Calculate the mean pixel values in the bounding box, which will be used
  # to fill the cutout region.
  mean = tf.reduce_mean(image[min_y:max_y + 1, min_x:max_x + 1], axis=[0, 1])

  # Cutout mask will be size pad_size_heigh * 2 by pad_size_width * 2 if the
  # region lies entirely within the bbox.
  box_height = max_y - min_y + 1
  box_width = max_x - min_x + 1
  pad_size_height = tf.cast(pad_fraction * (box_height/2), tf.int32)
  pad_size_width = tf.cast(pad_fraction * (box_width/2), tf.int32)

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=min_y, maxval=max_y + 1, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=min_x, maxval=max_x + 1, dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size_height)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size_height)
  left_pad = tf.maximum(0, cutout_center_width - pad_size_width)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size_width)

  cutout_shape = [
      image_height - (lower_pad+upper_pad), image_width - (left_pad+right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)

  mask = tf.expand_dims(mask, 2)
  mask = tf.tile(mask, [1, 1, 3])

  return mask, mean


def bbox_cutout(image, bboxes, pad_fraction, replace_with_mean):
  """Applies cutout to the image according to bbox information.

    This is a cutout variant that using bbox information to make more informed
    decisions on where to place the cutout mask.

    Args:
      image: 3D uint8 Tensor.
      bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
        has 4 elements (min_y, min_x, max_y, max_x) of type float with values
        between [0, 1].
      pad_fraction: Float that specifies how large the cutout mask should be in
        in reference to the size of the original bbox. If pad_fraction is 0.25,
        then the cutout mask will be of shape
        (0.25 * bbox height, 0.25 * bbox width).
      replace_with_mean: Boolean that specified what value should be filled in
        where the cutout mask is applied. Since the incoming image will be of
        uint8 and will not have had any mean normalization applied, by default
        we set the value to be 128. If replace_with_mean is True then we find
        the mean pixel values across the channel dimension and use those to fill
        in where the cutout mask is applied.

    Returns:
      A tuple. First element is a tensor of the same shape as image that has
      cutout applied to it. Second element is the bboxes that were passed in
      that will be unchanged.
    """

  def apply_bbox_cutout(image, bboxes, pad_fraction):
    """Applies cutout to a single bounding box within image."""
    # Choose a single bounding box to apply cutout to.
    random_index = tf.random.uniform(
        shape=[], maxval=tf.shape(bboxes)[0], dtype=tf.int32)
    # Select the corresponding bbox and apply cutout.
    chosen_bbox = tf.gather(bboxes, random_index)
    mask, mean = _cutout_inside_bbox(image, chosen_bbox, pad_fraction)

    # When applying cutout we either set the pixel value to 128 or to the mean
    # value inside the bbox.
    replace = mean if replace_with_mean else 128

    # Apply the cutout mask to the image. Where the mask is 0 we fill it with
    # `replace`.
    image = tf.where(
        tf.equal(mask, 0),
        tf.cast(
            tf.ones_like(image, dtype=image.dtype) * replace, dtype=image.dtype),
        image)
    return image

  # Check to see if there are boxes, if so then apply boxcutout.
  image = tf.cond(
      tf.equal(tf.size(bboxes), 0), lambda: image, lambda: apply_bbox_cutout(
          image, bboxes, pad_fraction))

  return image, bboxes


def normalize(image: tf.Tensor, mean: float = 0.5, std: float = 0.5) -> tf.Tensor:
  return (image-mean) / std


def renormalize(image: tf.Tensor, mean: float = 0.5,
                std: float = 0.5) -> tf.Tensor:
  return image*std + mean
