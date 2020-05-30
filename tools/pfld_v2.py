import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug import KeypointsOnImage
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tools.base import BaseHelper, INFO, ERROR, NOTE
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from tools.pfld import calculate_pitch_yaw_roll


class PFLDV2Helper(BaseHelper):
  def __init__(self, image_ann: str, in_hw: tuple, landmark_num: int,
               attribute_num: int, validation_split=0.1,
               num_parallel_calls: int = -1):
    self.in_hw = np.array(in_hw)
    assert self.in_hw.ndim == 1
    self.landmark_num = landmark_num
    self.attribute_num = attribute_num
    self.validation_split = validation_split  # type:float
    if image_ann == None:
      self.train_list = None
      self.test_list = None
      self.val_list = None
    else:
      img_ann_list: dict = np.load(image_ann, allow_pickle=True)[()]
      self.train_list: Tuple[np.ndarray, np.ndarray, np.ndarray] = img_ann_list['train_list']
      self.test_list: Tuple[np.ndarray, np.ndarray, np.ndarray] = img_ann_list['test_list']
      self.val_list = img_ann_list.get('val_list', None)
      if self.val_list == None:
        self.val_list = self.test_list

      self.train_total_data: int = img_ann_list['train_num']
      self.test_total_data: int = img_ann_list['test_num']
      self.val_total_data = img_ann_list.get('val_num', None)
      if self.val_total_data == None:
        self.val_total_data = self.test_total_data

    self.iaaseq: iaa.Augmenter = iaa.OneOf([
        iaa.Fliplr(0.5),  # 50% 镜像
        iaa.Affine(rotate=(-10, 10)),  # 随机旋转
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
    ])
    self.num_parallel_calls = num_parallel_calls
    self.TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

  def resize_img(self, raw_img: tf.Tensor) -> tf.Tensor:
    return tf.image.resize(raw_img, self.in_hw, tf.image.ResizeMethod.BILINEAR, antialias=True)

  def read_img(self, img_path):
    image = tf.io.read_file(img_path)
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.decode_png(image, channels=3))
    return image

  def augment_img(self, img, landmark):
    img, landmark = self.iaaseq(image=img, keypoints=landmark[None, ...])
    landmark = landmark[0]
    landmark = np.maximum(0,
                          np.minimum(img.shape[1::-1], landmark, dtype=np.float32),
                          dtype=np.float32)
    return img, landmark

  @staticmethod
  def crop_img(img, im_hw, landmark, scale: float = 1.2):
    xy = tf.reduce_min(landmark, axis=0)
    zz = tf.reduce_max(landmark, axis=0)
    wh = zz - xy + 1
    center = tf.cast(xy + wh / 2, tf.int32)
    boxsize = tf.cast(tf.reduce_max(wh) * scale, tf.int32)
    xy = tf.cast(center - boxsize // 2, tf.int32)
    x1, y1 = xy[0], xy[1]
    x2, y2 = x1 + boxsize, y1 + boxsize
    height, width = im_hw[0], im_hw[1]
    dx = tf.maximum(0, -x1)
    dy = tf.maximum(0, -y1)
    x1 = tf.maximum(0, x1)
    y1 = tf.maximum(0, y1)
    edx = tf.maximum(0, x2 - width)
    edy = tf.maximum(0, y2 - height)
    x2 = tf.minimum(width, x2)
    y2 = tf.minimum(height, y2)
    nh, nw = y2 - y1, x2 - x1
    imgT = tf.image.crop_to_bounding_box(img, y1, x1, nh, nw)
    # imgT = img[y1:y2, x1:x2]
    if tf.reduce_any(tf.stack([dx, dy, edx, edy]) > 0):
      imgT = tf.image.pad_to_bounding_box(imgT, dy, dx, dy + nh + edy, dx + nw + edx)
      # imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

    return imgT, landmark - tf.cast(xy, tf.float32), [nh, nw]

  def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int, is_augment: bool,
                     is_normlize: bool, is_training: bool) -> tf.data.Dataset:
    print(INFO, 'data augment is ', str(is_augment))

    def _parser_wrapper(path, im_hw, landmark, attr) -> [tf.Tensor, tf.Tensor]:
      raw_img = self.read_img(path)
      if is_augment == True:
        raw_img, landmark, new_hw = self.crop_img(
            raw_img, im_hw, landmark, scale=tf.random.uniform([], 1., 1.4))
        raw_img, landmark = tf.numpy_function(
            self.augment_img, [raw_img, landmark], [tf.uint8, tf.float32])

      else:
        raw_img, landmark, new_hw = self.crop_img(raw_img, im_hw, landmark, scale=1.2)

      raw_img.set_shape([None, None, 3])
      landmark.set_shape([self.landmark_num, 2])

      eular = tf.numpy_function(calculate_pitch_yaw_roll, [
                                tf.gather(landmark, self.TRACKED_POINTS)], tf.float32)

      # NOTE recale landmark to 0~1
      landmark = landmark / tf.cast(new_hw[::-1], tf.float32)
      raw_img = self.resize_img(raw_img)

      if is_normlize == True:
        # NOTE normlized image
        img = self.normlize_img(raw_img)
      else:
        img = tf.cast(raw_img, tf.float32)
      # img [h,w,3] landmark [68,2] attr [6] eular [3]
      return img, landmark, attr, eular

    def _batch_parser(img: tf.Tensor, landmark, attr, eular) -> [
            tf.Tensor, tf.Tensor]:
      """ 
          process ann , calc the attribute weights 

          return : img , [landmarks-attribute_weight-euluar]
      """
      mat_ratio = tf.reduce_mean(attr, axis=0, keepdims=True)
      mat_ratio = tf.where(mat_ratio > 0, 1. / mat_ratio,
                           tf.ones([1, self.attribute_num]) * batch_size)
      attribute_weight = tf.matmul(attr, mat_ratio, transpose_b=True)  # [n,1]
      # NOTE avoid when image don't no special attribute,the loss's weight is 0
      attribute_weight = tf.where(attribute_weight == 0,
                                  tf.ones_like(attribute_weight),
                                  attribute_weight)

      labels = tf.concat([
          tf.reshape(landmark, [self.batch_size, -1]),
          attribute_weight,
          eular], -1)
      # set shape
      img.set_shape([None, self.in_hw[0], self.in_hw[1], 3])
      labels.set_shape([None, self.landmark_num * 2 + 1 + 3])

      return img, labels

    if is_training:
      ds = (tf.data.Dataset.from_tensor_slices(self.train_list)
            .shuffle(batch_size * 500)
            .repeat()
            .map(_parser_wrapper, self.num_parallel_calls)
            .batch(batch_size, True)
            .map(_batch_parser, -1)
            .prefetch(-1))
    else:
      ds = (tf.data.Dataset.from_tensor_slices(self.test_list)
            .map(_parser_wrapper, -1)
            .batch(batch_size, True)
            .map(_batch_parser, -1)
            .prefetch(-1))

    return ds

  def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True):
    """ draw img and show bbox , set ann = None will not show bbox

    Parameters
    ----------
    img : np.ndarray

    ann : np.ndarray

       shape : [p,x,y,w,h]

    is_show : bool

        show image
    """
    if ann.ndim == 2:
      landmark = ann
    else:
      landmark, attribute, euler = np.split(
          ann, [self.landmark_num * 2,
                self.landmark_num * 2 + self.attribute_num])

      landmark = landmark.reshape(-1, 2) * img.shape[1::-1]

    for (x, y) in landmark.astype('uint32'):
      cv2.circle(img, (x, y), 3, (255, 0, 0), 1)
    plt.imshow(img)
    if is_show:
      plt.show()
