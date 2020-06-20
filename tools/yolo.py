import numpy as np
import os
import cv2
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from math import cos, sin
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
from scipy.special import expit
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras as k
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from matplotlib.pyplot import text
from PIL import Image, ImageFont, ImageDraw
from tools.base import BaseHelper, INFO, ERROR, NOTE
from tools.bbox_utils import bbox_iou, center_to_corner, tf_bbox_iou, nms_oneclass
from tools.custom import focal_sigmoid_cross_entropy_with_logits
from pathlib import Path
import shutil
from tqdm import trange, tqdm
from termcolor import colored
from typing import List, Tuple, AnyStr, Iterable
from more_itertools import chunked


def fake_iou(a: np.ndarray, b: np.ndarray) -> float:
  """set a,b center to same,then calc the iou value

  Parameters
  ----------
  a : np.ndarray
      shape = [n,1,2]
  b : np.ndarray
      shape = [m,2]

  Returns
  -------
  float
      iou value
      shape = [n,m]
  """
  a_maxes = a / 2.
  a_mins = -a_maxes

  b_maxes = b / 2.
  b_mins = -b_maxes

  iner_mins = np.maximum(a_mins, b_mins)
  iner_maxes = np.minimum(a_maxes, b_maxes)
  iner_wh = np.maximum(iner_maxes - iner_mins, 0.)
  iner_area = iner_wh[..., 0] * iner_wh[..., 1]

  s1 = a[..., 0] * a[..., 1]
  s2 = b[..., 0] * b[..., 1]

  return iner_area / (s1 + s2 - iner_area)


def coordinate_offset(anchors: np.ndarray, out_hw: np.ndarray) -> np.array:
  """construct the anchor coordinate offset array , used in convert scale

  Parameters
  ----------
  anchors : np.ndarray
      anchors shape = [n,] = [ n x [m,2]]
  out_hw : np.ndarray
      output height width shape = [n,2]

  Returns
  -------
  np.array
      scale shape = [n,] = [n x [h_n,w_n,m,2]]
  """
  if len(anchors) != len(out_hw):
    raise ValueError(f'anchors len {len(anchors)} is not equal out_hw len {len(out_hw)}')
  grid = []
  for l in range(len(anchors)):
    grid_y = np.tile(np.reshape(
        np.arange(0, stop=out_hw[l][0]), [-1, 1, 1, 1]), [1, out_hw[l][1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=out_hw[l][1]), [
                     1, -1, 1, 1]), [out_hw[l][0], 1, 1, 1])
    grid.append(np.concatenate([grid_x, grid_y], axis=-1))
  return np.array(grid)


def bbox_crop(ann: np.ndarray, crop_box=None, allow_outside_center=True) -> np.ndarray:
  """ Crop bounding boxes according to slice area.

  Parameters
  ----------
  ann : np.ndarray

      (n,5) [p,x1,y1,x2,y1]

  crop_box : optional

      crop_box [x1,y1,x2,y2] , by default None

  allow_outside_center : bool, optional

      by default True

  Returns
  -------
  np.ndarray

      ann
      Cropped bounding boxes with shape (M, 4+) where M <= N.
  """
  ann = ann.copy()
  if crop_box is None:
    return ann
  if sum([int(c is None) for c in crop_box]) == 4:
    return ann

  l, t, r, b = crop_box

  left = l if l else 0
  top = t if t else 0
  right = r if r else np.inf
  bottom = b if b else np.inf
  crop_bbox = np.array((left, top, right, bottom))

  if allow_outside_center:
    mask = np.ones(ann.shape[0], dtype=bool)
  else:
    centers = (ann[:, 1:3] + ann[:, 3:5]) / 2
    mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

  # transform borders
  ann[:, 1:3] = np.maximum(ann[:, 1:3], crop_bbox[:2])
  ann[:, 3:5] = np.minimum(ann[:, 3:5], crop_bbox[2:4])
  ann[:, 1:3] -= crop_bbox[:2]
  ann[:, 3:5] -= crop_bbox[:2]

  mask = np.logical_and(mask, (ann[:, 1:3] < ann[:, 3:5]).all(axis=1))
  ann = ann[mask]
  return ann


def bbox_crop_constraints(ann: np.ndarray, im_wh: np.ndarray,
                          min_scale: float = 0.3, max_scale: float = 1,
                          max_aspect_ratio: float = 2,
                          max_trial: float = 50,
                          constraints: float = None) -> [np.ndarray, list]:
  """
      Crop an image randomly with bounding box constraints.

      [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
     Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
     SSD: Single Shot MultiBox Detector. ECCV 2016.

  Parameters
  ----------
  ann : np.ndarray

      (n,5) [p,x1,y1,x2,y1]

  im_wh : np.ndarray

      im wh

  min_scale : float, optional

      The minimum ratio between a cropped region and the original image. by default 0.3

  max_scale : float, optional

      The maximum ratio between a cropped region and the original image. by default 1

  max_aspect_ratio : float, optional

      The maximum aspect ratio of cropped region. by default 2

  constraints : float, optional

      by default None

  max_trial : float, optional

      Maximum number of trials for each constraint before exit no matter what. by default 50

  Returns
  -------
  [np.ndarray, list]

      new ann, crop idx : (0, 0, w, h)

  """
  # default params in paper
  if constraints is None:
    constraints = (
        (0.1, None),
        (0.3, None),
        (0.5, None),
        (0.7, None),
        (0.9, None),
        (None, 1),
    )

  w, h = im_wh

  candidates = [(0, 0, w, h)]
  for min_iou, max_iou in constraints:
    min_iou = -np.inf if min_iou is None else min_iou
    max_iou = np.inf if max_iou is None else max_iou

    for _ in range(max_trial):
      scale = np.random.uniform(min_scale, max_scale)
      aspect_ratio = np.random.uniform(
          max(1 / max_aspect_ratio, scale * scale),
          min(max_aspect_ratio, 1 / (scale * scale)))
      crop_h = int(h * scale / np.sqrt(aspect_ratio))
      crop_w = int(w * scale * np.sqrt(aspect_ratio))

      crop_t = np.random.randint(h - crop_h)
      crop_l = np.random.randint(w - crop_w)
      crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

      if len(ann) == 0:
        top, bottom = crop_t, crop_t + crop_h
        left, right = crop_l, crop_l + crop_w
        return ann, np.array((left, top, right, bottom), 'int32')

      iou = bbox_iou(ann[:, 1:], crop_bb[np.newaxis])
      if min_iou <= iou.min() and iou.max() <= max_iou:
        top, bottom = crop_t, crop_t + crop_h
        left, right = crop_l, crop_l + crop_w
        candidates.append((left, top, right, bottom))
        break

  # random select one
  while candidates:
    crop = candidates.pop(np.random.randint(0, len(candidates)))
    new_ann = bbox_crop(ann, crop, allow_outside_center=False)
    if new_ann.size < 1:
      continue
    new_crop = (crop[0], crop[1], crop[2], crop[3])
    return new_ann, np.array(new_crop, 'int32')
  return ann, np.array((0, 0, w, h), 'int32')


class YOLOHelper(BaseHelper):
  def __init__(self, image_ann: str, class_num: int, anchors: str,
               in_hw: tuple, out_hw: tuple,
               resize_method: str = 'origin',
               augment_method: str = 'origin',
               num_parallel_calls: int = -1):
    """ yolo helper

    Parameters
    ----------
    image_ann : str

        image ann `.npy` file path

    class_num : int

    anchors : str

        anchor `.npy` file path

    in_hw : tuple

        default input image height width

    out_hw : tuple

        default output height width

    resize_method : str, optional

        train image resize method, ['origin','gulon','none']

    augment_method : str, optional

        train image augment method, ['origin','iaa']

    num_parallel_calls : int, optional

        tf.dataset.map(num_parallel_calls)

    """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    if image_ann == None:
      self.train_list: np.ndarray = None
      self.val_list: np.ndarray = None
      self.test_list: np.ndarray = None
    else:
      img_ann_list = np.load(image_ann, allow_pickle=True)[()]

      self.train_list: str = img_ann_list['train_data']
      self.val_list: str = img_ann_list['val_data']
      self.test_list: str = img_ann_list['test_data']
      self.train_total_data: int = img_ann_list['train_num']
      self.val_total_data: int = img_ann_list['val_num']
      self.test_total_data: int = img_ann_list['test_num']
    self.resize_method = resize_method
    self.augment_method = augment_method
    self.num_parallel_calls = num_parallel_calls
    self.org_in_hw = np.array(in_hw)
    self.org_out_hw = np.array(out_hw)
    assert self.org_in_hw.ndim == 1
    assert self.org_out_hw.ndim == 2
    self.in_hw = tf.Variable(self.org_in_hw, trainable=False)
    self.out_hw = tf.Variable(self.org_out_hw, trainable=False)
    if class_num:
      self.class_num = class_num  # type:int
    if anchors:
      self.anchors = np.load(anchors)  # type:np.ndarray
      self.anchor_number = len(self.anchors[0])
      self.output_number = len(self.anchors)
      self.__flatten_anchors = np.reshape(self.anchors, (-1, 2))
      self.xy_offsets: List[ResourceVariable] = [tf.Variable(self.calc_xy_offset(self.out_hw[i]), trainable=False)
                                                 for i in range(self.output_number)]

    self.iaaseq = iaa.Sequential([
        iaa.SomeOf([1, None], [
            iaa.MultiplyHueAndSaturation(mul_hue=(0.7, 1.3), mul_saturation=(0.7, 1.3),
                                         per_channel=True),
            iaa.Multiply((0.5, 1.5), per_channel=True),
            iaa.SigmoidContrast((3, 8)),
        ], True),
        iaa.SomeOf([1, None], [
            iaa.Fliplr(0.5),
            iaa.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                       backend='cv2'),
            iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                       backend='cv2'),
            iaa.Affine(rotate=(-15, 15),
                       backend='cv2')
        ], True)
    ], True)

    self.colormap = [
        (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
        (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
        (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
        (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
        (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
        (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214),
        (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
        (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), (255, 122, 8),
        (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
        (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), (20, 255, 0),
        (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
        (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200)]

  def _xy_grid_index(self, out_hw: np.ndarray, box_xy: np.ndarray, layer: int) -> [np.ndarray, np.ndarray]:
    """ get xy index in grid scale

    Parameters
    ----------
    box_xy : np.ndarray
        value = [x,y]
    layer  : int
        layer index

    Returns
    -------
    [np.ndarray,np.ndarray]

        index xy : = [idx,idy]
    """
    return np.floor(box_xy * out_hw[layer][:: -1]).astype('int')

  def _get_anchor_index(self, wh: np.ndarray) -> [np.ndarray, np.ndarray]:
    """get the max iou anchor index

    Parameters
    ----------
    wh : np.ndarray
        shape = [num_box,2]
        value = [w,h]

    Returns
    -------
    np.ndarray, np.ndarray
        max iou anchor index
        layer_idx shape = [num_box, num_anchor]
        anchor_idx shape = [num_box, num_anchor]
    """
    iou = fake_iou(np.expand_dims(wh, -2), self.__flatten_anchors)
    # sort iou score in decreasing order
    best_anchor = np.argsort(-iou, -1)  # shape = [num_box, num_anchor]
    return np.divmod(best_anchor, self.anchor_number)

  @staticmethod
  def parser_example(stream: bytes) -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """ parser yolo tfrecord example

    Parameters
    ----------
    stream : bytes

    Returns
    -------
    [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        img_str, img_name, ann, img_hw
    """
    features = tf.io.parse_single_example(stream, {
        'img': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.VarLenFeature(tf.float32),
        'x1': tf.io.VarLenFeature(tf.float32),
        'y1': tf.io.VarLenFeature(tf.float32),
        'x2': tf.io.VarLenFeature(tf.float32),
        'y2': tf.io.VarLenFeature(tf.float32),
        'img_hw': tf.io.VarLenFeature(tf.int64),
    })
    img_str = features['img']
    img_name = features['name']
    ann = tf.concat([features['label'].values[:, None],
                     features['x1'].values[:, None],
                     features['y1'].values[:, None],
                     features['x2'].values[:, None],
                     features['y2'].values[:, None]], 1)
    img_hw = features['img_hw'].values

    return img_str, img_name, ann, img_hw

  @staticmethod
  def calc_xy_offset(out_hw: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    """ for dynamic sacle get xy offset tensor for loss calc

    Parameters
    ----------
    out_hw : tf.Tensor

    Returns
    -------

    [tf.Tensor]

        xy offset : shape [out h , out w , 1 , 2] type=tf.float32

    """
    grid_y = tf.tile(tf.reshape(tf.range(0, out_hw[0]),
                                [-1, 1, 1, 1]), [1, out_hw[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, out_hw[1]),
                                [1, -1, 1, 1]), [out_hw[0], 1, 1, 1])
    xy_offset = tf.concat([grid_x, grid_y], -1)
    return tf.cast(xy_offset, tf.float32)

  @staticmethod
  def corner_to_center(ann: np.ndarray, in_hw: np.ndarray) -> np.ndarray:
    """ conrner ann to center ann
        xyxy ann to xywh ann

    Parameters
    ----------
    ann : np.ndarray

        xyxyann n*[p,x1,y1,x2,y2]
        NOTE all pixel scale

    in_hw : np.ndarray

    Returns
    -------
    np.ndarray

        xywhann n*[p,x,y,w,h]

        NOTE scale = [0~1] x,y is center point
    """
    return np.hstack([
        ann[:, 0:1],
        ((ann[:, 1:2] + ann[:, 3:4]) / 2) / in_hw[1],
        ((ann[:, 2:3] + ann[:, 4:5]) / 2) / in_hw[0],
        (ann[:, 3:4] - ann[:, 1:2]) / in_hw[1],
        (ann[:, 4:5] - ann[:, 2:3]) / in_hw[0]])

  @staticmethod
  def center_to_corner(ann: np.ndarray, in_hw: np.ndarray) -> np.ndarray:
    """ center ann to conrner ann
        xywh ann to xyxy ann

    Parameters
    ----------
    ann : np.ndarray

        xywhann n*[p,x,y,w,h]
        NOTE scale = [0~1] x,y is center point

    in_hw : np.ndarray

    Returns
    -------
    np.ndarray

        xyxyann n*[p,x1,y1,x2,y2]
        NOTE all pixel scale

    """
    return np.hstack([
        ann[:, 0:1],
        (ann[:, 1:2] - ann[:, 3:4] / 2) * in_hw[1],
        (ann[:, 2:3] - ann[:, 4:5] / 2) * in_hw[0],
        (ann[:, 1:2] + ann[:, 3:4] / 2) * in_hw[1],
        (ann[:, 2:3] + ann[:, 4:5] / 2) * in_hw[0]])

  def ann_to_label(self, in_hw: np.ndarray, out_hw: np.ndarray, ann: np.ndarray) -> tuple:
    """convert the annotaion to yolo v3 label~

    Parameters
    ----------
    ann : np.ndarray
        annotation shape :[n,5] value : n*[p,x1,y1,x2,y2]

    Returns
    -------
    tuple
        labels list value :[output_number*[ out_h, out_w, anchor_num, class +5 +1 ]]

        NOTE The last element of the last dimension is the allocation flag,
         which means that there is ground truth at this position.
        Can use only one output label find all ground truth~
        ```python
        new_anns = []
        for i in range(h.output_number):
            new_ann = labels[i][np.where(labels[i][..., -1] == 1)]
            new_anns.append(np.c_[np.argmax(new_ann[:, 5:], axis=-1), new_ann[:, :4]])
        np.allclose(new_anns[0], new_anns[1])  # True
        np.allclose(new_anns[1], new_anns[2])  # True
        ```

    """
    labels = [np.zeros((out_hw[i][0], out_hw[i][1], len(self.anchors[i]),
                        5 + self.class_num + 1), dtype='float32') for i in range(self.output_number)]

    ann = self.corner_to_center(ann, in_hw)
    layer_idx, anchor_idx = self._get_anchor_index(ann[:, 3: 5])
    for box, l, n in zip(ann, layer_idx, anchor_idx):
      # NOTE box [x y w h] are relative to the size of the entire image [0~1]
      bb = box[1: 5]
      cnt = np.zeros(self.output_number, np.bool)  # assigned flag
      for i in range(len(l)):
        x, y = self._xy_grid_index(out_hw, bb[0: 2], l[i])  # [x index , y index]
        if cnt[l[i]] or labels[l[i]][y, x, n[i], 4] == 1.:
          # 1. when this output layer already have ground truth, skip
          # 2. when this grid already being assigned, skip
          continue
        labels[l[i]][y, x, n[i], 0: 4] = bb
        labels[l[i]][y, x, n[i], 4] = (0. if cnt.any() else 1.)
        labels[l[i]][y, x, n[i], 5 + int(box[0])] = 1.
        labels[l[i]][y, x, n[i], -1] = 1.  # set gt flag = 1
        cnt[l[i]] = True  # output layer ground truth flag
        if cnt.all():
          # when all output layer have ground truth, exit
          break
    return labels

  def label_to_ann(self, labels: tuple, thersh=.7) -> np.ndarray:
    """reverse the labels to annotation

    Parameters
    ----------
    labels : np.ndarray

    Returns
    -------
    np.ndarray
        annotaions
    """
    new_ann = np.vstack([label[np.where(label[..., 4] > thersh)] for label in labels])
    new_ann = np.c_[np.argmax(new_ann[:, 5:], axis=-1), new_ann[:, :4]]
    new_ann = self.center_to_corner(new_ann, self.org_in_hw)
    return new_ann

  @staticmethod
  def validate_ann(clses: tf.Tensor, x1: tf.Tensor, y1: tf.Tensor,
                   x2: tf.Tensor, y2: tf.Tensor,
                   im_w: tf.Tensor, im_h: tf.Tensor) -> tf.Tensor:
    """ when resize or augment img, need validate ann value """
    x1 = tf.clip_by_value(x1, 0, im_w - 1)
    y1 = tf.clip_by_value(y1, 0, im_h - 1)
    x2 = tf.clip_by_value(x2, 0, im_w - 1)
    y2 = tf.clip_by_value(y2, 0, im_h - 1)
    new_ann = tf.concat([clses, x1, y1, x2, y2], -1)
    new_ann.set_shape((None, None))

    bbox_w = new_ann[:, 3] - new_ann[:, 1]
    bbox_h = new_ann[:, 4] - new_ann[:, 2]
    new_ann = tf.boolean_mask(new_ann, tf.logical_and(bbox_w > 1, bbox_h > 1))
    return new_ann

  def iaa_augment_img(self, img: np.ndarray, ann: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ augmenter for image

    Parameters
    ----------
    img : np.ndarray

        img src

    ann : np.ndarray

        one annotation
        [p,x1,y1,x2,y2]

    Returns
    -------
    tuple

        [image src,ann] after data augmenter
        image src dtype is uint8
    """
    clses = ann[:, 0: 1]
    bbox = ann[:, 1:]
    im_wh = img.shape[1::-1]

    bbs = BoundingBoxesOnImage.from_xyxy_array(bbox, shape=img.shape)
    image_aug, bbs_aug = self.iaaseq(image=img, bounding_boxes=bbs)
    new_bbox = bbs_aug.to_xyxy_array()

    # remove out of bound bbox and the bbox which w or h < 0
    x1, y1, x2, y2 = np.split(new_bbox, 4, -1)
    x1 = np.clip(x1, 0, im_wh[0] - 1)
    y1 = np.clip(y1, 0, im_wh[1] - 1)
    x2 = np.clip(x2, 0, im_wh[0] - 1)
    y2 = np.clip(y2, 0, im_wh[1] - 1)
    new_ann = np.concatenate([clses, x1, y1, x2, y2], -1)

    bbox_w = new_ann[:, 3] - new_ann[:, 1]
    bbox_h = new_ann[:, 4] - new_ann[:, 2]
    new_ann = new_ann[np.logical_and(bbox_w > 1, bbox_h > 1)]
    return image_aug, new_ann

  def origin_augment_img(self, img: tf.Tensor, ann: tf.Tensor,
                         hue=0.3, sat=0.2, val=0.3) -> [tf.Tensor, tf.Tensor]:
    """ augmenter for image

    Parameters
    ----------
    img : tf.Tensor

        img src

    ann : tf.Tensor

        one annotation
        [p,x1,y1,x2,y2]

    Returns
    -------
    tuple

        [image src,ann] after data augmenter
        image src dtype is uint8
    """
    if hue > 0:
      img = tf.image.random_hue(img, hue)
    if sat > 0:
      img = tf.image.random_saturation(img, 1 - sat, 1 + sat)
    if val > 0:
      img = tf.image.random_brightness(img, val)
    return img, ann

  def auto_augment_img(self, img: tf.Tensor, ann: tf.Tensor,
                       augmentation_name: str = 'v2') -> [tf.Tensor, tf.Tensor]:
    from transforms.image.auto_augment import distort_image_with_autoaugment
    from transforms.image.box_utils import yxyx_to_xyxy, normalize_boxes, denormalize_boxes
    clas, xyxybox = tf.split(ann, [1, 4], 1)
    yxyxbox = yxyx_to_xyxy(xyxybox)
    img, yxyxbox = distort_image_with_autoaugment(img, yxyxbox, augmentation_name=augmentation_name)
    xyxybox = yxyx_to_xyxy(yxyxbox)
    ann = tf.concat([clas, xyxybox], -1)
    return img, ann

  def origin_resize_train_img(self, img: tf.Tensor, in_hw: tf.Tensor,
                              ann: tf.Tensor, min_scale=0.25, max_scale=2,
                              jitter=0.3, flip=True) -> [tf.Tensor, tf.Tensor]:
    """ when training first crop image and resize image and keep ratio

    Parameters
    ----------
    img : tf.Tensor

    in_hw : tf.Tensor

    ann : tf.Tensor

    Returns
    -------
    [tf.Tensor, tf.Tensor]
        img, ann
    """
    iw, ih = tf.cast(tf.shape(img)[1], tf.float32), tf.cast(tf.shape(img)[0], tf.float32)
    w, h = tf.cast(in_hw[1], tf.float32), tf.cast(in_hw[0], tf.float32)
    clses, x1, y1, x2, y2 = tf.split(ann, 5, -1)

    new_ar = (w / h) * (tf.random.uniform([], 1 - jitter, 1 + jitter) /
                        tf.random.uniform([], 1 - jitter, 1 + jitter))
    scale = tf.random.uniform([], min_scale, max_scale)
    ratio = tf.cond(tf.less(new_ar, 1),
                    lambda: scale * new_ar,
                    lambda: scale / new_ar)
    ratio = tf.maximum(ratio, 1)
    nw, nh = tf.cond(tf.less(new_ar, 1),
                     lambda: (ratio * h, scale * h),
                     lambda: (scale * w, ratio * w))
    dx = tf.random.uniform([], 0, w - nw)
    dy = tf.random.uniform([], 0, h - nh)
    img = tf.image.resize(img, [tf.cast(nh, tf.int32), tf.cast(nw, tf.int32)])

    def crop_and_pad(image, dx, dy):
      dy_t = tf.cast(tf.maximum(-dy, 0), tf.int32)
      dx_t = tf.cast(tf.maximum(-dx, 0), tf.int32)
      image = tf.image.crop_to_bounding_box(
          image, dy_t, dx_t,
          tf.minimum(tf.cast(h, tf.int32), tf.cast(nh, tf.int32)),
          tf.minimum(tf.cast(w, tf.int32), tf.cast(nw, tf.int32)))
      image = tf.image.pad_to_bounding_box(
          image, tf.cast(tf.maximum(dy, 0), tf.int32),
          tf.cast(tf.maximum(dx, 0), tf.int32), tf.cast(
              h, tf.int32), tf.cast(w, tf.int32))
      return image
    img = tf.cond(tf.logical_or(nw > w, nh > h),
                  lambda: crop_and_pad(img, dx, dy),
                  lambda: tf.image.pad_to_bounding_box(
        img, tf.cast(tf.maximum(dy, 0), tf.int32),
        tf.cast(tf.maximum(dx, 0), tf.int32),
        tf.cast(h, tf.int32), tf.cast(w, tf.int32)))

    x1 = x1 * nw / iw + dx
    x2 = x2 * nw / iw + dx
    y1 = y1 * nh / ih + dy
    y2 = y2 * nh / ih + dy

    if flip:
      img, x1, x2 = tf.cond(
          tf.less(tf.random.uniform([]), 0.5),
          lambda: (tf.image.flip_left_right(img), w - x2, w - x1),
          lambda: (img, x1, x2))

    new_ann = self.validate_ann(clses, x1, y1, x2, y2, w, h)
    return img, new_ann

  def gluon_resize_train_img(self, img: tf.Tensor, in_hw: tf.Tensor,
                             ann: tf.Tensor, min_scale: float = 0.3,
                             max_scale: float = 1, max_aspect_ratio: float = 2,
                             max_trial: int = 50) -> [tf.Tensor, tf.Tensor]:
    new_ann, crop_idx = tf.numpy_function(
        bbox_crop_constraints, [ann, tf.shape(img)[1::-1],
                                tf.constant(min_scale), tf.constant(max_scale),
                                tf.constant(max_aspect_ratio), tf.constant(max_trial)], [tf.float32, tf.int32])
    # crop_idx (x1, y1, x2, y2)
    img = img[crop_idx[1]:crop_idx[3], crop_idx[0]:crop_idx[2], :]
    img, new_ann = self.resize_img(img, in_hw, new_ann)
    return img, new_ann

  def resize_img(self, img: tf.Tensor, in_hw: tf.Tensor,
                 ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    """
    resize image and keep ratio

    Parameters
    ----------
    img : tf.Tensor

    ann : tf.Tensor


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        img, ann [ uint8 , float32 ]
    """
    img_hw = tf.shape(img, tf.int64)[:2]
    iw, ih = tf.cast(img_hw[1], tf.float32), tf.cast(img_hw[0], tf.float32)
    w, h = tf.cast(in_hw[1], tf.float32), tf.cast(in_hw[0], tf.float32)
    clses, x1, y1, x2, y2 = tf.split(ann, 5, -1)

    img.set_shape((None, None, 3))
    """ transform factor """
    def _resize(img, clses, x1, y1, x2, y2):
      nh = ih * tf.minimum(w / iw, h / ih)
      nw = iw * tf.minimum(w / iw, h / ih)
      dx = (w - nw) / 2
      dy = (h - nh) / 2
      img = tf.image.resize(img, [tf.cast(nh, tf.int32),
                                  tf.cast(nw, tf.int32)],
                            'nearest', antialias=True)
      img = tf.image.pad_to_bounding_box(img, tf.cast(dy, tf.int32),
                                         tf.cast(dx, tf.int32),
                                         tf.cast(h, tf.int32),
                                         tf.cast(w, tf.int32))
      x1 = x1 * nw / iw + dx
      x2 = x2 * nw / iw + dx
      y1 = y1 * nh / ih + dy
      y2 = y2 * nh / ih + dy
      return img, clses, x1, y1, x2, y2

    img, clses, x1, y1, x2, y2 = tf.cond(tf.reduce_all(tf.equal(in_hw, img_hw)),
                                         lambda: (img, clses, x1, y1, x2, y2),
                                         lambda: _resize(img, clses, x1, y1, x2, y2))

    new_ann = self.validate_ann(clses, x1, y1, x2, y2, w, h)
    return img, new_ann

  def read_img(self, img_path: str) -> tf.Tensor:
    """ read image
    """
    return tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)

  def decode_img(self, img_str: bytes) -> tf.Tensor:
    """ decode image string
    """
    return tf.image.decode_jpeg(img_str, channels=3)

  def process_img(self, img: tf.Tensor, ann: tf.Tensor, in_hw: tf.Tensor,
                  is_augment: bool, is_resize: bool,
                  is_normlize: bool) -> [tf.Tensor, tf.Tensor]:
    """ origin yolo process image and true box , if is training then use data augmenter

    Parameters
    ----------
    img : tf.Tensor
        image srs
    ann : tf.Tensor
        one annotation
    is_augment : bool
        wether to use data augmenter
    is_resize : bool
        wether to resize the image
    is_normlize : bool
        wether to normlize the image

    Returns
    -------
    tuple
        image src , true box
    """
    if is_resize and is_augment:
      # print(INFO, f'resize_method : {self.resize_method}')
      if self.resize_method == 'origin':
        img, ann = self.origin_resize_train_img(img, in_hw, ann)
      elif self.resize_method == 'gluon':
        img, ann = self.gluon_resize_train_img(img, in_hw, ann)
      elif self.resize_method == 'none':
        img, ann = self.resize_img(img, in_hw, ann)
      else:
        raise ValueError(f'resize_method: {self.resize_method} is error')
    elif is_resize:
      img, ann = self.resize_img(img, in_hw, ann)
    if is_augment:
      # print(INFO, f'augment_method : {self.augment_method}')
      if self.augment_method == 'origin':
        img, ann = self.origin_augment_img(img, ann)
      elif self.augment_method == 'iaa':
        img, ann = tf.numpy_function(self.iaa_augment_img, [img, ann], [tf.uint8, tf.float32])
      elif self.augment_method == 'autoaug':
        img, ann = self.auto_augment_img(img, ann)
      else:
        raise ValueError(f'augment_method: {self.augment_method} is error')
    if is_normlize:
      img = self.normlize_img(img)
    else:
      img = tf.cast(img, tf.float32)
    return img, ann

  def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int, is_augment: bool,
                     is_normlize: bool, is_training: bool) -> tf.data.Dataset:
    print(INFO, 'data augment is ', str(is_augment))

    def _parser_wrapper(stream: bytes):
      # NOTE use wrapper function and dynamic list construct (x,(y_1,y_2,...))
      img_str, _, ann, _ = self.parser_example(stream)
      # load image
      img = self.decode_img(img_str)
      # process image
      img, ann = self.process_img(img, ann, self.in_hw,
                                  is_augment, True, is_normlize)
      # make labels
      labels = tf.numpy_function(self.ann_to_label,
                                 [self.in_hw, self.out_hw, ann],
                                 [tf.float32] * len(self.anchors))
      # set shape
      img.set_shape((None, None, 3))
      for i in range(len(self.anchors)):
        labels[i].set_shape((None, None, len(self.anchors[i]), self.class_num + 5 + 1))
      return img, tuple(labels)

    if is_training:
      dataset = (tf.data.TFRecordDataset(image_ann_list, num_parallel_reads=4).
                 shuffle(batch_size * 500).
                 repeat().
                 map(_parser_wrapper, self.num_parallel_calls).
                 batch(batch_size, True).
                 prefetch(-1))
    else:
      def _parser_wrapper(stream: bytes):
        img_str, img_name, ann, img_hw = self.parser_example(stream)
        raw_img = self.decode_img(img_str)
        det_img, _ = self.process_img(raw_img, tf.zeros(
            [0, 5]), self.in_hw, is_augment, True, is_normlize)
        return det_img, img_name, tf.RaggedTensor.from_tensor(ann), img_hw

      dataset = (tf.data.TFRecordDataset(image_ann_list, num_parallel_reads=4).
                 map(_parser_wrapper, -1).
                 batch(batch_size, True).
                 prefetch(-1))
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    return dataset

  def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True, scores=None) -> np.ndarray:
    """ draw img and show bbox , set ann = None will not show bbox

    Parameters
    ----------
    img : np.ndarray

    ann : np.ndarray

        scale is all image pixal scale
        shape : [p,x1,y1,x2,y2]

    is_show : bool

        show image
    """
    if isinstance(ann, np.ndarray):
      p = ann[:, 0]
      xyxybox = ann[:, 1:]
      for i, a in enumerate(xyxybox):
        classes = int(p[i])
        r_top = tuple(np.maximum(np.minimum(a[0:2], img.shape[1::-1]), 0).astype(int))
        l_bottom = tuple(np.maximum(np.minimum(a[2:], img.shape[1::-1]), 0).astype(int))
        r_bottom = (r_top[0], l_bottom[1])
        org = (np.maximum(np.minimum(r_bottom[0], img.shape[1] - 12), 0),
               np.maximum(np.minimum(r_bottom[1], img.shape[0] - 12), 0))
        cv2.rectangle(img, r_top, l_bottom, self.colormap[classes])
        if isinstance(scores, np.ndarray):
          cv2.putText(img, f'{classes} {scores[i]:.2f}',
                      org, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      0.5, self.colormap[classes], thickness=1)
        else:
          cv2.putText(img, f'{classes}', org,
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                      self.colormap[classes], thickness=1)

    if is_show:
      imshow((img).astype('uint8'))
      show()

    return img.astype('uint8')


class MultiScaleTrain(Callback):
  def __init__(self, h: YOLOHelper, interval: int = 10, scale_range: list = [-3, 3]):
    """ Multi-scale training callback

        NOTE This implementation will lead to the lack of multi-scale
             training in several batches after the end of validation

    Parameters
    ----------
    h : YOLOHelper

    interval : int, optional

        change scale batch interval, by default 10

    scale_range : list, optional

        change scale range, by default [-3, 3]
        eg.
        ```
            org_input_size = 416
            x = 2 # in range(-3,3)
            input_size = org_input_size + (x * 32)
                       = 416 + (2 * 32)
                       = 480
        ```
    """
    super().__init__()
    self.h = h
    self.interval = interval
    self.scale_range = np.arange(scale_range[0], scale_range[1])
    self.cur_scl = scale_range[1]  # default max size
    self.flag = True  # change flag
    self.count = 1

  def on_train_begin(self, logs=None):
    K.set_value(self.h.in_hw, self.h.org_in_hw + 32 * self.cur_scl)
    K.set_value(self.h.out_hw, self.h.org_out_hw + np.power(2,
                                                            np.arange(self.h.output_number))[:, None] * self.cur_scl)
    for i, out_hw in enumerate(self.h.out_hw):
      K.set_value(self.h.xy_offsets[i], self.h.calc_xy_offset(out_hw))

    print(f'\n {NOTE} : Train input image size : [{self.h.in_hw[0]},{self.h.in_hw[1]}]')

  def on_epoch_begin(self, epoch, logs=None):
    self.flag = True

  def on_train_batch_end(self, batch, logs=None):
    if self.flag == True:
      if self.count % self.interval == 0:
        # random choice resize scale
        self.cur_scl = np.random.choice(self.scale_range)
        K.set_value(self.h.in_hw, self.h.org_in_hw + 32 * self.cur_scl)
        K.set_value(self.h.out_hw, self.h.org_out_hw + np.power(2,
                                                                np.arange(self.h.output_number))[:, None] * self.cur_scl)
        for i, out_hw in enumerate(self.h.out_hw):
          K.set_value(self.h.xy_offsets[i], self.h.calc_xy_offset(out_hw))
        self.count = 1
        print(f'\n {NOTE} : Train input image size : [{self.h.in_hw[0]},{self.h.in_hw[1]}]')
      else:
        self.count += 1

  def on_test_begin(self, logs=None):
    """ change to orginal image size """
    if self.flag == True:
      self.flag = False
      K.set_value(self.h.in_hw, self.h.org_in_hw)
      K.set_value(self.h.out_hw, self.h.org_out_hw)
      for i, out_hw in enumerate(self.h.out_hw):
        K.set_value(self.h.xy_offsets[i], self.h.calc_xy_offset(out_hw))


class YOLOLoss(Loss):
  def __init__(self, h: YOLOHelper, iou_thresh: float, obj_thresh: float,
               obj_weight: float, noobj_weight: float, wh_weight: float,
               xy_weight: float, cls_weight: float, use_focalloss: bool,
               focal_gamma: float, focal_alpha: float, layer: int, verbose=1,
               reduction='auto', name=None):
    """ yolo loss obj

    Parameters
    ----------
    h : YOLOHelper

    iou_thresh : float

    obj_weight : float

    noobj_weight : float

    wh_weight : float

    layer : int
        the current layer index

    """
    super().__init__(reduction=reduction, name=name)
    self.h = h
    self.iou_thresh = iou_thresh
    self.obj_thresh = obj_thresh
    self.obj_weight = obj_weight
    self.noobj_weight = noobj_weight
    self.wh_weight = wh_weight
    self.xy_weight = xy_weight
    self.cls_weight = cls_weight
    if use_focalloss == True:
      self.bce_fn = (lambda labels, logits:
                     focal_sigmoid_cross_entropy_with_logits(
                         labels, logits,
                         focal_gamma, focal_alpha))
    else:
      self.bce_fn = (lambda labels, logits:
                     tf.nn.sigmoid_cross_entropy_with_logits(
                         labels, logits))

    # focal_gamma: float, alpha: float,
    self.layer = layer
    self.anchors: np.ndarray = np.copy(self.h.anchors[self.layer])
    self.xy_offset: ResourceVariable = self.h.xy_offsets[self.layer]
    self.verbose = verbose
    self.op_list = []
    self.lookups: Iterable[Tuple[ResourceVariable, AnyStr]] = []
    with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                     reuse=tf.compat.v1.AUTO_REUSE):

      self.tp: ResourceVariable = tf.compat.v1.get_variable(
          'tp', (), tf.float32,
          tf.zeros_initializer(),
          trainable=False)

      if self.verbose > 0:
        names = ['r', 'p']
        self.lookups.extend([
            (tf.compat.v1.get_variable(name, (), tf.float32,
                                       tf.zeros_initializer(),
                                       trainable=False),
                name)
            for name in names])

    if self.verbose > 1:
      with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                       reuse=tf.compat.v1.AUTO_REUSE):
        names = ['xy', 'wh', 'obj', 'noobj', 'cls']
        self.lookups.extend([
            (tf.compat.v1.get_variable(name, (), tf.float32,
                                       tf.zeros_initializer(),
                                       trainable=False),
             name)
            for name in names])

  @staticmethod
  def xywh_to_grid(all_true_xy: tf.Tensor, all_true_wh: tf.Tensor,
                   out_hw: tf.Tensor, xy_offset: tf.Tensor,
                   anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    """convert true label xy wh to grid scale

    Returns
    -------
    [tf.Tensor, tf.Tensor]

        grid_true_xy, grid_true_wh shape = [out h ,out w,anchor num , 2 ]

    """
    grid_true_xy = (all_true_xy * out_hw[::-1]) - xy_offset
    grid_true_wh = tf.math.log(all_true_wh / anchors)
    return grid_true_xy, grid_true_wh

  @staticmethod
  def xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor,
                  out_hw: tf.Tensor, xy_offset: tf.Tensor,
                  anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

    Returns
    -------
    [tf.Tensor, tf.Tensor]

        [all_pred_xy, all_pred_wh]
    """
    all_pred_xy = (tf.sigmoid(grid_pred_xy) + xy_offset) / out_hw[::-1]
    all_pred_wh = tf.exp(grid_pred_wh) * anchors
    return all_pred_xy, all_pred_wh

  def smoothl1loss(self, labels: tf.Tensor, predictions: tf.Tensor, delta=1.0):
    error = tf.math.subtract(predictions, labels)
    abs_error = tf.math.abs(error)
    quadratic = tf.math.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = tf.math.subtract(abs_error, quadratic)
    return tf.math.add(tf.math.multiply(0.5, tf.math.multiply(quadratic, quadratic)),
                       tf.math.multiply(delta, linear))

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """ reshape y pred """
    out_hw = tf.cast(tf.shape(y_true)[1:3], tf.float32)

    y_true = tf.reshape(y_true, [-1, out_hw[0], out_hw[1],
                                 self.h.anchor_number, self.h.class_num + 5 + 1])
    y_pred = tf.reshape(y_pred, [-1, out_hw[0], out_hw[1],
                                 self.h.anchor_number, self.h.class_num + 5])

    """ split the label """
    grid_pred_xy = y_pred[..., 0:2]
    grid_pred_wh = y_pred[..., 2:4]
    pred_confidence = y_pred[..., 4:5]
    pred_cls = y_pred[..., 5:]

    all_true_xy = y_true[..., 0:2]
    all_true_wh = y_true[..., 2:4]
    true_confidence = y_true[..., 4:5]
    true_cls = y_true[..., 5:5 + self.h.class_num]
    location_mask = tf.cast(y_true[..., -1], tf.bool)

    obj_mask = true_confidence
    obj_mask_bool = tf.cast(y_true[..., 4], tf.bool)

    """ calc the ignore mask  """
    all_pred_xy, all_pred_wh = self.xywh_to_all(grid_pred_xy, grid_pred_wh,
                                                out_hw, self.xy_offset, self.anchors)
    all_pred_bbox = tf.concat([all_pred_xy - all_pred_wh / 2,
                               all_pred_xy + all_pred_wh / 2], -1)
    all_true_bbox = tf.concat([all_true_xy - all_true_wh / 2,
                               all_true_xy + all_true_wh / 2], -1)
    obj_cnt = tf.reduce_sum(obj_mask)

    def lmba(bc):
      # NOTE use location_mask find all ground truth
      one_all_true_bbox = tf.boolean_mask(all_true_bbox[bc], location_mask[bc])
      iou_score = tf_bbox_iou(all_pred_bbox[bc], one_all_true_bbox)  # [h,w,anchor,box_num]
      # NOTE find this layer gt and pred iou score
      idx = tf.where(tf.boolean_mask(obj_mask_bool[bc], location_mask[bc]))
      mask_iou_score = tf.gather_nd(tf.boolean_mask(iou_score, obj_mask_bool[bc]), idx, 1)
      with tf.control_dependencies(
              [self.tp.assign_add(tf.reduce_sum(tf.cast(mask_iou_score > self.iou_thresh, tf.float32)))]):
        layer_iou_score = tf.squeeze(tf.gather(iou_score, idx, axis=-1), -1)
        layer_match = tf.reduce_sum(
            tf.cast(layer_iou_score > self.iou_thresh, tf.float32), -1, keepdims=True)
        # if iou for any ground truth larger than iou_thresh, the pred is true.
        match_num = tf.reduce_sum(
            tf.cast(iou_score > self.iou_thresh, tf.float32), -1, keepdims=True)
      return (tf.cast(tf.less(match_num, 1), tf.float32),
              tf.cast(tf.less(layer_match, 1), tf.float32))

    ignore_mask, layer_ignore_mask = tf.map_fn(
        lmba, tf.range(self.h.batch_size), dtype=(tf.float32, tf.float32))
    """ calc recall precision """
    pred_confidence_sigmod = tf.sigmoid(pred_confidence)
    fp = tf.reduce_sum((tf.cast(pred_confidence_sigmod > self.obj_thresh,
                                tf.float32) * layer_ignore_mask) * (1 - obj_mask))
    fn = tf.reduce_sum((tf.cast(pred_confidence_sigmod < self.obj_thresh,
                                tf.float32) + layer_ignore_mask) * obj_mask)

    precision = tf.math.divide_no_nan(self.tp, (self.tp + fp))
    recall = tf.math.divide_no_nan(self.tp, (self.tp + fn))

    """ calc the loss dynamic weight """
    grid_true_xy, grid_true_wh = self.xywh_to_grid(all_true_xy, all_true_wh,
                                                   out_hw, self.xy_offset, self.anchors)
    # NOTE When wh=0 , tf.log(0) = -inf, so use tf.where to avoid it
    grid_true_wh = tf.where(tf.tile(obj_mask_bool[..., tf.newaxis], [1, 1, 1, 1, 2]),
                            grid_true_wh, tf.zeros_like(grid_true_wh))
    coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

    """ calc the loss """
    xy_loss = tf.reduce_sum(
        obj_mask * coord_weight * self.xy_weight * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=grid_true_xy, logits=grid_pred_xy), [1, 2, 3, 4])

    wh_loss = tf.reduce_sum(
        obj_mask * coord_weight * self.wh_weight * self.smoothl1loss(
            labels=grid_true_wh, predictions=grid_pred_wh), [1, 2, 3, 4])

    obj_loss = self.obj_weight * tf.reduce_sum(
        obj_mask * self.bce_fn(true_confidence, pred_confidence),
        [1, 2, 3, 4])

    noobj_loss = self.noobj_weight * tf.reduce_sum(
        (1 - obj_mask) * ignore_mask * self.bce_fn(true_confidence, pred_confidence),
        [1, 2, 3, 4])

    cls_loss = tf.reduce_sum(
        obj_mask * self.cls_weight * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_cls, logits=pred_cls), [1, 2, 3, 4])

    """ sum loss """
    if self.verbose > 0:
      self.op_list.extend([
          self.lookups[0][0].assign(recall),
          self.lookups[1][0].assign(precision),
          self.tp.assign(0)
      ])
    if self.verbose > 1:
      lknm = len(self.lookups)
      self.op_list.extend([self.lookups[-5][0].assign(tf.reduce_mean(xy_loss)),
                           self.lookups[-4][0].assign(tf.reduce_mean(wh_loss)),
                           self.lookups[-3][0].assign(tf.reduce_mean(obj_loss)),
                           self.lookups[-2][0].assign(tf.reduce_mean(noobj_loss)),
                           self.lookups[-1][0].assign(tf.reduce_mean(cls_loss))])

    with tf.control_dependencies(self.op_list):
      total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss
    return total_loss


class YOLOIouLoss(YOLOLoss):
  def __init__(self, h: YOLOHelper, iou_thresh: float, obj_thresh: float,
               obj_weight: float, noobj_weight: float, bbox_weight: float,
               cls_weight: float, use_focalloss: bool,
               focal_gamma: float, focal_alpha: float,
               layer: int, iou_method: str,
               reduction='auto', name=None):
    self.reduction = reduction
    self.name = name
    self.h = h
    self.iou_thresh = iou_thresh
    self.obj_thresh = obj_thresh
    self.obj_weight = obj_weight
    self.iou_method = iou_method
    self.noobj_weight = noobj_weight
    self.bbox_weight = bbox_weight
    self.cls_weight = cls_weight
    if use_focalloss == True:
      self.bce_fn = (lambda labels, logits:
                     focal_sigmoid_cross_entropy_with_logits(
                         labels, logits,
                         focal_gamma, focal_alpha))
    else:
      self.bce_fn = (lambda labels, logits:
                     tf.nn.sigmoid_cross_entropy_with_logits(
                         labels, logits))
    self.layer = layer
    self.anchors: np.ndarray = np.copy(self.h.anchors[self.layer])
    self.xy_offset: ResourceVariable = self.h.xy_offsets[self.layer]
    self.op_list = []
    self.lookups: Iterable[Tuple[ResourceVariable, AnyStr]] = []
    if self.verbose > 1:
      with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                       reuse=tf.compat.v1.AUTO_REUSE):
        names = ['bbox', 'obj', 'noobj', 'cls']
        self.lookups.extend([
            (tf.compat.v1.get_variable(name, (), tf.float32,
                                       tf.zeros_initializer(),
                                       trainable=False),
                name)
            for name in names])

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """ reshape y pred """
    out_hw = tf.cast(tf.shape(y_true)[1:3], tf.float32)

    y_true = tf.reshape(y_true, [-1, out_hw[0], out_hw[1],
                                 self.h.anchor_number, self.h.class_num + 5 + 1])
    y_pred = tf.reshape(y_pred, [-1, out_hw[0], out_hw[1],
                                 self.h.anchor_number, self.h.class_num + 5])

    """ split the label """
    grid_pred_xy = y_pred[..., 0:2]
    grid_pred_wh = y_pred[..., 2:4]
    pred_confidence = y_pred[..., 4:5]
    pred_cls = y_pred[..., 5:]
    pred_confidence = tf.clip_by_value(pred_confidence, -16.118095, 15.942385)
    pred_cls = tf.clip_by_value(pred_cls, -16.118095, 15.942385)

    all_true_xy = y_true[..., 0:2]
    all_true_wh = y_true[..., 2:4]
    true_confidence = y_true[..., 4:5]
    true_cls = y_true[..., 5:5 + self.h.class_num]
    location_mask = tf.cast(y_true[..., -1], tf.bool)

    obj_mask = true_confidence
    obj_mask_bool = tf.cast(y_true[..., 4], tf.bool)

    """ calc the ignore mask  """
    all_pred_xy, all_pred_wh = self.xywh_to_all(grid_pred_xy, grid_pred_wh,
                                                out_hw, self.xy_offset, self.anchors)
    all_pred_bbox = tf.concat([all_pred_xy - all_pred_wh / 2,
                               all_pred_xy + all_pred_wh / 2], -1)
    all_true_bbox = tf.concat([all_true_xy - all_true_wh / 2,
                               all_true_xy + all_true_wh / 2], -1)
    obj_cnt = tf.reduce_sum(obj_mask)

    def lmba(bc):
      # NOTE use location_mask find all ground truth
      one_all_true_bbox = tf.boolean_mask(all_true_bbox[bc], location_mask[bc])
      iou_score = tf_bbox_iou(all_pred_bbox[bc], one_all_true_bbox)  # [h,w,anchor,box_num]
      # if iou for any ground truth larger than iou_thresh, the pred is true.
      match_num = tf.reduce_sum(tf.cast(iou_score > self.iou_thresh, tf.float32), -1, keepdims=True)
      return tf.cast(tf.less(match_num, 1), tf.float32)

    ignore_mask = tf.map_fn(lmba, tf.range(self.h.batch_size), dtype=tf.float32)

    """ calc the loss """
    bbox_loss = self.bbox_weight * tf.reduce_sum(
        obj_mask * (1 - tf_bbox_iou(all_pred_bbox,
                                    all_true_bbox[..., None, :], method=self.iou_method)),
        [1, 2, 3, 4])

    obj_loss = self.obj_weight * tf.reduce_sum(
        obj_mask * self.bce_fn(true_confidence, pred_confidence),
        [1, 2, 3, 4])

    noobj_loss = self.noobj_weight * tf.reduce_sum(
        (1 - obj_mask) * ignore_mask * self.bce_fn(true_confidence, pred_confidence),
        [1, 2, 3, 4])

    cls_loss = tf.reduce_sum(
        obj_mask * self.cls_weight * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_cls, logits=pred_cls), [1, 2, 3, 4])

    """ sum loss """
    if self.verbose > 1:
      lknm = len(self.lookups)
      self.op_list.extend([self.lookups[-4][0].assign(tf.reduce_mean(bbox_loss)),
                           self.lookups[-3][0].assign(tf.reduce_mean(obj_loss)),
                           self.lookups[-2][0].assign(tf.reduce_mean(noobj_loss)),
                           self.lookups[-1][0].assign(tf.reduce_mean(cls_loss))])

    with tf.control_dependencies(self.op_list):
      total_loss = obj_loss + noobj_loss + cls_loss + bbox_loss
    return total_loss


def parser_outputs(outputs: List[List[np.ndarray]], orig_hws: List[np.ndarray],
                   obj_thresh: float, nms_thresh: float, iou_method: str, h: YOLOHelper
                   ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """ yolo parser one image output

      outputs : batch * [box,clss,score]

      box : [x1, y1, x2, y2]
      clss : [class]
      score : [score]
  """
  results = []

  for y_pred, orig_hw in zip(outputs, orig_hws):
    # In order to ensure the consistency of the framework code reshape here.
    y_pred = [np.reshape(pred, list(pred.shape[:-1]) + [h.anchor_number,
                                                        5 + h.class_num])
              for pred in y_pred]
    """ box list """
    _xyxy_box = []
    _xyxy_box_scores = []
    """ preprocess label """
    for l, pred_label in enumerate(y_pred):
      """ split the label """
      pred_xy = pred_label[..., 0: 2]
      pred_wh = pred_label[..., 2: 4]
      pred_confidence = pred_label[..., 4: 5]
      pred_cls = pred_label[..., 5:]
      if h.class_num > 1:
        box_scores = expit(pred_cls) * expit(pred_confidence)
      else:
        box_scores = expit(pred_confidence)

      """ reshape box  """
      # NOTE tf_xywh_to_all will auto use sigmoid function
      pred_xy_A, pred_wh_A = YOLOLoss.xywh_to_all(
          pred_xy, pred_wh, h.org_out_hw[l],
          h.xy_offsets[l], h.anchors[l])
      # NOTE boxes from xywh to xyxy
      boxes = np.concatenate((pred_xy_A.numpy(), pred_wh_A.numpy()), -1)
      boxes = boxes * np.tile(h.org_in_hw[::-1], [2])
      boxes[..., :2] -= boxes[..., 2:] / 2
      boxes[..., 2:] += boxes[..., :2]
      # NOTE reverse boxes to orginal image scale
      scale = np.min(h.org_in_hw / orig_hw)
      xy_off = ((h.org_in_hw - orig_hw * scale) / 2)[::-1]
      boxes = (boxes - np.tile(xy_off, [2])) / scale
      boxes = np.reshape(boxes, (-1, 4))
      box_scores = np.reshape(box_scores, (-1, h.class_num))
      """ append box and scores to global list """
      _xyxy_box.append(boxes)
      _xyxy_box_scores.append(box_scores)

    xyxy_box = np.concatenate(_xyxy_box, axis=0)
    xyxy_box_scores = np.concatenate(_xyxy_box_scores, axis=0)

    mask = xyxy_box_scores >= obj_thresh

    """ do nms for every classes"""
    _boxes = []
    _scores = []
    _classes = []
    for c in range(h.class_num):
      class_boxes = xyxy_box[mask[:, c]]
      class_box_scores = xyxy_box_scores[:, c][mask[:, c]]
      select = nms_oneclass(class_boxes, class_box_scores,
                            nms_thresh, method=iou_method)
      class_boxes = class_boxes[select]
      class_box_scores = class_box_scores[select]
      _boxes.append(class_boxes)
      _scores.append(class_box_scores)
      _classes.append(np.ones_like(class_box_scores) * c)

    box: np.ndarray = np.concatenate(_boxes, axis=0)
    clss: np.ndarray = np.concatenate(_classes, axis=0)
    score: np.ndarray = np.concatenate(_scores, axis=0)
    results.append([box, clss, score])
  return results


def yolo_infer(img_path: Path, infer_model: k.Model,
               result_path: Path, h: YOLOHelper,
               iou_method: str = 'iou',
               obj_thresh: float = .7, nms_thresh: float = .3):
  """ yolo infer function

  Parameters
  ----------
  img_path : Path

      image path or image dir path

  infer_model : k.Model

      infer model

  result_path : Path

      result path dir

  h : YOLOHelper

  obj_thresh : float, optional

      object detection thresh, by default .7

  nms_thresh : float, optional

      iou thresh , by default .3

  """

  print(INFO, f'Load Images from {str(img_path)}')
  if img_path.is_dir():
    img_paths = []
    for suffix in ['bmp', 'jpg', 'jpeg', 'png']:
      img_paths += list(map(str, img_path.glob(f'*.{suffix}')))
  elif img_path.is_file():
    img_paths = [str(img_path)]
  else:
    ValueError(f'{ERROR} img_path `{str(img_path)}` is invalid')

  if result_path is not None:
    print(INFO, f'Load NNcase Results from {str(result_path)}')
    if result_path.is_dir():
      ncc_results: np.ndarray = np.array([np.fromfile(
          str(result_path / (Path(img_paths[i]).stem + '.bin')),
          dtype='float32') for i in range(len(img_paths))])
    elif result_path.is_file():
      ncc_results = np.expand_dims(np.fromfile(str(result_path),
                                               dtype='float32'), 0)  # type:np.ndarray
    else:
      ValueError(f'{ERROR} result_path `{str(result_path)}` is invalid')
  else:
    ncc_results = None

  print(INFO, f'Infer Results')
  orig_hws = []
  det_imgs = []
  for img_path in img_paths:
    img = h.read_img(img_path)
    orig_hws.append(img.numpy().shape[:2])
    det_img, _ = h.process_img(img, np.zeros([0, 5], 'float32'), h.org_in_hw, False, True, True)
    det_imgs.append(det_img)
  det_imgs = tf.stack(det_imgs)
  orig_hws = np.array(orig_hws)

  outputs = infer_model.predict(det_imgs, len(orig_hws))
  # NOTE change outputs List to n*[layer_num*[arr]]
  outputs = [[output[i] for output in outputs] for i in range(len(orig_hws))]
  """ parser batch out """
  results = parser_outputs(outputs, orig_hws, obj_thresh,
                           nms_thresh, iou_method, h)

  if result_path is None:
    """ draw gpu result """
    for img_path, (bbox, cals, scores) in zip(img_paths, results):
      draw_img = h.read_img(img_path)

      h.draw_image(draw_img.numpy(), np.hstack([cals[:, None], bbox]), scores=scores)
  else:
    """ draw gpu result and nncase result """
    ncc_preds = []
    for ncc_result in ncc_results:
      split_idx = np.cumsum([np.prod(h.org_out_hw[l]) * h.anchor_number *
                             (h.class_num + 5) for l in range(h.output_number - 1)])
      preds: List[np.ndarray] = np.split(ncc_result, split_idx)
      preds = [np.transpose(np.reshape(pred, [h.anchor_number * (5 + h.class_num)] + list(h.org_out_hw[l])),
                            [1, 2, 0]) for (l, pred) in enumerate(preds)]
      ncc_preds.append(preds)

    ncc_results = parser_outputs(ncc_preds, orig_hws, obj_thresh,
                                 nms_thresh, iou_method, h)
    for img_path, (bbox, cals, scores), (ncc_bbox, ncc_cals, ncc_scores) in zip(img_paths, results, ncc_results):
      draw_img = h.read_img(img_path)
      gpu_img = h.draw_image(draw_img.numpy(), np.hstack([cals[:, None], bbox]),
                             is_show=False, scores=scores)
      ncc_img = h.draw_image(draw_img.numpy(), np.hstack([ncc_cals[:, None], ncc_bbox]),
                             is_show=False, scores=ncc_scores)
      fig: plt.Figure = plt.figure(figsize=(8, 3))
      ax1 = plt.subplot(121)  # type:plt.Axes
      ax2 = plt.subplot(122)  # type:plt.Axes
      ax1.imshow(gpu_img)
      ax2.imshow(ncc_img)
      ax1.set_title('GPU Infer')
      ax2.set_title('Ncc Infer')
      fig.tight_layout()
      plt.axis('off')
      plt.xticks([])
      plt.yticks([])
      plt.show()


class YOLOMap(k.callbacks.Callback):
  def __init__(self, h: YOLOHelper,
               test_dataset: tf.data.Dataset,
               val_map: ResourceVariable,
               epoch_freq: int,
               class_names: list,
               obj_thresh=.45,
               nms_thresh=.5,
               nms_iou_method='iou',
               iou_thresh=.5,
               verbose=False):
    """Calculate the AP given the recall and precision array 1st) We compute a
    version of the measured precision/recall curve with precision monotonically
    decreasing 2nd) We compute the AP as the area under this curve by numerical
    integration."""
    self.h = h
    self.class_names = class_names
    self.class_num = len(self.class_names)
    assert self.class_num == self.h.class_num, 'Class names not equal YOLOHelper class num'
    self.iou_thresh = iou_thresh
    self.obj_thresh = obj_thresh
    self.nms_iou_method = nms_iou_method
    self.nms_thresh = nms_thresh
    self.test_dataset = test_dataset
    self.verbose = verbose
    self.epoch_freq = epoch_freq
    self.val_map = val_map

  def _voc_ap(self, rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

  def calculate_aps(self):
    true_res = {}
    pred_res = []
    idx = 0
    APs = {}
    for det_img, _, true_anns, orig_hws in self.test_dataset:
      outputs = self.model.predict(det_img)
      outputs = [[output[i] for output in outputs] for i in range(len(orig_hws))]

      results = parser_outputs(outputs, orig_hws.numpy(), self.obj_thresh,
                               self.nms_thresh, self.nms_iou_method, self.h)
      for (out_boxes, out_classes, out_scorees), true_ann in zip(results, true_anns):
        pred = np.hstack([np.ones((len(out_boxes), 1)) * idx,
                          out_classes.reshape((-1, 1)), out_scorees.reshape(-1, 1), out_boxes])
        true_ann = np.reshape(true_ann.values.numpy(), (-1, 5))
        pred_res.append(pred)
        true_res[idx] = true_ann
        idx += 1
    pred_res = np.vstack(pred_res)
    for cls in range(self.class_num):
      pred_res_cls = pred_res[pred_res[:, 1] == cls]
      if len(pred_res_cls) == 0:
        APs[cls] = 0
        continue
      true_res_cls = {}
      npos = 0
      for index in true_res:
        objs = [obj for obj in true_res[index] if obj[0] == cls]
        npos += len(objs)
        BBGT = np.array([x[1:] for x in objs])
        true_res_cls[index] = {
            'bbox': BBGT,
            'difficult': [False] * len(objs),
            'det': [False] * len(objs)}

      ids = pred_res_cls[:, 0]
      scores = pred_res_cls[:, 2]
      bboxs = pred_res_cls[:, 3:]
      sorted_ind = np.argsort(-scores)
      bboxs = bboxs[sorted_ind, :]
      ids = ids[sorted_ind]

      nd = len(ids)
      tp = np.zeros(nd)
      fp = np.zeros(nd)
      for j in range(nd):
        res = true_res_cls[ids[j]]
        bbox = bboxs[j, :].astype(float)
        ovmax = -np.inf
        BBGT = res['bbox'].astype(float)
        if BBGT.size > 0:
          overlaps = bbox_iou(BBGT, bbox)
          ovmax = np.max(overlaps)
          jmax = np.argmax(overlaps)
        if ovmax > self.iou_thresh:
          if not res['difficult'][jmax]:
            if not res['det'][jmax]:
              tp[j] = 1.
              res['det'][jmax] = 1
            else:
              fp[j] = 1.
        else:
          fp[j] = 1.

      fp = np.cumsum(fp)
      tp = np.cumsum(tp)
      rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
      prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
      ap = self._voc_ap(rec, prec)
      APs[cls] = ap
    return APs

  def on_epoch_begin(self, epoch, logs=None):
    if (epoch > 2) and ((epoch % self.epoch_freq) == 0):
      logs = logs or {}
      APs = self.calculate_aps()
      mAP = np.mean([APs[cls] for cls in APs])
      if self.verbose:
        for cls in range(self.class_num):
          if cls in APs:
            print(f'{self.class_names[cls]} ap: {APs[cls]:.4f}')
        print(f'mAP: {mAP:.4f}')
      K.set_value(self.val_map, mAP)
      logs['mAP'] = mAP


def yolo_eval(infer_model: k.Model, h: YOLOHelper, det_obj_thresh: float,
              det_iou_thresh: float, mAp_iou_thresh: float, class_name: list,
              iou_method: str = 'iou', save_result: bool = False,
              save_result_dir: str = 'tmp', batch: int = 32):
  """ calc yolo pre-class Ap and mAp

  Parameters
  ----------
  infer_model : k.Model

  h : YOLOHelper

  det_obj_thresh : float

      detection obj thresh

  det_iou_thresh : float

      detection iou thresh

  mAp_iou_thresh : float

      mAp iou thresh

  save_result : bool

      when save result, while save `tmp/detection-results` and `tmp/ground-truth`.

  save_result_dir : str

      default `tmp`

  batch : int 

      default 32
  """
  res_path = Path(save_result_dir + '/detection-results')
  gt_path = Path(save_result_dir + '/ground-truth')
  if gt_path.exists():
    shutil.rmtree(str(gt_path))
  if res_path.exists():
    shutil.rmtree(str(res_path))
  gt_path.mkdir(parents=True)
  res_path.mkdir(parents=True)
  class_name = np.array(class_name)
  h.set_dataset(batch, False, True, False)
  for det_imgs, img_names, true_anns, orig_hws in tqdm(h.test_dataset, total=int(h.test_epoch_step)):
    img_names = img_names.numpy().astype('str')
    orig_hws = orig_hws.numpy()

    outputs = infer_model.predict(det_imgs, len(img_names))
    # NOTE change outputs List to n*[layer_num*[arr]]
    outputs = [[output[i] for output in outputs] for i in range(len(orig_hws))]
    results = parser_outputs(outputs, orig_hws, det_obj_thresh, det_iou_thresh, iou_method, h)
    for img_name, (p_xyxy, p_clas, p_score), true_ann in zip(img_names,
                                                             results,
                                                             true_anns):
      true_ann = np.reshape(true_ann.values.numpy(), (-1, 5))
      p_s = p_score[:, None].astype('<U7')
      p_c = class_name[p_clas[:, None].astype(np.int)]
      p_x = p_xyxy.astype(np.int32).astype('<U6')
      res_arr = np.concatenate([p_c, p_s, p_x], -1)
      np.savetxt(str(res_path / f'{img_name}.txt'), res_arr, fmt='%s')

      true_clas, true_xyxy = np.split(true_ann, [1], -1)
      t_c = class_name[true_clas.astype(np.int)]
      t_x = true_xyxy.astype(np.int32).astype('<U6')
      t_arr = np.concatenate([t_c, t_x], -1)
      np.savetxt(str(gt_path / f'{img_name}.txt'), t_arr, fmt='%s')
