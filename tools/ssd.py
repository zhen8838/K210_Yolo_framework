import tensorflow as tf
from tensorflow.python.keras.losses import huber_loss
import numpy as np
from numpy import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
from pathlib import Path
import cv2
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from tools.bbox_utils import tf_center_to_corner, bbox_iou, bbox_iof, tf_bbox_iou, nms_oneclass
from tools.base import BaseHelper
from typing import List, Tuple, AnyStr, Iterable
from scipy.special import softmax
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.framework.ops import _EagerTensorBase, EagerTensor
from tools.base import INFO, NOTE, ERROR
from itertools import product
from toolz import partial
from tools.yolo import YOLOHelper
import sys


def encode_bbox(matches, anchors, variances):
  g_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 - anchors[:, :2]
  g_cxcy /= (variances[0] * anchors[:, 2:])
  g_wh = (matches[:, 2:] - matches[:, :2]) / anchors[:, 2:]
  g_wh = np.log(g_wh) / variances[1]
  # return target for smooth_l1_loss
  return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]


def tf_encode_bbox(matches: tf.Tensor, anchors: tf.Tensor, variances) -> tf.Tensor:
  """ encode bbox

  Parameters
  ----------
  matches : tf.Tensor

      gt matches 

  anchors : tf.Tensor

      anchors x*[x,y,w,h]

  variances :

      rate

  Returns
  -------
  tf.Tensor

  """
  g_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 - anchors[:, :2]
  g_cxcy = g_cxcy / (variances[0] * anchors[:, 2:])
  g_wh = (matches[:, 2:] - matches[:, :2]) / anchors[:, 2:]
  g_wh = tf.math.log(g_wh) / variances[1]
  # return target for smooth_l1_loss
  return tf.concat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode_bbox(bbox: np.ndarray, anchors: np.ndarray, variances: np.ndarray):
  """Decode locations from predictions using anchors to undo
  the encoding we did for offset regression at train time.

  Parameters
  ----------
  bbox : np.ndarray

      pred bbox

  anchors : np.ndarray

      anchors x*[x,y,w,h]

  variances : np.ndarray

      rate

  Returns
  -------
  np.ndarray

     decode bbox, n*[x1,y1,x2,y2]

  """

  boxes = np.concatenate((
      anchors[:, :2] + bbox[:, :2] * variances[0] * anchors[:, 2:],
      anchors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
  boxes[:, :2] -= boxes[:, 2:] / 2
  boxes[:, 2:] += boxes[:, :2]
  return boxes


class SSDHelper(BaseHelper):
  def __init__(self, image_ann: str,
               in_hw: tuple,
               class_num: int,
               anchor_widths: list,
               anchor_steps: list,
               pos_thresh: float,
               variances: float,
               **kwargs):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None
    self.hash_table: tf.lookup.StaticHashTable = None
    self.resize_method = kwargs.get('resize_method')
    self.augment_method = kwargs.get('augment_method')

    used_label_map = kwargs.get('used_label_map')
    if used_label_map != None:
      self.hash_table = YOLOHelper.get_hash_table(used_label_map)
    img_ann_list = np.load(image_ann, allow_pickle=True)[()]

    # NOTE can use dict set trian and test dataset
    self.train_list: str = img_ann_list['train_data']
    self.val_list: str = img_ann_list['val_data']
    self.test_list: str = img_ann_list['test_data']
    self.train_total_data: int = img_ann_list['train_num']
    self.val_total_data: int = img_ann_list['val_num']
    self.test_total_data: int = img_ann_list['test_num']
    self.anchor_widths = anchor_widths
    self.anchor_steps = anchor_steps
    self.anchors: _EagerTensorBase = tf.convert_to_tensor(
        self._get_anchors(in_hw, anchor_widths, anchor_steps), tf.float32)
    self.corner_anchors: _EagerTensorBase = tf_center_to_corner(self.anchors, False)
    self.anchors_num: int = self.anchors.shape.as_list()[0]
    self.class_num: int = class_num
    self.org_in_hw: np.ndarray = np.array(in_hw)
    self.in_hw: _EagerTensorBase = tf.Variable(self.org_in_hw, trainable=False)
    self.pos_thresh: float = pos_thresh
    self.variances: _EagerTensorBase = tf.convert_to_tensor(variances, tf.float32)

    self.iaaseq = iaa.Sequential([
        iaa.SomeOf([1, None], [
            iaa.Fliplr(0.5),
            iaa.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                       backend='cv2'),
            iaa.Affine(translate_percent={"x": (-0.15, 0.15),
                                          "y": (-0.15, 0.15)},
                       backend='cv2'),
            iaa.Affine(rotate=(-15, 15),
                       backend='cv2')
        ], True)
    ], True)

  @staticmethod
  def _get_anchors(in_hw: List[int],
                   anchor_widths: Iterable[Tuple[int, int]],
                   anchor_steps: Iterable[Tuple[int, int]]) -> np.ndarray:
    feature_maps = [[np.ceil(in_hw[0] / step).astype(np.int),
                     np.ceil(in_hw[1] / step).astype(np.int)] for step in anchor_steps]

    """ get anchors """
    anchors = []
    for k, f in enumerate(feature_maps):
      min_sizes = anchor_widths[k]
      for i, j in product(range(f[0]), range(f[1])):
        for min_size in min_sizes:
          s_kx = min_size[1] / in_hw[1]
          s_ky = min_size[0] / in_hw[0]
          dense_cx = [x * anchor_steps[k] / in_hw[1] for x in [j + 0.5]]
          dense_cy = [y * anchor_steps[k] / in_hw[0] for y in [i + 0.5]]
          for cy, cx in product(dense_cy, dense_cx):
            anchors += [cx, cy, s_kx, s_ky]

    anchors = np.array(anchors).reshape(-1, 4).astype('float32')
    return anchors

  def read_img(self, img_path: tf.string) -> tf.Tensor:
    return tf.image.decode_jpeg(tf.io.read_file(img_path), 3)

  def decode_img(self, img_str: bytes) -> tf.Tensor:
    return tf.image.decode_jpeg(img_str, channels=3)

  def process_img(self, img: np.ndarray, ann: np.ndarray, in_hw: np.ndarray,
                  is_augment: bool, is_resize: bool, is_normlize: bool
                  ) -> Tuple[np.ndarray, List[np.ndarray]]:
    if is_resize and is_augment:
      # print(INFO, f'resize_method : {self.resize_method}')
      if self.resize_method == 'origin':
        img, ann = YOLOHelper.origin_resize_train_img(img, in_hw, ann)
      elif self.resize_method == 'gluon':
        img, ann = YOLOHelper.gluon_resize_train_img(img, in_hw, ann)
      elif self.resize_method == 'none':
        img, ann = YOLOHelper.resize_img(img, in_hw, ann)
      else:
        raise ValueError(f'resize_method: {self.resize_method} is error')
    elif is_resize:
      img, ann = YOLOHelper.resize_img(img, in_hw, ann)
    if is_augment:
      # print(INFO, f'augment_method : {self.augment_method}')
      if self.augment_method == 'origin':
        img, ann = YOLOHelper.origin_augment_img(img, ann)
      elif self.augment_method == 'iaa':
        img, ann = tf.numpy_function(
            partial(YOLOHelper.iaa_augment_img, iaaseq=self.iaaseq),
            [img, ann], [tf.uint8, tf.float32])
      elif self.augment_method == 'autoaug':
        img, ann = YOLOHelper.auto_augment_img(img, ann)
      else:
        raise ValueError(f'augment_method: {self.augment_method} is error')
    if is_normlize:
      img = self.normlize_img(img)
    else:
      img = tf.cast(img, tf.float32)
    return img, ann

  def ann_to_label(self, bbox: tf.Tensor, clses: tf.Tensor, in_hw: tf.Tensor) -> List[tf.Tensor]:

    bbox = bbox / tf.tile(tf.cast(in_hw[::-1], tf.float32), [2])
    clses = clses + 1  # NOTE ssd defalut class 0 is background, so need +1

    overlaps = tf_bbox_iou(bbox, self.corner_anchors)
    best_prior_overlap = tf.reduce_max(overlaps, 1)
    best_prior_idx = tf.argmax(overlaps, 1, tf.int32)
    valid_gt_idx = tf.greater_equal(best_prior_overlap, 0.2)
    valid_gt_idx.set_shape([None])
    best_prior_idx_filter = tf.boolean_mask(best_prior_idx, valid_gt_idx, axis=0)

    def t_fn():
      label_loc = tf.zeros((self.anchors_num, 4))
      label_conf = tf.zeros((self.anchors_num, 1))
      return label_loc, label_conf

    def f_fn():
      best_truth_overlap = tf.reduce_max(overlaps, 0)
      best_truth_idx = tf.argmax(overlaps, 0, tf.int32)
      best_truth_overlap = tf.tensor_scatter_nd_update(
          best_truth_overlap, best_prior_idx_filter[:, None],
          tf.ones_like(best_prior_idx_filter, tf.float32) * 2.)
      best_truth_idx = tf.tensor_scatter_nd_update(
          best_truth_idx, best_prior_idx[:, None],
          tf.range(tf.size(best_prior_idx), dtype=tf.int32))

      matches = tf.gather(bbox, best_truth_idx)
      label_conf = tf.gather(clses, best_truth_idx)
      # filter gt and anchor overlap less than pos_thresh, set as background
      label_conf = tf.where(best_truth_overlap[:, None] < self.pos_thresh,
                            tf.zeros_like(label_conf), label_conf)
      # encode matches gt to network label
      label_loc = tf_encode_bbox(matches, self.anchors, self.variances)
      return label_loc, label_conf

    return tf.cond(tf.less_equal(tf.size(best_prior_idx_filter), 0), t_fn, f_fn)

  def draw_image(self, img: tf.Tensor, ann: List[tf.Tensor],
                 is_show: bool = True) -> np.ndarray:
    bbox, score, clses = ann
    if isinstance(img, EagerTensor):
      img = img.numpy()
      bbox = bbox.numpy()
      score = score.numpy()
      clses = clses.numpy()
    for i, cls_idx in enumerate(clses):
      cv2.rectangle(img, tuple(bbox[i][:2].astype(int)),
                    tuple(bbox[i][2:].astype(int)),
                    (255, 0, 0))
      cv2.putText(img, str(cls_idx),
                  tuple(bbox[i][2:].astype(int) - [18, 10]),
                  cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), thickness=1)
    if is_show:
      imshow(img)
      show()
    return img

  def build_datapipe(self, image_ann_list: np.ndarray,
                     batch_size: int,
                     is_augment: bool, is_normlize: bool,
                     is_training: bool) -> tf.data.Dataset:
    print(INFO, 'data augment is ', str(is_augment))
    print(INFO, 'use hash map is ', str(True if self.hash_table else False))
    parser_example_fn = (partial(YOLOHelper.parser_example_with_hash_table,
                                 table=self.hash_table)
                         if self.hash_table
                         else YOLOHelper.parser_example)

    def _parser_wrapper(stream: bytes) -> tf.Tensor:
      img_str, _, ann, _ = parser_example_fn(stream)
      img = self.decode_img(img_str)
      img, ann = self.process_img(img, ann, self.in_hw,
                                  is_augment, True, is_normlize)
      clses, bbox = tf.split(ann, [1, 4], axis=1)

      label = tf.concat(self.ann_to_label(bbox, clses, in_hw=self.in_hw), -1)

      img.set_shape((None, None, 3))
      label.set_shape((None, 5))
      return img, label

    if is_training:
      dataset = (tf.data.TFRecordDataset(image_ann_list, num_parallel_reads=4).
                 shuffle(batch_size * 500).
                 repeat().
                 map(_parser_wrapper, -1).
                 batch(batch_size, True).
                 prefetch(-1))
    else:
      dataset = (tf.data.TFRecordDataset(image_ann_list, num_parallel_reads=4).
                 map(_parser_wrapper, -1).
                 batch(batch_size, True).
                 prefetch(-1))

    return dataset


class SSDLoss(tf.keras.losses.Loss):
  def __init__(self, h: SSDHelper, loc_weight=1, conf_weight=1,
               negpos_ratio=7, reduction='auto', name=None):
    """ SSD Weighted Loss
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    super().__init__(reduction, name)
    self.negpos_ratio = negpos_ratio
    self.h = h
    self.loc_weight = loc_weight
    self.conf_weight = conf_weight
    self.anchors = self.h.anchors
    self.anchors_num = self.h.anchors_num
    self.class_num = self.h.class_num
    self.op_list = []
    names = ['loc', 'conf']
    self.lookups: Iterable[Tuple[ResourceVariable, AnyStr]] = [
        (tf.Variable(0, name=name, shape=(),
                     dtype=tf.float32, trainable=False), name)
        for name in names]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bc_num = tf.shape(y_pred)[0]
    loc_data, conf_data = tf.split(y_pred, [4, 1 + self.class_num], -1)
    loc_t, conf_t = tf.split(y_true, [4, 1], -1)

    # find have bbox but no landmark location
    pos_conf_mask = tf.not_equal(conf_t, 0)
    conf_t = tf.cast(conf_t, tf.int32)
    # Localization Loss (Smooth L1)
    pos_loc_mask = tf.tile(pos_conf_mask, [1, 1, 4])
    loc_p = tf.reshape(tf.boolean_mask(loc_data, pos_loc_mask), (-1, 4))  # 792,4
    loc_t = tf.reshape(tf.boolean_mask(loc_t, pos_loc_mask), (-1, 4))
    loss_loc = tf.reduce_sum(huber_loss(loc_p, loc_t))

    # Compute max conf across batch for hard negative mining
    batch_conf = tf.reshape(conf_data, (-1, self.class_num + 1))  # 10,16800,2 -> 10*16800,2
    loss_conf = (tf.reduce_logsumexp(batch_conf, 1, True) -
                 tf.gather_nd(
                     batch_conf,
        tf.concat([tf.range(tf.shape(batch_conf)[0])[:, None],
                   tf.reshape(conf_t, (-1, 1))], 1))[:, None])

    # Hard Negative Mining
    loss_conf = loss_conf * tf.reshape(tf.cast(tf.logical_not(pos_conf_mask), tf.float32), (-1, 1))
    loss_conf = tf.reshape(loss_conf, (bc_num, -1))
    idx_rank = tf.argsort(tf.argsort(loss_conf, 1, direction='DESCENDING'), 1)

    num_pos_conf = tf.reduce_sum(tf.cast(pos_conf_mask, tf.float32), 1)
    num_neg_conf = tf.minimum(self.negpos_ratio * num_pos_conf,
                              tf.cast(tf.shape(pos_conf_mask)[1], tf.float32) - 1.)
    neg_conf_mask = tf.less(tf.cast(idx_rank, tf.float32),
                            tf.tile(num_neg_conf, [1, tf.shape(pos_conf_mask)[1]]))[..., None]

    # calc pos , neg confidence loss
    pos_idx = tf.tile(pos_conf_mask, [1, 1, self.class_num + 1])
    neg_idx = tf.tile(neg_conf_mask, [1, 1, self.class_num + 1])

    conf_p = tf.reshape(tf.boolean_mask(
        conf_data,
        tf.equal(tf.logical_or(pos_idx, neg_idx), True)), (-1, self.class_num + 1))
    conf_t = tf.boolean_mask(
        conf_t,
        tf.equal(tf.logical_or(pos_conf_mask, neg_conf_mask), True))

    loss_conf = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(conf_t, conf_p))

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / num_pos_conf
    num_pos_conf = tf.maximum(tf.reduce_sum(num_pos_conf), 1)  # 正样本个数
    loss_loc = self.loc_weight * (loss_loc / num_pos_conf)
    loss_conf = self.conf_weight * (loss_conf / num_pos_conf)
    self.op_list.extend([
        self.lookups[0][0].assign(loss_loc),
        self.lookups[1][0].assign(loss_conf)])
    with tf.control_dependencies(self.op_list):
      total_loss = loss_loc + loss_conf
    return total_loss


def parser_outputs(outputs: List[np.ndarray], orig_hws: List[np.ndarray], obj_thresh: float,
                   nms_thresh: float, batch: int, h: SSDHelper
                   ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  bbox_outs, class_outs = outputs
  results = []
  for bbox, clses, orig_hw in zip(bbox_outs, class_outs, orig_hws):
    """ softmax class"""
    clses = softmax(clses, -1)
    """ decode bbox """
    bbox = decode_bbox(bbox, h.anchors.numpy(),
                       h.variances.numpy())
    bbox = bbox * np.tile(h.org_in_hw[::-1], [2])
    """ reverse img """
    scale = np.min(h.org_in_hw / np.array(orig_hw))
    xy_off = ((h.org_in_hw - np.array(orig_hw) * scale) / 2)[::-1]
    bbox = (bbox - np.tile(xy_off, [2])) / scale

    _scores = np.max(clses[:, 1:], axis=-1)
    _boxes = bbox[_scores > obj_thresh]
    _classes = np.argmax(clses[:, 1:], axis=-1)[_scores > obj_thresh]
    _scores = _scores[_scores > obj_thresh]

    """ do nms """
    """ keep top-k before NMS """
    order = np.argsort(_scores)[::-1]
    _boxes = _boxes[order]
    _scores = _scores[order]
    _classes = _classes[order]
    keep = nms_oneclass(_boxes, _scores, nms_thresh)
    _boxes = _boxes[keep]
    _scores = _scores[keep]
    _classes = _classes[keep]
    results.append([_boxes, _scores, _classes])

  return results


def ssd_infer(img_path: Path, infer_model: tf.keras.Model,
              result_path: Path, h: SSDHelper,
              obj_thresh: float = 0.6,
              nms_thresh: float = 0.5):
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
      ncc_results = np.array([np.fromfile(
          str(result_path / (Path(img_paths[i]).stem + '.bin')),
          dtype='float32') for i in range(len(img_paths))])  # type:np.ndarray
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
    det_img, *_ = h.process_img(img, np.zeros((0, 5), np.float32), h.in_hw, False, True, True)
    det_imgs.append(det_img)
  batch = len(det_imgs)
  outputs = infer_model.predict(tf.stack(det_imgs), batch)

  """ parser batch out """
  results = parser_outputs(np.split(outputs, [4], -1),
                           orig_hws, obj_thresh, nms_thresh, batch, h)

  if result_path is None:
    """ draw gpu result """
    for img_path, (bbox, score, clses) in zip(img_paths, results):
      draw_img = h.read_img(img_path)
      h.draw_image(draw_img.numpy(), [bbox, np.ones_like(score[:, None]), clses])
  else:
    """ draw gpu result and nncase result """
    ncc_preds = [[], []]
    for ncc_result in ncc_results:
      # NOTE ncc result is (N,C,H,W) ,but output bbox no `Channels` , so is [anchor_num,4]
      ncc_result = np.reshape(ncc_result, (len(outputs[0]), -1))
      ncc_result = np.split(ncc_result, [4], -1)
      ncc_preds[0].append(ncc_result[0])
      ncc_preds[1].append(ncc_result[1])
    ncc_preds = [np.stack(pred) for pred in ncc_preds]
    ncc_results = parser_outputs(ncc_preds, orig_hws, obj_thresh, nms_thresh, batch, h)
    for img_path, (bbox, score), (ncc_bbox, ncc_score) in zip(img_paths, results, ncc_results):
      draw_img = h.read_img(img_path)
      gpu_img = h.draw_image(draw_img.numpy(), [bbox, np.ones_like(score[:, None])], False)
      ncc_img = h.draw_image(draw_img.numpy(), [ncc_bbox, np.ones_like(ncc_score[:, None])], False)
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
