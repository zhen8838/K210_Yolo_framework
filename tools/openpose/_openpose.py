import tensorflow as tf
from tools.training_engine import BaseTrainingLoop, EmaHelper, BaseHelperV2
from typing import Tuple, List
from pathlib import Path
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import random
from tools.openpose.openopse_agument import (ImageMeta, pose_random_scale,
                                             pose_flip,
                                             pose_resize_shortestedge_random,
                                             pose_crop_random,
                                             pose_resize_shortestedge_fixed,
                                             pose_crop_center)
from transforms.image import ops as image_ops


class OpenPoseHelper(BaseHelperV2):
  """OpenPoseHelper

    hparams:
      scale: 8 # output height width reduce scale, for mbv1 is 8
      sigma: 8. # heatmap gaussian sigma
      parts: 19 # dataset point part number, for coco is 19
      vecs: [[2, 9],[9, 10],[10, 11],[2, 12],[12, 13], [13, 14], [2, 3], [3, 4], [4, 5], [3, 17], [2, 6], [6, 7], [7, 8], [6, 18], [2, 1], [1, 15], [1, 16], [15, 17], [16, 18]] # dataset point line vector
  """

  def set_datasetlist(self):
    meta = np.load(self.dataset_root, allow_pickle=True)[()]

    self.epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    self.test_list: str = None
    self.unlabel_list: str = None

    self.train_list: str = meta['train_list']
    # self.train_list: Tuple[tf.Tensor, tf.RaggedTensor] = (
    #     tf.constant(meta['train_list'][0], tf.string),
    #     tf.ragged.constant(meta['train_list'][1],
    #                        inner_shape=(19, 2),
    #                        dtype=tf.float32))

    self.train_total_data: int = meta['train_num']
    self.val_list: str = meta['val_list']
    # self.val_list: Tuple[tf.Tensor, tf.RaggedTensor] = (
    #     tf.constant(meta['val_list'][0], tf.string),
    #     tf.ragged.constant(meta['val_list'][1],
    #                        inner_shape=(19, 2),
    #                        dtype=tf.float32))

    self.val_total_data: int = meta['val_num']

    self.test_list = self.val_list
    self.test_total_data = self.val_total_data

    del meta  # NOTE free memory

    self.target_hw: Tuple[int, int] = None
    if hasattr(self.hparams, 'scale'):
      self.target_hw = (self.in_hw[0] // self.hparams.scale,
                        self.in_hw[1] // self.hparams.scale)
    # NOTE self.hparams.vecs need be numpy array
    self.hparams.vecs = np.array(self.hparams.vecs) - 1

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True,
                           is_train: bool = True) -> tf.data.Dataset:

    def np_process(img, joint_list):
      meta = ImageMeta(img, joint_list, self.in_hw,
                       self.hparams.parts, self.hparams.vecs, self.hparams.sigma)
      if is_augment:
        meta = pose_random_scale(meta)
        # meta = pose_rotation(meta)
        meta = pose_flip(meta)
        meta = pose_resize_shortestedge_random(meta)
        meta = pose_crop_random(meta)
      else:
        meta = pose_resize_shortestedge_fixed(meta)
        meta = pose_crop_center(meta)
      heatmap = meta.get_heatmap_v(self.target_hw)
      vectormap = meta.get_vectormap_v(self.target_hw)
      # NOTE meta.img, heatmap, vectormap type: uint8, float32, float32
      return meta.img, heatmap, vectormap

    # NOTE 还是有必要使用filter_fn过滤一些没必要训练的图像。
    def filter_fn(img_str, joint_list):
      return tf.logical_not(tf.logical_and(tf.size(joint_list) == 0, tf.random.uniform([]) > 0.2))

    def parse_example(raw_example):
      example = tf.io.parse_single_example(raw_example, {
          'img': tf.io.FixedLenFeature([], dtype=tf.string),
          'joint': tf.io.VarLenFeature(dtype=tf.float32)
      })
      return example['img'], example['joint'].values

    def parser(img_str, joint_list):
      img = tf.image.decode_jpeg(img_str, channels=3)
      joint_list = tf.reshape(joint_list, [-1, self.hparams.parts, 2])
      img, heatmap, vectormap = tf.numpy_function(
          np_process,
          [img, joint_list],
          [tf.uint8, tf.float32, tf.float32])
      img.set_shape([self.in_hw[0], self.in_hw[1], 3])
      heatmap.set_shape([self.target_hw[0], self.target_hw[1], self.hparams.parts])
      vectormap.set_shape([self.target_hw[0], self.target_hw[1], self.hparams.parts * 2])
      if is_normalize:
        img = image_ops.normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      return img, heatmap, vectormap

    if is_train:
      ds = (tf.data.TFRecordDataset(self.train_list, num_parallel_reads=4).
            shuffle(batch_size * 200).
            repeat().
            map(parse_example, num_parallel_calls=-1).
            filter(filter_fn).
            map(parser, num_parallel_calls=-1).
            batch(batch_size).
            prefetch(tf.data.experimental.AUTOTUNE))
    else:
      ds = (tf.data.TFRecordDataset(self.val_list, num_parallel_reads=4).
            shuffle(batch_size * 200).
            map(parse_example, num_parallel_calls=-1).
            map(parser, num_parallel_calls=-1).
            batch(batch_size).
            prefetch(tf.data.experimental.AUTOTUNE))

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    return self.build_train_datapipe(batch_size, False, is_normalize, is_train=False)

  # def get_heatmap(self, im_h, im_w, joint_list, th=4.6052, sigma=8.):

  #   heatmap: tf.Variable = tf.Variable(np.zeros((self.hparams.parts, im_h, im_w)), trainable=False)

  #   for joints in joint_list:
  #     for i, center in enumerate(joints):
  #       if center[0] < 0 or center[1] < 0:
  #         continue

  #       delta = tf.sqrt(th * 2)
  #       # p0 -> x,y    p1 -> x,y
  #       im_wh = tf.cast((im_w, im_h), tf.float32)
  #       p0 = tf.cast(tf.maximum(0., center - delta * sigma), tf.int32)
  #       p1 = tf.cast(tf.minimum(im_wh, center + delta * sigma), tf.int32)

  #       x = tf.range(p0[0], p1[0])[None, :, None]
  #       y = tf.range(p0[1], p1[1])[:, None, None]

  #       p = tf.concat([x + tf.zeros_like(y), tf.zeros_like(x) + y], axis=-1)
  #       exp = tf.reduce_sum(tf.square(tf.cast(p, tf.float32) - center), -1) / (2. * sigma * sigma)
  #       # use indices update point area
  #       indices = tf.concat([tf.ones(p.shape[:-1] + [1], tf.int32) * i,
  #                            p[..., ::-1]], -1)
  #       # NOTE p is [x,y] , but `gather_nd` and `scatter_nd` require [y,x]
  #       old_center_area = tf.gather_nd(heatmap, indices)
  #       center_area = tf.minimum(tf.maximum(old_center_area, tf.exp(-exp)), 1.0)
  #       center_area = tf.where(exp > th, old_center_area, center_area)

  #       heatmap.scatter_nd_update(indices, center_area)
  #   # use indices update heatmap background NOTE scatter_nd can't use -1
  #   heatmap.scatter_update(tf.IndexedSlices(
  #       tf.clip_by_value(1. - tf.reduce_max(heatmap, axis=0), 0., 1.),
  #       self.hparams.parts - 1))

  #   heatmap_tensor = tf.transpose(heatmap, (1, 2, 0))

  #   if self.target_hw:
  #     heatmap_tensor = tf.image.resize(heatmap_tensor, self.target_hw)

  #   return heatmap_tensor

  # def get_vectormap(self, im_h, im_w, joint_list):
  #   vectormap = tf.Variable(np.zeros((self.hparams.parts * 2, im_h, im_w),
  #                                    dtype=np.float32), trainable=False)
  #   countmap = tf.Variable(np.zeros((self.hparams.parts, im_h, im_w),
  #                                   dtype=np.float32), trainable=False)

  #   for joints in joint_list:
  #     for plane_idx, (j_idx1, j_idx2) in enumerate(self.hparams.vecs):

  #       center_from = joints[j_idx1]
  #       center_to = joints[j_idx2]

  #       if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
  #         continue

  #       self.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

  #   # 通过countmap减小vectormap的数量级
  #   tile_countmap = tf.repeat(tf.cast(countmap.value(), tf.float32), 2, axis=0)
  #   div_vectormap = tf.math.divide_no_nan(vectormap.value(), tile_countmap)
  #   vectormap_tensor = tf.transpose(div_vectormap, [1, 2, 0])

  #   if self.target_hw:
  #     vectormap_tensor = tf.image.resize(vectormap_tensor, self.target_hw)

  #   return vectormap_tensor

  @staticmethod
  def put_vectormap(tfvectormap: tf.Variable, tfcountmap: tf.Variable,
                    plane_idx, center_from, center_to, threshold=8):
    """ tf版 """
    _, height, width = tfvectormap.shape[:3]

    # p0,p1,vector --> x,y
    vector = center_to - center_from
    p0 = tf.maximum(0, tf.cast(tf.minimum(center_from, center_to) - threshold, tf.int32))
    p1 = tf.minimum([width, height], tf.cast(
        tf.maximum(center_from, center_to) + threshold, tf.int32))

    norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=-1))

    if norm == 0:
      return
    vector = vector / norm
    # p --> x,y
    x = tf.range(p0[0], p1[0])[None, :, None]
    y = tf.range(p0[1], p1[1])[:, None, None]
    p = tf.concat([x + tf.zeros_like(y), tf.zeros_like(x) + y], axis=-1)
    bec = tf.cast(p - center_from, tf.float32)

    dist = tf.abs((bec[..., 0] * vector[..., 1]) - (bec[..., 1] * vector[..., 0]))

    plane_indices = tf.ones(p.shape[:-1] + [1], tf.int32)  # [h,w,1]
    # vmap_indices --> [2,area_h,area_w,3]  b,y,x
    vmap_indices = tf.stack([
        tf.concat([plane_indices * (plane_idx * 2 + 0), p[..., ::-1]], -1),
        tf.concat([plane_indices * (plane_idx * 2 + 1), p[..., ::-1]], -1)], -2)

    old_vamp_area = tf.gather_nd(tfvectormap, vmap_indices)
    # 通过boolean_mask过滤不满足要求的点,并进行更新
    valid_vmap_indices = tf.boolean_mask(vmap_indices, tf.less_equal(dist, threshold))
    tfvectormap.scatter_nd_update(valid_vmap_indices, tf.tile(
        vector[None, :], [valid_vmap_indices.shape[0], 1]))

    count_indices = tf.concat([tf.ones(p.shape[:-1] + [1], tf.int32) * plane_idx,
                               p[..., ::-1]], -1)
    valid_count_indices = tf.boolean_mask(count_indices, dist > threshold)
    tfcountmap.scatter_nd_add(valid_count_indices, tf.ones(valid_count_indices.shape[0]))

  @staticmethod
  def draw_test_image(inp, heatmap, vectmap):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    plt.imshow(OpenPoseHelper.get_bgimg(inp))

    a = fig.add_subplot(2, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(OpenPoseHelper.get_bgimg(inp, target_size=(
        heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
    tmp = np.amax(heatmap, axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = vectmap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    plt.imshow(OpenPoseHelper.get_bgimg(inp, target_size=(
        vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    plt.imshow(OpenPoseHelper.get_bgimg(inp, target_size=(
        vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    plt.show()

  @staticmethod
  def get_bgimg(inp, target_size=None):
    # inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_size:
      inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp


class OpenPoseLoop(BaseTrainingLoop):
  def set_metrics_dict(self):
    d = {
        'train': {
            'loss': tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'loss_lastlayer': tf.keras.metrics.Mean('lossl', dtype=tf.float32),
            'loss_lastlayer_paf': tf.keras.metrics.Mean('paf', dtype=tf.float32),
            'loss_lastlayer_heat': tf.keras.metrics.Mean('heat', dtype=tf.float32)
        },
        'val': {
            'loss': tf.keras.metrics.Mean('vloss', dtype=tf.float32),
            'loss_lastlayer': tf.keras.metrics.Mean('vlossl', dtype=tf.float32),
            'loss_lastlayer_paf': tf.keras.metrics.Mean('vpaf', dtype=tf.float32),
            'loss_lastlayer_heat': tf.keras.metrics.Mean('vheat', dtype=tf.float32)
        }
    }
    return d

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: List[tf.Tensor]):
      """Per-Replica training step function."""
      image, heatmap, vectormap = inputs
      batch_size = tf.cast(tf.shape(image)[0], tf.float32)
      with tf.GradientTape() as tape:
        outputs: List[tf.Tensor] = self.train_model(image, training=True)
        # outputs is [l1_vectmap,l1_vectmap,l2_vectmap,l2_vectmap,...]
        loss = []
        for (l1, l2) in outputs:
          loss_l1 = tf.nn.l2_loss(l1 - vectormap)
          loss_l2 = tf.nn.l2_loss(l2 - heatmap)
          loss.append(tf.reduce_mean([loss_l1, loss_l2]))
        loss_wd = tf.reduce_sum(self.train_model.losses)
        loss = tf.reduce_sum(loss) + loss_wd

      scaled_loss = self.optimizer_minimize(loss, tape,
                                            self.optimizer,
                                            self.train_model)

      if self.hparams.ema.enable:
        self.ema.update()
      # loss metric
      loss_ll_paf = tf.reduce_sum(loss_l1) / batch_size
      loss_ll_heat = tf.reduce_sum(loss_l2) / batch_size
      loss_ll = tf.reduce_sum([loss_ll_paf, loss_ll_heat])
      metrics.loss.update_state(scaled_loss)
      metrics.loss_lastlayer.update_state(loss_ll)
      metrics.loss_lastlayer_paf.update_state(loss_ll_paf)
      metrics.loss_lastlayer_heat.update_state(loss_ll_heat)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):
    if self.hparams.ema.enable:
      val_model = self.ema.model
    else:
      val_model = self.val_model

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      image, heatmap, vectormap = inputs
      batch_size = tf.cast(tf.shape(image)[0], tf.float32)
      l1, l2 = val_model(image, training=False)
      loss_l1 = tf.nn.l2_loss(l1 - vectormap)
      loss_l2 = tf.nn.l2_loss(l2 - heatmap)
      loss = tf.reduce_mean([loss_l1, loss_l2])

      loss_wd = tf.reduce_sum(val_model.losses)
      loss = tf.reduce_sum(loss) + loss_wd
      metrics.loss.update_state(loss)
      loss_ll_paf = tf.reduce_sum(loss_l1) / batch_size
      loss_ll_heat = tf.reduce_sum(loss_l2) / batch_size
      loss_ll = tf.reduce_sum([loss_ll_paf, loss_ll_heat])
      metrics.loss.update_state(loss)
      metrics.loss_lastlayer.update_state(loss_ll)
      metrics.loss_lastlayer_paf.update_state(loss_ll_paf)
      metrics.loss_lastlayer_heat.update_state(loss_ll_heat)

    for inputs in dataset:
      self.run_step_fn(step_fn, args=(inputs,))
