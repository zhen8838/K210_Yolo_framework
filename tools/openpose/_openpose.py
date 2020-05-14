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
                                             pose_crop_random)
from transforms.image import ops as image_ops


class OpenPoseHelper(BaseHelperV2):

  def set_datasetlist(self):
    meta = np.load(self.dataset_root, allow_pickle=True)[()]

    self.epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    self.test_list: str = None
    self.unlabel_list: str = None

    self.train_list: Tuple[tf.Tensor, tf.RaggedTensor] = (
        tf.constant(meta['train_list'][0], tf.string),
        tf.ragged.constant(meta['train_list'][1],
                           inner_shape=(19, 2),
                           dtype=tf.float32))

    self.train_total_data: int = meta['train_num']

    self.val_list: Tuple[tf.Tensor, tf.RaggedTensor] = (
        tf.constant(meta['val_list'][0], tf.string),
        tf.ragged.constant(meta['val_list'][1],
                           inner_shape=(19, 2),
                           dtype=tf.float32))

    self.val_total_data: int = meta['val_num']

    self.test_list = self.val_list
    self.test_total_data = self.val_total_data

    del meta  # NOTE free memory

    self.target_hw: Tuple[int, int] = None
    if hasattr(self.hparams, 'scale'):
      self.target_hw = (self.in_hw[0] // self.hparams.scale,
                        self.in_hw[1] // self.hparams.scale)

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:

    def np_process(img, joint_list):
      meta = ImageMeta(img, joint_list, self.in_hw,
                       self.hparams.parts, self.hparams.vecs, self.hparams.sigma)
      if is_augment:
        meta = pose_random_scale(meta)
        # meta = pose_rotation(meta)
        meta = pose_flip(meta)
        meta = pose_resize_shortestedge_random(meta)
        meta = pose_crop_random(meta)
      heatmap = meta.get_heatmap(self.target_hw)
      vectormap = meta.get_vectormap(self.target_hw)
      # NOTE meta.img, heatmap, vectormap type: uint8, float16, float16
      return meta.img, heatmap, vectormap

    def parser(path, joint_list):
      img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
      img, heatmap, vectormap = tf.numpy_function(
          np_process,
          [img, joint_list],
          [tf.uint8, tf.float16, tf.float16])
      if is_normalize:
        img = image_ops.normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      return img, heatmap, vectormap

    ds = (tf.data.Dataset.from_tensor_slices(self.train_list).
          shuffle(batch_size * 200).
          repeat().
          map(parser, num_parallel_calls=-1).
          batch(batch_size).
          prefetch(tf.data.experimental.AUTOTUNE))
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    return self.build_train_datapipe(batch_size, False, is_normalize)

  # def get_heatmap(self, im_h, im_w, joint_list, th=4.6052, sigma=8.):

  #   target_size = (self.in_hw[0] // self.hparams.scale, self.in_hw[1] // self.hparams.scale)

  #   heatmap: tf.Variable = tf.Variable(tf.zeros((self.hparams.parts, im_h, im_w)), trainable=False)

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

  #   if target_size:
  #     heatmap_tensor = tf.image.resize(heatmap_tensor, target_size)

  #   return tf.cast(heatmap_tensor, tf.float16)

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

      with tf.GradientTape() as tape:
        logit_sup = self.train_model(image, training=True)

        loss = (
            loss_xe + self.hparams.fixmatchmixup.wu * loss_xeu +
            self.hparams.fixmatchmixup.wmu * loss_xeu_mix + loss_wd)

      scaled_loss = self.optimizer_minimize(loss, tape, self.optimizer,
                                            self.train_model)

      if self.hparams.ema.enable:
        self.ema.update()


      metrics.loss.update_state(scaled_loss)
      metrics.acc.update_state(sup_label, logit_sup)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      datas, labels = inputs['data'], inputs['label']
      logits = self.val_model(datas, training=False)
      loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
      loss_xe = tf.reduce_mean(loss_xe)
      loss_wd = tf.reduce_sum(self.val_model.losses)
      loss = loss_xe + loss_wd
      metrics.loss.update_state(loss)
      metrics.acc.update_state(labels, logits)

    for inputs in dataset:
      self.strategy.experimental_run_v2(step_fn, args=(inputs,))
