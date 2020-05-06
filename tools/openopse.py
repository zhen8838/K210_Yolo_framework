import tensorflow as tf
from tools.training_engine import BaseTrainingLoop, EmaHelper, BaseHelperV2
from typing import Tuple, List
from pathlib import Path
import numpy as np


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

  def set_dataset(self, batch_size, is_augment, is_normalize=True):
    return super().set_dataset(batch_size, is_augment, is_normalize=is_normalize)

  def get_heatmap(self, im_h, im_w, joint_list, th=4.6052, sigma=8.):
    target_size = (self.in_hw[0] // self.hparams.scale, self.in_hw[1] // self.hparams.scale)

    heatmap: tf.Variable = tf.Variable(tf.zeros((im_h, im_w, self.hparams.parts)), trainable=False)

    for joints in joint_list:
      for i, center in enumerate(joints):
        if center[0] < 0 or center[1] < 0:
          continue

        delta = tf.sqrt(th * 2)
        # p0 -> x,y    p1 -> x,y
        im_wh = tf.cast((im_w, im_h), tf.float32)
        p0 = tf.cast(tf.maximum(0., center - delta * sigma), tf.int32)
        p1 = tf.cast(tf.minimum(im_wh, center + delta * sigma), tf.int32)

        x = tf.range(p0[0], p1[0])[None, :, None]
        y = tf.range(p0[1], p1[1])[:, None, None]

        p = tf.concat([x + tf.zeros_like(y), tf.zeros_like(x) + y], axis=-1)
        exp = tf.reduce_sum(tf.square(tf.cast(p, tf.float32) - center), -1) / (2. * sigma * sigma)
        # use indices update point area
        indices = tf.concat([p[..., ::-1],
                             tf.ones(p.shape[:-1] + [1], tf.int32) * i], -1)
        # NOTE p is [x,y] , but `gather_nd` and `scatter_nd` require [y,x]
        old_center_area = tf.gather_nd(heatmap, indices)
        center_area = tf.minimum(tf.maximum(old_center_area, tf.exp(-exp)), 1.0)
        center_area = tf.where(exp > th, old_center_area, center_area)

        heatmap.scatter_nd_update(indices, center_area)
    # use indices update heatmap background
    x = tf.range(0, im_w)[None, :, None]
    y = tf.range(0, im_h)[:, None, None]
    yx = tf.concat([tf.zeros_like(x) + y, x + tf.zeros_like(y)], axis=-1)
    # NOTE scatter_nd can't use -1
    indices = tf.concat([yx, tf.ones(yx.shape[:-1] + [1], tf.int32) * (self.hparams.parts - 1)], -1)
    heatmap.scatter_nd_update(
        indices,
        tf.clip_by_value(1. - tf.reduce_max(heatmap, axis=-1), 0., 1.))

    heatmap_tensor = heatmap.read_value()

    heatmap_tensor = tf.transpose(heatmap_tensor, (1, 2, 0))

    # background
    heatmap_tensor[:, :, -1] = tf.clip_by_value(1 -
                                                tf.reduce_max(heatmap_tensor, axis=-1), 0.0, 1.0)

    if target_size:
      heatmap_tensor = tf.image.resize(heatmap_tensor, target_size)

    return tf.cast(heatmap_tensor, tf.float16)
