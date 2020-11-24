import tensorflow as tf
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from tools.ssd import SSDHelper, SSDLoss, YOLOHelper
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List
from toolz import reduce, map, partial
from itertools import product
from models.networks4k210 import ullfd_k210
k = tf.keras
kl = tf.keras.layers
np.set_printoptions(suppress=True)


def test_process_img():
  h = SSDHelper('/home/zqh/workspace/mix-tfrecord/example.npy',
                [240, 320], 2,
                [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
                [8, 16, 32], 0.35, [0.1, 0.2],
                used_label_map={'No-Mask': 1,
                                'Mask': 0},
                resize_method='gluon',
                augment_method='origin')

  parser_example_fn = (partial(
      YOLOHelper.parser_example_with_hash_table,
      table=h.hash_table)
      if h.hash_table
      else YOLOHelper.parser_example)
  is_augment = True
  is_normlize = True

  def _parser_wrapper(stream: bytes) -> tf.Tensor:
    img_str, _, ann, _ = parser_example_fn(stream)
    img = h.decode_img(img_str)
    img, ann = h.process_img(img, ann, h.in_hw,
                             is_augment, True, is_normlize)
    # label = tf.concat(h.ann_to_label(*ann, in_hw=h.in_hw), -1)
    # img.set_shape((None, None, 3))
    # label.set_shape((None, 5))
    return img, ann

  ds = tf.data.TFRecordDataset(h.train_list, num_parallel_reads=4).map(_parser_wrapper, -1)
  iters = iter(ds)
  img, ann = next(iters)


def test_train_ssd():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  h = SSDHelper('/home/zqh/workspace/mix-tfrecord/example.npy',
                [240, 320], 2,
                [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
                [8, 16, 32], 0.35, [0.1, 0.2],
                used_label_map={'No-Mask': 1,
                                'Mask': 0},
                resize_method='gluon',
                augment_method='origin')

  h.set_dataset(4, True, True, True)
  ds = h.train_dataset.take(100)
  infer_model, train_model = ullfd_k210((240, 320, 3), 2, [[[16, 16], [32, 32]], [
      [64, 64], [128, 128]], [[256, 256], [512, 512]]])
  ssdloss = SSDLoss(h)
  iters = iter(ds)
  # for batch in ds:

  image, y_true = next(iters)
  # cles = y_true[..., -1]
  # cles[cles > 0] mask to the 
  y_pred = train_model(image, training=True)
  # y_true 出来有500多了，肯定哪里出错了
  loss = ssdloss(y_true, y_pred)
  loss
