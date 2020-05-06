# %%
import tensorflow as tf
import sys
sys.path.insert(0, '/home/zqh/Documents/K210_Yolo_framework')
from models.darknet import compose, DarknetConv2D
import numpy as np
from tools.bbox_utils import center_to_corner, bbox_iou, tf_bbox_iou
from tools.retinaface import RetinaFaceHelper, RetinaFaceLoss, tf_encode_bbox, tf_encode_landm, parser_outputs, softmax, decode_bbox, decode_landm, nms_oneclass, huber_loss
from models.networks import retinafacenet, retinaface_slim
from models.networks4k210 import retinafacenet_k210, retinafacenet_k210_v1, retinafacenet_k210_v2, retinafacenet_k210_v3
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Tuple, List
from toolz import reduce, map
from itertools import product
# from make_anchor_list import runkMeans
k = tf.keras
kl = tf.keras.layers
np.set_printoptions(suppress=True)

# %%


def test_resize_img():
  """ 测试resize图像 """

  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 3
  for i in np.random.choice(np.arange(len(h.train_list)), 10):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)
    ann = tf.split(ann, [4, 10, 1], 1)
    img, *ann = h.resize_img(img, *ann, h.in_hw)
    h.draw_image(img, ann)


def test_conunt_img_anchor():
  """ 统计resize之后的图像anchor分布 """

  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[10, 10], [16, 16], [24, 24]], [[32, 32], [48, 48]], [[64, 64], [96, 96]],
       [[128, 128], [192, 192], [256, 256]]], [8, 16, 32, 64], 0.35, [0.1, 0.2])
  i = 3
  bboxs = []
  for i in range(len(h.train_list)):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)
    ann = tf.split(ann, [4, 10, 1], 1)
    img, *ann = h.resize_img(img, *ann, h.in_hw)
    bbox = ann[0] / np.tile(h.org_in_hw[::-1], [2])
    bboxs.append(bbox)
  bboxs = tf.concat(bboxs, 0)
  aw, ah = h.anchors[:, 2], h.anchors[:, 3]

  w = bboxs[:, 2:3] - bboxs[:, 0:1]
  h = bboxs[:, 3:4] - bboxs[:, 1:2]

  plt.scatter(w.numpy(), h.numpy())
  plt.scatter(aw.numpy(), ah.numpy())
  plt.show()


def test_resize_train_img():
  """ 测试resize 训练图像  NOTE 使用随机裁剪"""
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 20
  for i in np.random.choice(np.arange(len(h.train_list)), 10):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)
    ann = tf.split(ann, [4, 10, 1], 1)
    img, *ann = tf.numpy_function(h._crop_with_constraints, [img, *ann, h.in_hw],
                                  [tf.uint8, tf.float32, tf.float32, tf.float32])
    img, *ann = h.resize_img(img, *ann, h.in_hw)
    h.draw_image(img, ann)


def test_generate_quan_img():
  """ 利用训练图像生成量化数据集 NOTE 使用随机裁剪"""
  # h = RetinaFaceHelper(
  #     'data/retinaface_img_ann.npy', [240, 320],
  #     [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
  #     [8, 16, 32], 0.35, [0.1, 0.2])
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])

  i = 10
  d = Path(
      '/home/zqh/workspace/vim3l/aml_npu_sdk/acuity-toolkit/retinaface/data_2')
  for i, idx in enumerate(np.random.choice(np.arange(len(h.train_list)), 100)):
    img_path, ann = np.copy(h.train_list[idx])
    img = h.read_img(img_path)
    ann = tf.split(ann, [4, 10, 1], 1)
    img, *ann = tf.numpy_function(h._crop_with_constraints, [img, *ann, h.in_hw],
                                  [tf.uint8, tf.float32, tf.float32, tf.float32])
    img, *ann = h.resize_img(img, *ann, h.in_hw)
    tf.io.write_file(str(d / f'{i}.jpg'), tf.image.encode_jpeg(img))
    # h.draw_image(img, ann)


def test_augment_img():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 3
  in_hw = tf.convert_to_tensor(h.org_in_hw)
  for i in np.random.choice(np.arange(len(h.train_list)), 10):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)
    ann = tf.split(ann, [4, 10, 1], 1)
    img, *ann = tf.numpy_function(h._crop_with_constraints, [img, *ann, h.in_hw],
                                  [tf.uint8, tf.float32, tf.float32, tf.float32])
    img, *ann = h.resize_img(img, *ann, h.in_hw)
    img, *ann = tf.numpy_function(h.augment_img, [img, *ann],
                                  [tf.uint8, tf.float32, tf.float32, tf.float32])
    img = h.augment_img_color(img)
    h.draw_image(img, ann)


def test_ann_to_label():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 3
  for i in np.random.choice(np.arange(len(h.train_list)), 1000):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)  # 3279.8
    ann = tf.split(ann, [4, 10, 1], 1)  # 184.5
    img, *ann = tf.numpy_function(
        h._crop_with_constraints, [img, *ann, h.in_hw],
        [tf.uint8, tf.float32, tf.float32, tf.float32])  # 1168.7
    img, *ann = h.resize_img(img, *ann, h.in_hw)  # 8561.9
    img, *ann = tf.numpy_function(
        h.augment_img, [img, *ann],
        [tf.uint8, tf.float32, tf.float32, tf.float32])  # 11343.9
    img = h.augment_img_color(img)  # 4447.6
    label = h.ann_to_label(*ann, h.in_hw)  # 15431.0
    h.draw_image(img, ann)


def test_label_to_ann():
  """ 检测生成label转换为ann是否正确 """
  # ! 首先检查640x640时的转换
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 312
  img_path, ann = np.copy(h.train_list[i])
  img = h.read_img(img_path)  # 3279.8
  ann = tf.split(ann, [4, 10, 1], 1)  # 184.5
  img, *ann = h.resize_img(img, *ann, h.in_hw)  # 8561.9
  old_bbox, old_landm, old_conf = ann
  label = h.ann_to_label(*ann, h.in_hw)  # 15431.0
  label_loc, label_landm, label_conf = label
  tf.boolean_mask(label_loc, tf.equal(label_conf, 1)[:, 0])
  re_label_landm = decode_landm(label_landm.numpy(), h.anchors.numpy(),
                                h.variances.numpy())
  re_label_bbox = decode_bbox(label_loc.numpy(), h.anchors.numpy(),
                              h.variances.numpy())

  re_landm = tf.boolean_mask(re_label_landm, tf.equal(label_conf, 1)[:, 0])
  re_bbox = tf.boolean_mask(re_label_bbox, tf.equal(label_conf, 1)[:, 0])
  re_bbox = re_bbox * tf.tile(tf.cast(h.in_hw[::-1], tf.float32), [2])
  re_landm = re_landm * tf.tile(tf.cast(h.in_hw[::-1], tf.float32), [5])
  print(old_bbox)
  print(re_bbox)
  print(old_landm)
  print(re_landm)
  h.draw_image(img, ann)
  # ! 首先检查240x320时的转换
  h = RetinaFaceHelper('data/retinaface_img_ann.npy', [240, 320],
                       [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
                       [8, 16, 32, 64], 0.35, [0.1, 0.2])
  i = 312
  img_path, ann = np.copy(h.train_list[i])
  img = h.read_img(img_path)  # 3279.8
  ann = tf.split(ann, [4, 10, 1], 1)  # 184.5
  img, *ann = h.resize_img(img, *ann, h.in_hw)  # 8561.9
  old_bbox, old_landm, old_conf = ann
  label = h.ann_to_label(*ann, h.in_hw)  # 15431.0
  label_loc, label_landm, label_conf = label

  re_label_landm = decode_landm(label_landm.numpy(), h.anchors.numpy(),
                                h.variances.numpy())
  re_label_bbox = decode_bbox(label_loc.numpy(), h.anchors.numpy(),
                              h.variances.numpy())

  re_landm = tf.boolean_mask(re_label_landm, tf.equal(label_conf, 1)[:, 0])
  re_bbox = tf.boolean_mask(re_label_bbox, tf.equal(label_conf, 1)[:, 0])
  re_bbox = re_bbox * tf.tile(tf.cast(h.in_hw[::-1], tf.float32), [2])
  re_landm = re_landm * tf.tile(tf.cast(h.in_hw[::-1], tf.float32), [5])
  print(old_bbox)
  print(re_bbox)
  print(old_landm)
  print(re_landm)
  h.draw_image(img, ann)


def test_process_img():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 3
  for i in np.random.choice(np.arange(len(h.train_list)), 1000):
    img_path, ann = np.copy(h.train_list[i])
    img = h.read_img(img_path)  # 3279.8
    img, *ann = h.process_img(img, ann, h.in_hw, True, True, False)
    h.draw_image(img, ann)


def test_data_pipe_wrapper():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  i = 3
  image_ann_list = h.train_list
  is_augment = True
  is_normlize = False

  @tf.function
  def _wrapper(i: tf.Tensor) -> tf.Tensor:
    path, ann = tf.numpy_function(lambda idx: tuple(image_ann_list[idx]), [i],
                                  [tf.string, tf.float32])
    img = h.read_img(path)
    ann = tf.split(ann, [4, 10, 1], 1)

    if is_augment:
      img, *ann = tf.numpy_function(
          h._crop_with_constraints, [img, *ann, h.in_hw],
          [tf.uint8, tf.float32, tf.float32, tf.float32])
      img, *ann = h.resize_img(img, *ann, in_hw=h.in_hw)

      img, *ann = tf.numpy_function(
          h.augment_img, [img, *ann],
          [tf.uint8, tf.float32, tf.float32, tf.float32])
      img = h.augment_img_color(img)
    else:
      img, *ann = h.resize_img(img, *ann, in_hw=h.in_hw)
    if is_normlize:
      img = h.normlize_img(img)

    # img = tf.transpose(img, [2, 0, 1])

    label = tf.concat(h.ann_to_label(*ann, in_hw=h.in_hw), -1)

    # img.set_shape((3, None, None))
    # label.set_shape((None, 15))
    return img, label

  for i in np.random.choice(np.arange(len(h.train_list)), 20):
    img, label = _wrapper(tf.constant(i))
    plt.imshow(img.numpy())
    plt.show()


def test_tf_label_to_ann_compare():
  """ 检测移植后生成的label与原始代码生成label是否一致 """
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])

  for number in range(10):
    with np.load(
        f"/home/zqh/workspace/Pytorch_Retinaface/loss_match_{number}.npz",
            allow_pickle=True) as d:
      anchors = d['anchors']
      loc_t = d['loc_t']
      landm_t = d['landm_t']
      conf_t = d['conf_t']
      targets = d['targets']

    assert np.allclose(h.anchors, anchors)

    for i in range(len(targets)):
      bbox, landm, clses = tf.split(targets[i], [4, 10, 1], 1)
      # NOTE 这里对比之前需要把ann_to_label里面的除img_hw给注释掉.
      label_loc, label_landm, label_conf = h.ann_to_label(
          bbox, landm, clses, h.in_hw)
      val_idx = np.where(label_conf.numpy() > 0)[0]
      val_idx2 = np.where(conf_t[i] > 0)
      print(i)
      assert np.allclose(loc_t[i][val_idx2], label_loc.numpy()[val_idx])
      assert np.allclose(landm_t[i][val_idx2], label_landm.numpy()[val_idx])
      assert np.allclose(conf_t[i][val_idx2], label_conf.numpy()[val_idx])


def test_data_process_pipe_line():
  """ 测试数据生成的pipe line 以及时间 """
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  image_ann_list = h.train_list
  is_augment = True
  is_normlize = True

  @tf.function
  def _wrapper(i: tf.Tensor) -> tf.Tensor:
    path, ann = tf.numpy_function(lambda idx: tuple(image_ann_list[idx]), [i],
                                  [tf.string, tf.float32])
    img = h.read_img(path)
    ann = tf.split(ann, [4, 10, 1], 1)

    if is_augment:
      img, *ann = tf.numpy_function(
          h._crop_with_constraints, [img, *ann, h.in_hw],
          [tf.uint8, tf.float32, tf.float32, tf.float32])
      img, *ann = h.resize_img(img, *ann, h.in_hw)
    else:
      img, *ann = h.resize_img(img, *ann, h.in_hw)
    if is_augment:
      img, *ann = tf.numpy_function(
          h.augment_img, [img, *ann],
          [tf.uint8, tf.float32, tf.float32, tf.float32])
    if is_normlize:
      img = h.normlize_img(img)

    img = tf.transpose(img, [2, 0, 1])

    label = tf.concat(h.ann_to_label(*ann, in_hw=h.in_hw), -1)

    img.set_shape((3, None, None))
    label.set_shape((None, 15))

    return img, label

  i = 3
  for i in np.random.choice(np.arange(len(h.train_list)), 1000):
    _wrapper(tf.constant(i))
    # tuple : 46322.4
    # list : 46362.2
    # 分开赋值 : 46344.4


def dev_loss_compare():
  """ 测试移植后的loss和原始代码loss是否一样
    1. 首先根据标签生成label
    2. 再根据label计算loss
    """
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  lsfn = RetinaFaceLoss(h)

  def call(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bc_num = tf.shape(y_pred)[0]
    loc_data, landm_data, conf_data = tf.split(y_pred, [4, 10, 2], -1)
    loc_t, landm_t, conf_t = tf.split(y_true, [4, 10, 1], -1)
    # landmark loss
    pos_landm_mask = tf.greater(conf_t, 0.)  # get valid landmark num
    num_pos_landm = tf.maximum(
        tf.reduce_sum(tf.cast(pos_landm_mask, tf.float32)),
        1)  # sum pos landmark num
    pos_landm_mask = tf.tile(pos_landm_mask, [1, 1, 10])  # 10, 16800, 10
    # filter valid lanmark
    landm_p = tf.reshape(tf.boolean_mask(landm_data, pos_landm_mask), (-1, 10))
    landm_t = tf.reshape(tf.boolean_mask(landm_t, pos_landm_mask), (-1, 10))
    loss_landm = tf.reduce_sum(huber_loss(landm_t, landm_p))

    # find have bbox but no landmark location
    pos_conf_mask = tf.not_equal(conf_t, 0)
    # agjust conf_t, calc (have bbox,have landmark) and (have bbox,no landmark) location loss
    conf_t = tf.where(pos_conf_mask, tf.ones_like(conf_t, tf.int32),
                      tf.cast(conf_t, tf.int32))

    # Localization Loss (Smooth L1)
    pos_loc_mask = tf.tile(pos_conf_mask, [1, 1, 4])
    loc_p = tf.reshape(tf.boolean_mask(loc_data, pos_loc_mask), (-1, 4))  # 792,4
    loc_t = tf.reshape(tf.boolean_mask(loc_t, pos_loc_mask), (-1, 4))
    loss_loc = tf.reduce_sum(huber_loss(loc_p, loc_t))

    # Compute max conf across batch for hard negative mining
    batch_conf = tf.reshape(conf_data, (-1, 2))  # 10,16800,2 -> 10*16800,2
    loss_conf = (
        tf.reduce_logsumexp(batch_conf, 1, True) - tf.gather_nd(
            batch_conf,
            tf.concat([
                tf.range(tf.shape(batch_conf)[0])[:, None],
                tf.reshape(conf_t, (-1, 1))
            ], 1))[:, None])

    # Hard Negative Mining
    loss_conf = loss_conf * tf.reshape(
        tf.cast(tf.logical_not(pos_conf_mask), tf.float32), (-1, 1))
    loss_conf = tf.reshape(loss_conf, (bc_num, -1))
    idx_rank = tf.argsort(tf.argsort(loss_conf, 1, direction='DESCENDING'), 1)

    num_pos_conf = tf.reduce_sum(tf.cast(pos_conf_mask, tf.float32), 1)
    num_neg_conf = tf.minimum(
        lsfn.negpos_ratio * num_pos_conf,
        tf.cast(tf.shape(pos_conf_mask)[1], tf.float32) - 1.)
    neg_conf_mask = tf.less(
        tf.cast(idx_rank, tf.float32),
        tf.tile(num_neg_conf, [1, tf.shape(pos_conf_mask)[1]]))[..., None]

    # calc pos , neg confidence loss
    pos_idx = tf.tile(pos_conf_mask, [1, 1, 2])
    neg_idx = tf.tile(neg_conf_mask, [1, 1, 2])

    conf_p = tf.reshape(
        tf.boolean_mask(conf_data,
                        tf.equal(tf.logical_or(pos_idx, neg_idx), True)), (-1, 2))
    conf_t = tf.boolean_mask(
        conf_t, tf.equal(tf.logical_or(pos_conf_mask, neg_conf_mask), True))

    loss_conf = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(conf_t, conf_p))

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / num_pos_conf
    num_pos_conf = tf.maximum(tf.reduce_sum(num_pos_conf), 1)  # 正样本个数
    loss_loc /= num_pos_conf
    loss_conf /= num_pos_conf
    loss_landm /= num_pos_landm

    return loss_loc, loss_landm, loss_conf

  number = 4
  for number in range(10):
    with np.load(
        f'/home/zqh/workspace/Pytorch_Retinaface/loss_test_{number}.npz',
            allow_pickle=True) as d:
      loc_data: np.ndarray = d['loc_data']
      conf_data: np.ndarray = d['conf_data']
      landm_data: np.ndarray = d['landm_data']
      true_loc_t: np.ndarray = d['loc_t']
      true_landm_t: np.ndarray = d['landm_t']
      true_conf_t: np.ndarray = d['conf_t']
      true_loss_l: np.ndarray = d['loss_l'][()]  # 4.30260324
      true_loss_c: np.ndarray = d['loss_c'][()]  # 9.97558022
      true_loss_landm: np.ndarray = d['loss_landm'][()]  # 19.94988823
      targets: np.ndarray = d['targets']
    """ 1.生成label """
    ll = [[], [], []]
    for ann in targets:
      # NOTE 这里对比之前需要把ann_to_label里面的除img_hw给注释掉.
      for l, item in zip(
              ll, h.ann_to_label(*tf.split(ann, [4, 10, 1], 1), in_hw=h.in_hw)):
        l.append(item)
    loc_t, landm_t, conf_t = list(map(lambda l: np.stack(l), ll))

    true_y_true = np.concatenate(
        [true_loc_t, true_landm_t, true_conf_t[:, :, None]], -1)

    y_true = tf.convert_to_tensor(
        np.concatenate([loc_t, landm_t, conf_t], -1), tf.float32)
    """ 先对比label是否相同 """
    assert np.allclose(y_true.numpy(), true_y_true.astype(np.float32))

    y_pred = tf.convert_to_tensor(
        np.concatenate([loc_data, landm_data, conf_data], -1), tf.float32)

    loss_loc, loss_landm, loss_conf = call(y_true, y_pred)
    loss = loss_loc + loss_landm + loss_conf
    try:
      """ 有时候会有0.1的差异。排查中 """
      assert np.allclose(loss.numpy(),
                         true_loss_l + true_loss_c + true_loss_landm)
    except AssertionError:
      # print(f'{loss.numpy()}!={true_loss_l + true_loss_c + true_loss_landm} {true_loss_l} {true_loss_c} {true_loss_landm}')
      print(number)
      print(true_loss_l, true_loss_landm, true_loss_c)
      print(loss_loc.numpy(), loss_landm.numpy(), loss_conf.numpy())


def test_loss_compare():
  """ 测试移植后的loss和原始代码loss是否一样
    1. 首先根据标签生成label
    2. 再根据label计算loss
    """
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  lsfn = RetinaFaceLoss(h)
  for number in range(10):
    with np.load(
        f'/home/zqh/workspace/Pytorch_Retinaface/loss_test_{number}.npz',
            allow_pickle=True) as d:
      loc_data: np.ndarray = d['loc_data']
      conf_data: np.ndarray = d['conf_data']
      landm_data: np.ndarray = d['landm_data']
      true_loc_t: np.ndarray = d['loc_t']
      true_landm_t: np.ndarray = d['landm_t']
      true_conf_t: np.ndarray = d['conf_t']
      true_loss_l: np.ndarray = d['loss_l'][()]  # 4.30260324
      true_loss_c: np.ndarray = d['loss_c'][()]  # 9.97558022
      true_loss_landm: np.ndarray = d['loss_landm'][()]  # 19.94988823
      targets: np.ndarray = d['targets']
    """ 1.生成label """
    ll = [[], [], []]
    for ann in targets:
      # NOTE 这里对比之前需要把ann_to_label里面的除img_hw给注释掉.
      for l, item in zip(
              ll, h.ann_to_label(*tf.split(ann, [4, 10, 1], 1), in_hw=h.in_hw)):
        l.append(item)
    loc_t, landm_t, conf_t = list(map(lambda l: np.stack(l), ll))
    y_true = tf.convert_to_tensor(
        np.concatenate([loc_t, landm_t, conf_t], -1), tf.float32)
    y_pred = tf.convert_to_tensor(
        np.concatenate([loc_data, landm_data, conf_data], -1), tf.float32)

    loss = lsfn.call(y_true, y_pred)
    try:
      assert np.allclose(loss.numpy(),
                         true_loss_l + true_loss_c + true_loss_landm)
    except AssertionError:
      print(
          f'{loss.numpy()}!={true_loss_l + true_loss_c + true_loss_landm} {true_loss_l} {true_loss_c} {true_loss_landm}'
      )


def test_face_detector_1mb_anchor():
  """ 测试anchor和face_detector_1mb的anchor是否相同 """
  anchor_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
  steps = [8, 16, 32, 64]
  in_hw = [300, 300]
  feature_maps = [[
      np.ceil(in_hw[0] / step).astype(int),
      np.ceil(in_hw[1] / step).astype(int)
  ] for step in steps]

  anchors = []
  for k, f in enumerate(feature_maps):
    min_sizes = anchor_sizes[k]
    for i, j in product(range(f[0]), range(f[1])):
      for min_size in min_sizes:
        s_kx = min_size / in_hw[1]
        s_ky = min_size / in_hw[0]
        dense_cx = [x * steps[k] / in_hw[1] for x in [j + 0.5]]
        dense_cy = [y * steps[k] / in_hw[0] for y in [i + 0.5]]
        for cy, cx in product(dense_cy, dense_cx):
          anchors += [cx, cy, s_kx, s_ky]
  anchor = np.array(anchors).reshape(-1, 4)
  ture_anchor = np.load(
      '/home/zqh/workspace/Face-Detector-1MB-with-landmark/anchors.npy')

  ret_anchor = RetinaFaceHelper._get_anchors(in_hw, anchor_sizes, steps)

  assert np.allclose(anchor, ture_anchor)
  assert np.allclose(anchor, ret_anchor)


def test_face_detector_1mb_anchor_240_320():
  """ 测试240*320下的anchor feature map尺寸"""
  anchor_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
  steps = [8, 16, 32, 64]
  in_hw = [240, 320]
  feature_maps = [[
      np.ceil(in_hw[0] / step).astype(int),
      np.ceil(in_hw[1] / step).astype(int)
  ] for step in steps]

  anchors = []
  for k, f in enumerate(feature_maps):
    min_sizes = anchor_sizes[k]
    for i, j in product(range(f[0]), range(f[1])):
      for min_size in min_sizes:
        s_kx = min_size / in_hw[1]
        s_ky = min_size / in_hw[0]
        dense_cx = [x * steps[k] / in_hw[1] for x in [j + 0.5]]
        dense_cy = [y * steps[k] / in_hw[0] for y in [i + 0.5]]
        for cy, cx in product(dense_cy, dense_cx):
          anchors += [cx, cy, s_kx, s_ky]
  anchor = np.array(anchors).reshape(-1, 4)
  ture_anchor = np.load(
      '/home/zqh/workspace/Face-Detector-1MB-with-landmark/anchors.npy')

  ret_anchor = RetinaFaceHelper._get_anchors(in_hw, anchor_sizes, steps)

  assert np.allclose(anchor, ture_anchor)
  assert np.allclose(anchor, ret_anchor)


def test_retinafacenet_240_300_infer():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[10, 10], [16, 16], [24, 24]], [[32, 32], [48, 48]], [[64, 64], [96, 96]],
       [[128, 128], [192, 192], [256, 256]]], [8, 16, 32, 64], 0.35, [0.1, 0.2])
  input_shape = [240, 320, 3]
  infer_model, train_model = retinaface_slim(input_shape)
  infer_model.load_weights(
      'log/240_320_retinaface_slim_exp/infer_model_238.h5', by_name=True)

  batch_size = 1
  obj_thresh = 0.7
  nms_thresh = 0.4
  img_path, ann = np.copy(h.train_list[312])
  img = h.read_img(img_path)
  det_img, *ann = h.process_img(img, ann, h.in_hw, False, True, True)
  label_loc, label_landm, label_conf = h.ann_to_label(*ann, h.in_hw)
  re_label_landm = decode_landm(label_landm.numpy(), h.anchors.numpy(),
                                h.variances.numpy())
  re_label_bbox = decode_bbox(label_loc.numpy(), h.anchors.numpy(),
                              h.variances.numpy())

  re_bbox = tf.boolean_mask(re_label_bbox, tf.equal(label_conf, 1)[:, 0])
  re_landm = tf.boolean_mask(re_label_landm, tf.equal(label_conf, 1)[:, 0])
  # re_bbox =
  # re_landm = re_landm * tf.tile(tf.cast(h.in_hw[::-1], tf.float32), [5])

  outputs = infer_model.predict(det_img[None, ...])
  bbox_outs, landm_outs, class_outs = outputs
  bbox, landm, clses = bbox_outs[0], landm_outs[0], class_outs[0]
  """ softmax class"""
  clses = softmax(clses, -1)
  score = clses[:, 1]
  """ decode """
  bbox = decode_bbox(bbox, h.anchors.numpy(), h.variances.numpy())
  bbox = bbox * np.tile(h.org_in_hw[::-1], [2])
  """ landmark """
  landm = decode_landm(landm, h.anchors.numpy(), h.variances.numpy())
  landm = landm * np.tile(h.org_in_hw[::-1], [5])
  """ filter low score """
  inds = np.where(score > obj_thresh)[0]
  bbox = bbox[inds]
  landm = landm[inds]
  score = score[inds]
  """ keep top-k before NMS """
  order = np.argsort(score)[::-1]
  bbox = bbox[order]
  landm = landm[order]
  score = score[order]
  """ do nms """
  keep = nms_oneclass(bbox, score, nms_thresh)

  bbox = bbox[keep]
  landm = landm[keep]
  score = score[keep]
  """ reverse img """
  # bbox, landm = reverse_ann(bbox, landm, h.org_in_hw, np.array(orig_hw))
  # results.append([bbox, landm, score])
  h.draw_image(det_img.numpy(), [bbox, landm, np.ones_like(score)])


def test_retinafacenet_240_300_infer_ncc():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[13, 17], [28, 38]], [[74, 98], [119, 161]], [[171, 217], [234, 299]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  input_shape = [240, 320, 3]

  obj_thresh = 0.7
  nms_thresh = 0.4
  arr = np.fromfile('tmp/face_infer/face1.bin', np.float32)
  assert len(arr) == 3160 * 4 + 3160 * 10 + 3160 * 2
  bbox = arr[:3160 * 4].reshape((3160, -1))
  landm = arr[3160 * 4:3160 * 4 + 3160 * 10].reshape((3160, -1))
  clses = arr[3160 * 4 + 3160 * 10:].reshape((3160, -1))
  """ softmax class"""

  clses = softmax(clses, -1)
  score = clses[:, 1]
  """ decode """
  bbox = decode_bbox(bbox, h.anchors.numpy(), h.variances.numpy())
  bbox = bbox * np.tile(h.org_in_hw[::-1], [2])
  """ landmark """
  landm = decode_landm(landm, h.anchors.numpy(), h.variances.numpy())
  landm = landm * np.tile(h.org_in_hw[::-1], [5])
  """ filter low score """
  inds = np.where(score > obj_thresh)[0]
  bbox = bbox[inds]
  landm = landm[inds]
  score = score[inds]
  """ keep top-k before NMS """
  order = np.argsort(score)[::-1]
  bbox = bbox[order]
  landm = landm[order]
  score = score[order]
  """ do nms """
  keep = nms_oneclass(bbox, score, nms_thresh)

  bbox = bbox[keep]
  landm = landm[keep]
  score = score[keep]
  """ reverse img """
  # bbox, landm = reverse_ann(bbox, landm, h.org_in_hw, np.array(orig_hw))
  # results.append([bbox, landm, score])
  # h.draw_image(det_img.numpy(), [bbox, landm, np.ones_like(score)])
  print(np.hstack([bbox, score[:, None], landm]))


def test_retinafacenet():
  infer_model, train_model = retinafacenet(
      [640, 640, 3],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      64, 0.25)
  infer_model.load_weights(
      '/home/zqh/Documents/K210_Yolo_framework/log/640_640_retinaface_exp/auto_infer_138.h5',
      by_name=True)
  infer_model.summary()
  k.models.save_model(
      infer_model,
      '/home/zqh/Documents/K210_Yolo_framework/log/640_640_retinaface_exp/infer.h5'
  )


def test_retinafacenet_k210():
  """ v1 在回归时替换sparseconv为ssh模块，tflite=770KB ，kmdoel=244K 主内存=395kb """
  input_shape = [240, 320, 3]
  infer_model, train_model = retinafacenet_k210(input_shape)
  train_model.outputs
  train_model.summary()

  convert = tf.lite.TFLiteConverter.from_keras_model(train_model)
  tflitemodel = convert.convert()
  open("tflite/retinaface_k210_train.tflite", "wb").write(tflitemodel)
  print(Path('tflite/retinaface_k210_train.tflite').stat().st_size / 1024,
        'KB')  # 700KB
  # ! new_ncc compile tflite/retinaface_k210_train.tflite kmodel/retinaface_k210_train.kmodel -i tflite --dataset ~/workspace/retina_quan_img && du kmodel/retinaface_k210_train.kmodel -h


def test_retinafacenet_k210_v1():
  """ v1 在回归时替换sparseconv为ssh模块，tflite=1469.87KB ，kmdoel=412K 主内存=451.25kb """
  # input_shape = [240, 320, 3]
  # infer_model, train_model = retinafacenet_k210_v1(input_shape)

  # convert = tf.lite.TFLiteConverter.from_keras_model(train_model)
  # tflitemodel = convert.convert()
  # open("tflite/retinaface_k210_train_v1.tflite", "wb").write(tflitemodel)
  # print(Path('tflite/retinaface_k210_train_v1.tflite').stat().st_size / 1024, 'KB')

  # # ! new_ncc compile tflite/retinaface_k210_train_v1.tflite kmodel/retinaface_k210_train_v1.kmodel -i tflite --dataset ~/workspace/retina_quan_img && du kmodel/retinaface_k210_train_v1.kmodel -h
  input_shape = [240, 320, 3]
  infer_model, train_model = retinafacenet_k210_v1(
      input_shape,
      [[[13, 17], [28, 38]], [[74, 98], [119, 161]], [[171, 217], [234, 299]]])
  infer_model.load_weights('log/240_320_retinaface_k210_exp/auto_infer_358.h5')
  convert = tf.lite.TFLiteConverter.from_keras_model(infer_model)
  tflitemodel = convert.convert()
  open("tflite/retinaface_k210_v1_final.tflite", "wb").write(tflitemodel)
  print(
      Path('tflite/retinaface_k210_v1_final.tflite').stat().st_size / 1024, 'KB')

  # ! new_ncc compile tflite/retinaface_k210_v1_final.tflite kmodel/retinaface_k210_v1_final.kmodel -i tflite --dataset ~/workspace/retina_quan_img --input-std 1 --input-mean 0.5 && du kmodel/retinaface_k210_v1_final.kmodel -h
  # ! new_ncc infer kmodel/retinaface_k210_v1_final.kmodel tmp/face_infer --dataset faces --input-std 1 --input-mean 0.5
  # ! new_ncc infer kmodel/retinaface_k210_v1_final.kmodel tmp/face_quan_infer --dataset /home/zqh/workspace/retina_quan_img --input-std 1 --input-mean 0.5


def test_retinafacenet_k210_v2():
  """ v2 在原始的基础上添加了upsample模块，tflite=783 KB，kmdoel=248K 主内存=750kb """
  input_shape = [240, 320, 3]
  infer_model, train_model = retinafacenet_k210_v2(input_shape)

  convert = tf.lite.TFLiteConverter.from_keras_model(train_model)
  tflitemodel = convert.convert()
  open("tflite/retinaface_k210_train_v2.tflite", "wb").write(tflitemodel)
  print(
      Path('tflite/retinaface_k210_train_v2.tflite').stat().st_size / 1024, 'KB')

  # ! new_ncc compile tflite/retinaface_k210_train_v2.tflite kmodel/retinaface_k210_train_v2.kmodel -i tflite --dataset ~/workspace/retina_quan_img && du kmodel/retinaface_k210_train_v2.kmodel -h


def test_retinafacenet_k210_v3():
  """ v3 用了upsample模块以及ssh tflite=1649 KB，kmdoel=456K 主内存=825.0kb  """
  input_shape = [240, 320, 3]
  infer_model, train_model = retinafacenet_k210_v3(input_shape)
  train_model.outputs
  train_model.summary()

  convert = tf.lite.TFLiteConverter.from_keras_model(train_model)
  tflitemodel = convert.convert()
  open("tflite/retinaface_k210_train_v3.tflite", "wb").write(tflitemodel)
  print(
      Path('tflite/retinaface_k210_train_v3.tflite').stat().st_size / 1024,
      'KB')  # 1957.015625 kb

  # ! new_ncc compile tflite/retinaface_k210_train_v3.tflite kmodel/retinaface_k210_train_v3.kmodel -i tflite --dataset ~/workspace/retina_quan_img && du kmodel/retinaface_k210_train_v3.kmodel -h


def test_retinafacenet_slim():
  """ retinafacenet_slim模型 tflite=1340 KB，kmdoel=392K 主内存=552kb  """
  input_shape = [240, 320, 3]
  infer_model, train_model = retinaface_slim(input_shape)

  convert = tf.lite.TFLiteConverter.from_keras_model(train_model)
  tflitemodel = convert.convert()
  open("tflite/retinaface_slim.tflite", "wb").write(tflitemodel)
  print(Path('tflite/retinaface_slim.tflite').stat().st_size / 1024,
        'KB')  # 1340.0234375 kb
  # ! new_ncc compile tflite/retinaface_slim.tflite kmodel/retinaface_slim.kmodel -i tflite --dataset ~/workspace/retina_quan_img && du kmodel/retinaface_slim.kmodel -h


def test_find_anchor():
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [240, 320],
      [[[10, 10], [16, 16], [24, 24]], [[32, 32], [48, 48]], [[64, 64], [96, 96]],
       [[128, 128], [192, 192], [256, 256]]], [8, 16, 32, 64], 0.35, [0.1, 0.2])

  def f(img_path, ann):
    bbox = ann[:, :4]
    im = cv2.imread(img_path)
    scale = np.min(h.org_in_hw / im.shape[:2])
    return bbox * scale

  bbox = np.vstack([f(*ann) for ann in h.train_list])
  xywhbbox = np.hstack([(bbox[:, 2:] - bbox[:, :2]) / 2,
                        bbox[:, 2:] - bbox[:, :2]])
  xywhbbox = xywhbbox[np.min(xywhbbox[:, 2:], -1) > 10]

  initial_centroids = np.random.rand(6, 2) * np.array(h.org_in_hw[::-1])
  centroids, idx = runkMeans(xywhbbox[:, 2:], initial_centroids, 10, True)
  centroids = np.array(sorted(centroids, key=lambda x: (x[0])))
  # [[13 , 17],
  #  [28 , 38],
  #  [74 , 98],
  #  [119, 161],
  #  [171, 217],
  #  [234, 299]]


def test_generate_anchor_c():
  """ 把anchor转换为c文件 """
  # h = RetinaFaceHelper(
  #     'data/retinaface_img_ann.npy', [240, 320],
  #     [[[13, 17], [28, 38]], [[74, 98], [119, 161]], [[171, 217], [234, 299]]],
  #     [8, 16, 32], 0.35, [0.1, 0.2])
  h = RetinaFaceHelper(
      'data/retinaface_img_ann.npy', [640, 640],
      [[[16, 16], [32, 32]], [[64, 64], [128, 128]], [[256, 256], [512, 512]]],
      [8, 16, 32], 0.35, [0.1, 0.2])
  arr = np.ravel(h.anchors.numpy()).astype('str')
  print('Anchor num: ', len(arr) / 4)
  ss = ''
  for i, s in enumerate(arr):
    if i % 4 == 0:
      ss += '\n'
    ss += s + ','
  with open('tmp/prior.h', 'w') as f:
    f.write(
        f"#ifndef _PRIOR\n#define _PRIOR\n#include <stdint.h>\n#include <stdlib.h>\nfloat anchor[{len(arr)}]={'{'+ss+'}'};\n#endif"
    )


def test_generate_ssd_anchor_c():
  """ 将ssd的anchor转换为c """
  from tools.ssd import SSDHelper
  import numpy as np
  # arr = np.fromfile('tmp/ullfd_faces_res/face1.bin', np.float32)
  # len(arr)
  h = SSDHelper('data/wdface_voc_img_ann.npy', [240, 320], 1,
                [[[10, 10], [16, 16], [24, 24]], [[32, 32], [48, 48]],
                 [[64, 64], [96, 96]], [[128, 128], [192, 192], [256, 256]]],
                [8, 16, 32, 64], 0.35, [0.1, 0.2])
  arr = np.ravel(h.anchors.numpy()).astype('str')  # (4420, 4)
  ss = ''
  for i, s in enumerate(arr):
    if i % 4 == 0:
      ss += '\n'
    ss += s + ','
  with open('tmp/prior.h', 'w') as f:
    f.write(
        f"#ifndef _PRIOR\n#define _PRIOR\n#include <stdint.h>\n#include <stdlib.h>\nfloat anchor[{len(arr)}]={'{'+ss+'}'};\n#endif"
    )


def test_generate_ncc_infer_c():
  """ 把k210推理的结果转换为c """
  arr = np.fromfile('tmp/face_infer/face1.bin', np.float32).astype('str')
  assert len(arr) == 3160 * 4 + 3160 * 10 + 3160 * 2
  bbox = arr[:3160 * 4]
  landm = arr[3160 * 4:3160 * 4 + 3160 * 10]
  conf = arr[3160 * 4 + 3160 * 10:]

  with open('tmp/pred.h', 'w') as f:
    f.write(
        f"#ifndef _PRED\n#define _PRED\n#include <stdint.h>\n#include <stdlib.h>\n"
    )
    f.write(f"float pred_bbox[{len(bbox)}]={'{'+','.join(bbox)+'}'};\n")
    f.write(f"float pred_landm[{len(landm)}]={'{'+','.join(landm)+'}'};\n")
    f.write(f"float pred_conf[{len(conf)}]={'{'+','.join(conf)+'}'};\n")
    f.write("\n#endif")


def test_convert_to_pb():
  tf1 = tf.compat.v1
  tf1.disable_eager_execution()
  # tf.compat.v1.train.write_graph
  # tf.compat.v1.graph_util.convert_variables_to_constants
  # tf.compat.v1.lite.TFLiteConverter.from_frozen_graph
  from tensorflow.lite.python.util import set_tensor_shapes, is_frozen_graph, freeze_graph
  tf1.get_default_graph()
  tf1.keras.backend.clear_session()
  tf1.keras.backend.set_learning_phase(False)
  sess = tf1.keras.backend.get_session()
  init_graph = sess.graph
  input_shapes = None
  with init_graph.as_default():
    h5_model = tf1.keras.models.load_model('log/640_640_retinaface_exp/infer.h5')
    input_tensors = h5_model.inputs
    output_tensors = h5_model.outputs
    set_tensor_shapes(input_tensors, input_shapes)
    # graph_def = lite._freeze_graph(sess, input_tensors, output_tensors)
    graph_def = freeze_graph(sess, input_tensors, output_tensors)
    tf1.train.write_graph(
        graph_def,
        '/home/zqh/workspace/vim3l/aml_npu_sdk/acuity-toolkit/retinaface/data',
        'model.pb',
        as_text=False)
    in_nodes = [inp.op.name for inp in h5_model.inputs]
    out_nodes = [out.op.name for out in h5_model.outputs]
    print(in_nodes)
    print(out_nodes)
  tf.keras.backend.relu


def test_quantize_retinaface():
  import tensorflow_model_optimization as tfmot
  tfmot_sparsity = tfmot.sparsity.keras
  tfmot_quantization = tfmot.quantization.keras
  from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod

  with tfmot_quantization.quantize_scope():
    qmodel: k.Model = k.models.load_model(
        '/home/zqh/Downloads/log_retinaface_k210_quant_5/quantize_train_model_1479.h5')

  model: k.Model = k.models.load_model()
