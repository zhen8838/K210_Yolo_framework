""" yolo单元测试文件
"""
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.yolo import YOLOHelper, YOLOLoss, YOLOMap, parser_outputs
from tools.bbox_utils import corner_to_center, tf_bbox_iou
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.networks import yolo_mbv1
from typing import List

np.set_printoptions(suppress=True)

k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


def test_resize_img():
  """ 测试resize和darw NOTE 主要为了检验resize是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  i = 0
  for i in range(100, 120):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, new_ann = h.resize_img(img, h.in_hw, ann)
    h.draw_image(img.numpy(), new_ann.numpy())


def test_origin_resize_train_img():
  """ 测试原始yolov3 训练resize方法"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  in_hw = h.in_hw
  i = 214
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in range(100, 120):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, new_ann = h.origin_resize_train_img(img, in_hw, ann)
    h.draw_image(img.numpy(), new_ann.numpy())


def test_gluon_resize_train_img():
  """ 测试gluon版 yolov3 训练resize方法"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  in_hw = h.in_hw
  i = 214
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in range(100, 120):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, new_ann = h.gluon_resize_train_img(img, in_hw, ann)
    h.draw_image(img.numpy(), new_ann.numpy())


def test_origin_augment_img():
  """ 测试原版yolo augment"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  i = 213
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)

  for i in range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.origin_resize_train_img(img, h.in_hw, ann)
    img, ann = h.origin_augment_img(img, ann)
    h.draw_image(img.numpy(), ann.numpy())


def test_gluon_with_iaa_augment_img():
  """ 测试gluon与iaa augment"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  i = 213
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)

  for i in range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.gluon_resize_train_img(img, h.in_hw, ann)
    img, ann = tf.numpy_function(h.iaa_augment_img, [img, ann], [tf.uint8, tf.float32])
    h.draw_image(img.numpy(), ann.numpy())


def test_gluon_with_autoaugment_img():
  """ 测试gluon与autoaugment"""
  from transforms.image.auto_augment import distort_image_with_autoaugment
  from transforms.image.box_utils import yxyx_to_xyxy, normalize_boxes, denormalize_boxes
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  i = 213
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.gluon_resize_train_img(img, h.in_hw, ann)
    clas, xyxybox = tf.split(ann, [1, 4], 1)
    yxyxbox = yxyx_to_xyxy(xyxybox)
    img, yxyxbox = distort_image_with_autoaugment(img, yxyxbox, augmentation_name='v2')
    xyxybox = yxyx_to_xyxy(yxyxbox)
    ann = tf.concat([clas, xyxybox], -1)
    _ = h.draw_image(img.numpy(), ann.numpy())


def test_process_img():
  """ 测试处理图像流程,并绘制 NOTE 主要为了检验整个流程是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]],
                 resize_method='gluon',
                 augment_method='iaa')
  i = 213
  in_hw = h.in_hw
  is_resize = True
  is_augment = True
  is_normlize = False
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in tf.range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, new_ann = h.process_img(img, ann, in_hw, is_augment, is_resize, is_normlize)
    h.draw_image(img.numpy(), new_ann.numpy())


def test_process_widerface_img():
  """ 测试处理图像流程,并绘制 NOTE 主要为了检验整个流程是否正确"""
  h = YOLOHelper('data/wdface_voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  i = 213
  in_hw = h.in_hw
  is_resize = True
  is_augment = True
  is_normlize = False
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in tf.range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, new_ann = h.process_img(img, ann, in_hw, is_augment, is_resize, is_normlize)
    labels = tf.numpy_function(h.ann_to_label, [h.in_hw, h.out_hw, new_ann],
                               [tf.float32] * len(h.anchors))
    h.draw_image(img.numpy(), new_ann.numpy())


def test_multi_scale_process_img():
  """ 测试多尺度resize图像流程 NOTE 主要为了检验整个流程是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])

  for i, cur_scl in enumerate(range(-3, 3)):
    h.in_hw = h.org_in_hw + 32 * cur_scl
    h.out_hw = h.org_out_hw + np.power(2, np.arange(h.output_number))[:, None] * cur_scl
    for j in range(i * 10, (i + 1) * 10):
      path, ann, hw = h.train_list[j]
      img = h.read_img(path)
      img, ann = h.process_img(img, np.copy(ann), h.org_in_hw, True, True, False)
      h.draw_image(img, ann)


def test_label_to_ann_draw():
  """ 处理图像,并且从label转换为ann 并绘制 NOTE 主要为了检验整个处理以及label生成是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy', [224, 320], [[7, 10], [14, 20]],
                 resize_method='gluon',
                 augment_method='iaa')

  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in tf.range(120, 140):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.process_img(img, ann, h.in_hw, True, True, False)
    labels = tf.numpy_function(h.ann_to_label, [h.in_hw, h.out_hw, ann],
                               [tf.float32] * len(h.anchors))
    labels = [l.numpy() for l in labels]
    new_ann = h.label_to_ann(labels)
    h.draw_image(img.numpy(), new_ann)


def test_label_to_ann_compare():
  """ 处理图像,并且从label转换为ann 并对比 NOTE 主要为了检验整个处理以及label生成是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy', [224, 320], [[7, 10], [14, 20]])
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)
  for i in tf.range(156, 200):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.process_img(img, ann, h.in_hw, True, True, False)
    labels = tf.numpy_function(h.ann_to_label, [h.in_hw, h.out_hw, ann], [
                               tf.float32] * len(h.anchors))
    labels = [l.numpy() for l in labels]
    ann = ann.numpy()
    new_ann = h.label_to_ann(labels)
    try:
      ann = np.array(sorted(ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
      new_ann = np.array(sorted(new_ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
      assert np.allclose(ann, new_ann, 1e-04)
    except AssertionError:
      print(ann)
      print(new_ann)

    # h.draw_image(img.numpy(), new_ann)


def test_label_to_ann_v3_compare():
  """ 处理图像,并且从label转换为ann 并对比 NOTE 主要为了检验整个处理以及label生成是否正确"""
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor_v3.npy',
                 [416, 416], [[13, 13], [26, 26], [52, 52]])
  i = 213
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)

  for i in tf.range(156, 200):
    img_str, img_name, ann, hw = next(iters)
    img = h.decode_img(img_str)
    img, ann = h.process_img(img, ann, h.in_hw, True, True, False)
    labels = tf.numpy_function(h.ann_to_label, [h.in_hw, h.out_hw, ann],
                               [tf.float32] * len(h.anchors))
    labels = [l.numpy() for l in labels]
    new_ann = h.label_to_ann(labels)
    ann = ann.numpy()
    try:
      ann = np.array(sorted(ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
      new_ann = np.array(sorted(new_ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
      assert np.allclose(ann, new_ann, 1e-04)
    except AssertionError:
      print(ann)
      print(new_ann)

    # h.draw_image(img, new_ann)


def test_train_dataset_label_to_ann():
  """ 测试训练集dataset加载 """
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  h.set_dataset(1, True, False, True)
  i = 213
  iters = iter(h.train_dataset)
  for i in range(20):
    img, labels = next(iters)
    labels = [label[0].numpy() for label in labels]
    new_ann = h.label_to_ann(labels)
    h.draw_image(img[0].numpy(), new_ann)


def test_val_dataset_label_to_ann():
  """ 测试验证集dataset加载 """
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [224, 320], [[7, 10], [14, 20]])
  h.set_dataset(1, False, False, True)
  i = 213
  iters = iter(h.train_dataset)
  for i in range(20):
    img, labels = next(iters)
    labels = [label[0].numpy() for label in labels]
    new_ann = h.label_to_ann(labels)
    h.draw_image(img[0].numpy(), new_ann)


def test_yolo_loss_compare():
  """ 测试loss计算差异性～ """
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                 [608, 608], [[19, 19], [38, 38]])
  h.batch_size = 16

  """ 可以测试1 或 2 都是相同的~ 应该是没有错误了 """
  # with np.load('tmp/out.npz', allow_pickle=True) as npz:
  npz = np.load(
      '/home/zqh/Documents/ObjectDetection-OneStageDet/yolo/tmp/loss_detail.npy', allow_pickle=True)[()]
  outputs = npz['y_pred']
  target = npz['ann']
  loss_coord = npz['loss_coord']
  loss_conf = npz['loss_conf']
  loss_cls = npz['loss_cls']
  loss_tot = npz['loss_tot']
  anchor = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
  anchors_mask = np.array([(3, 4, 5), (0, 1, 2)])
  anchor = anchor[anchors_mask]
  h.anchors = anchor / [608, 608]
  h._YOLOHelper__flatten_anchors = np.reshape(h.anchors, (-1, 2))

  fn0 = YOLOLoss(h, 0.5, 0.7, 1, 1, 3, 2, 1, 0)
  fn1 = YOLOLoss(h, 0.5, 0.7, 1, 1, 3, 2, 1, 1)

  y_trues = [[], []]
  for i in range(len(target)):
    labels = h.ann_to_label(h.org_in_hw, h.org_out_hw, target[i])
    y_trues[0].append(labels[0])
    y_trues[1].append(labels[1])
  y_trues[0] = np.stack(y_trues[0])
  y_trues[1] = np.stack(y_trues[1])

  y_pred0 = np.reshape(np.transpose(outputs[0], [0, 2, 3, 1]), (16, 19, 19, 3, 25))
  y_pred1 = np.reshape(np.transpose(outputs[1], [0, 2, 3, 1]), (16, 38, 38, 3, 25))
  for self, y_true, y_pred in zip([fn0, fn1], y_trues, [y_pred0, y_pred1]):
    y_true, y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
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
        obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

    noobj_loss = self.noobj_weight * tf.reduce_sum(
        (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

    cls_loss = tf.reduce_sum(
        obj_mask * self.cls_weight * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_cls, logits=pred_cls), [1, 2, 3, 4])

    softmax_cls_loss = tf.reduce_sum(
        obj_mask[..., 0] * self.cls_weight * tf.nn.softmax_cross_entropy_with_logits(
            labels=true_cls, logits=pred_cls), [1, 2, 3])

    """ sum loss """
    obj_loss = tf.reduce_sum(obj_loss)
    noobj_loss = tf.reduce_sum(noobj_loss)
    cls_loss = tf.reduce_sum(cls_loss)
    softmax_cls_loss = tf.reduce_sum(softmax_cls_loss)
    xy_loss = tf.reduce_sum(xy_loss)
    wh_loss = tf.reduce_sum(wh_loss)

    assert np.allclose(loss_coord[self.layer], (xy_loss + wh_loss).numpy())
    assert np.allclose(loss_conf[self.layer], (obj_loss + noobj_loss).numpy())
    assert np.allclose(loss_cls[self.layer], softmax_cls_loss.numpy())


def test_yolo_loss_clac():
  """ 用于测试yolo loss中的计算细节
  """
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor_v3.npy',
                 [416, 416], [[13, 13], [26, 26], [52, 52]])
  # yloss = YOLOLoss(h, , ,
  _, net = yolo_mbv1([416, 416, 3], 3, 20, 1.0)
  # net.load_weights('log/20191024-214827/train_model_19.h5')
  idx = 12

  imgs = []
  y_true = []
  ds = tf.data.TFRecordDataset(h.train_list).map(h.parser_example)
  iters = iter(ds)

  for idx in range(12, 12 + 16):
    img_str, img_name, ann, hw = next(iters)
    raw_img = h.decode_img(img_str)
    # resize image -> image augmenter
    raw_img, ann = h.process_img(raw_img, ann, h.org_in_hw, False, True, False)
    # make labels
    print(ann)
    labels = h.ann_to_label(h.org_in_hw, h.org_out_hw, ann)
    # normlize image
    img = h.normlize_img(raw_img)
    imgs.append(img)
    y_true.append(labels[0])

  preds = net(tf.stack(imgs))
  # preds.shape  # TensorShape([16, 7, 10, 5, 25])
  obj_thresh = 0.7
  iou_thresh = 0.3
  obj_weight = 1
  noobj_weight = 1
  wh_weight = 1
  layer = 0
  y_true = np.stack(y_true)
  y_true.shape  # (16, 13, 13, 3, 26)
  y_pred = preds[0]  # (16, 13, 13, 3, 26)

  """ split the label """
  out_hw = tf.cast(tf.shape(y_true)[1:3], tf.float32)
  y_true = tf.reshape(y_true, [-1, out_hw[0], out_hw[1],
                               h.anchor_number, h.class_num + 5 + 1])
  y_pred = tf.reshape(y_pred, [-1, out_hw[0], out_hw[1],
                               h.anchor_number, h.class_num + 5])

  grid_pred_xy = y_pred[..., 0:2]
  grid_pred_wh = y_pred[..., 2:4]
  pred_confidence = y_pred[..., 4:5]
  pred_cls = y_pred[..., 5:]

  all_true_xy = y_true[..., 0:2]
  all_true_wh = y_true[..., 2:4]
  true_confidence = y_true[..., 4:5]
  true_cls = y_true[..., 5:5 + h.class_num]
  location_mask = tf.cast(y_true[..., -1], tf.bool)

  obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh
  obj_mask.shape
  obj_mask_bool = tf.cast(y_true[..., 4], tf.bool)

  """ calc the ignore mask  """
  pred_xy, pred_wh = YOLOLoss.xywh_to_all(grid_pred_xy, grid_pred_wh,
                                          out_hw, h.xy_offsets[layer], h.anchors[layer])

  # NOTE 添加 recall
  tp50 = tf.compat.v1.get_variable('tp50', (), tf.float32, tf.zeros_initializer())
  tp75 = tf.compat.v1.get_variable('tp75', (), tf.float32, tf.zeros_initializer())

  def lmba(bc):
    # bc=9

    # ! 使用location_mask找到全局的gt
    gt_xy = tf.boolean_mask(all_true_xy[bc], location_mask[bc])
    gt_wh = tf.boolean_mask(all_true_wh[bc], location_mask[bc])
    # iou score = [h,w,anchor,box_num]
    iou_score = YOLOLoss.iou(pred_xy[bc], pred_wh[bc], gt_xy, gt_wh)

    # NOTE 利用boolmask得到index， 用loc_iou_score计算的这一层的tp
    idx = tf.where(tf.boolean_mask(obj_mask_bool[bc], location_mask[bc]))
    mask_iou_score = tf.gather_nd(tf.boolean_mask(iou_score, obj_mask_bool[bc]), idx, 1)
    tp50.assign_add(tf.reduce_sum(tf.cast(mask_iou_score > 0.5, tf.float32)))
    tp75.assign_add(tf.reduce_sum(tf.cast(mask_iou_score > 0.75, tf.float32)))

    # 对于这一层的gt的iou score
    layer_iou_score = tf.squeeze(tf.gather(iou_score, idx, axis=-1), -1)
    layer_match50 = tf.reduce_sum(tf.cast(layer_iou_score > 0.5, tf.float32),
                                  -1, keepdims=True)
    layer_match75 = tf.reduce_sum(tf.cast(layer_iou_score > 0.75, tf.float32),
                                  -1, keepdims=True)

    # ! 修改为与任意一个gt的iou大于iou thresh即为预测正确。
    match_num = tf.reduce_sum(tf.cast(iou_score > iou_thresh, tf.float32),
                              -1, keepdims=True)
    return (tf.cast(tf.less(match_num, 1), tf.float32),
            tf.cast(tf.less(layer_match50, 1), tf.float32),
            tf.cast(tf.less(layer_match75, 1), tf.float32))

  # ignore mask 表示的是没有正确预测到物体的位置
  ignore_mask, layer_ignore_mask50, layer_ignore_mask75 = tf.map_fn(
      lmba, tf.range(16), dtype=(tf.float32, tf.float32, tf.float32))

  # ! FP=将负样本预测为正样本， 原本位置要为负样本且预测结果为正样本（预测置信度大于阈值且没有预测到任何一个有效目标）
  fp50 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) > obj_thresh,
                                tf.float32) * layer_ignore_mask50) * (1 - obj_mask))
  fp75 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) > obj_thresh,
                                tf.float32) * layer_ignore_mask75) * (1 - obj_mask))
  # ! FN=将正样本预测为负样本， 原本位置要为正样本且预测结果为负样本（预测置信度小于阈值或者没有预测到任何一个有效目标）
  fn50 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) < obj_thresh,
                                tf.float32) + layer_ignore_mask50) * obj_mask)
  fn75 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) < obj_thresh,
                                tf.float32) + layer_ignore_mask75) * obj_mask)

  precision50 = tp50 / (tp50 + fp50)
  precision75 = tp75 / (tp75 + fp75)

  recall50 = tp50 / (tp50 + fn50)
  recall75 = tp75 / (tp75 + fn75)

  grid_true_xy, grid_true_wh = YOLOLoss.xywh_to_grid(
      all_true_xy, all_true_wh, out_hw, h.xy_offsets[layer], h.anchors[layer])
  # NOTE When wh=0 , tf.log(0) = -inf, so use tf.where to avoid it
  grid_true_wh = tf.where(tf.tile(obj_mask_bool[..., tf.newaxis], [1, 1, 1, 1, 2]),
                          grid_true_wh, tf.zeros_like(grid_true_wh))

  """ define loss """
  coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

  xy_loss = tf.reduce_sum(
      obj_mask * coord_weight * tf.nn.sigmoid_cross_entropy_with_logits(
          labels=grid_true_xy, logits=grid_pred_xy), [1, 2, 3, 4])

  wh_loss = tf.reduce_sum(
      obj_mask * coord_weight * wh_weight * tf.square(tf.subtract(
          x=grid_true_wh, y=grid_pred_wh)), [1, 2, 3, 4])

  obj_loss = obj_weight * tf.reduce_sum(
      obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
          labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

  noobj_loss = noobj_weight * tf.reduce_sum(
      (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
          labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

  cls_loss = tf.reduce_sum(
      obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
          labels=true_cls, logits=pred_cls), [1, 2, 3, 4])

  total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss


def check_loss_is_nan():
  """ 检查loss出现nan的原因 """
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor_v3.npy',
                 [416, 416], [[13, 13], [26, 26], [52, 52]])
  h.set_dataset(16, True, True, True)
  _, net = yolo_mbv1([416, 416, 3], 3, 20, 1.0)

  fns: List[YOLOLoss] = [YOLOLoss(h, 0.5, 0.7, 1, 1, 3, 2, 1, 0, verbose=0),
                         YOLOLoss(h, 0.5, 0.7, 1, 1, 3, 2, 1, 1, verbose=0),
                         YOLOLoss(h, 0.5, 0.7, 1, 1, 3, 2, 1, 2, verbose=0)]

  def find_nan():
    for (img, y_trues) in h.train_dataset:
      y_preds = net(img)
      for l, fn in enumerate(fns):
        loss = fn(y_trues[l], y_preds[l])
        print('loss :', loss.numpy(), end='\r')
        if tf.math.is_nan(loss):
          return y_trues, y_preds

  y_trues, y_preds = find_nan()
  y_trues = [y_true.numpy() for y_true in y_trues]
  y_preds = [y_pred.numpy() for y_pred in y_preds]
  np.save('tmp/yolo_nan_debug.npz',
          {'y_trues': y_trues, 'y_preds': y_preds})


def test_sigmoid_loss_nan():
  """ 发现obj loss出现nan了，看看sigmoid_cross_entropy_with_logits什么时候会出现nan """
  tf.nn.sigmoid_cross_entropy_with_logits(7., np.NaN)
  tf.nn.sigmoid_cross_entropy_with_logits(np.NaN, 7.)
  infer_model, train_model = yolo_mbv1((416, 416, 3), 3, 20, 1.0)
  train_model.outputs
  infer_model.outputs


def test_bbox_iou():
  boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]]) / 20.
  boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]]) / 20.
  """ 按元素求iou """
  print(tf_bbox_iou(boxes1, boxes2[..., None, :], method='iou').numpy())
  # [[0.12500001] [0.        ]]
  print(tf_bbox_iou(boxes1, boxes2[..., None, :], method='diou').numpy())
  # [[ 0.00304878] [-0.6243095 ]]
  print(tf_bbox_iou(boxes1, boxes2[..., None, :], method='ciou').numpy())
  # [[ 0.00283995] [-0.6250417 ]]
  print(tf_bbox_iou(boxes1, boxes2[..., None, :], method='giou').numpy())
  # [[-0.07499996] [-0.93333334]]

  """ 交叉求iou """
  print(tf_bbox_iou(boxes1, boxes2, method='iou').numpy())
  # [[0.12500001 0.        ] [0.0625     0.        ]]
  print(tf_bbox_iou(boxes1, boxes2, method='diou').numpy())
  # [[ 0.00304878 -0.72169816] [-0.07980768 -0.6243095 ]]
  print(tf_bbox_iou(boxes1, boxes2, method='ciou').numpy())
  # [[ 0.00283995 -0.7217355 ] [-0.08119209 -0.6250417 ]]
  print(tf_bbox_iou(boxes1, boxes2, method='giou').numpy())
  # [[-0.07499996 -0.9469697 ] [-0.3660715  -0.93333334]]

  boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]]) / 20.
  print(tf_bbox_iou(boxes1[0], boxes1[0], method='iou').numpy())


def test_grad():
  offset = 0.
  a = tf.Variable([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
  b = tf.Variable([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
  with tf.GradientTape(True) as tape:
    a = a[..., None, :]
    b = b[..., None, :]
    tl = tf.maximum(a[..., :2], b[..., :2])
    br = tf.minimum(a[..., 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    iou = area_i / (area_a + area_b - area_i)

    outer_tl = tf.minimum(a[..., :2], b[..., :2])
    outer_br = tf.maximum(a[..., 2:4], b[..., 2:4])
    # two bbox center distance sum((b_cent-a_cent)^2)
    inter_diag = tf.reduce_sum(tf.square((b[..., :2] + b[..., 2:]) / 2
                                         - (a[..., :2] + a[..., 2:]) / 2 + offset), -1)
    # two bbox diagonal distance
    outer_diag = tf.reduce_sum(tf.square(outer_tl - outer_br + offset), -1)
    # calc ciou alpha paramter

    arctan = (tf.math.atan(tf.math.divide_no_nan(b[..., 2] - b[..., 0],
                                                 b[..., 3] - b[..., 1]))
              - tf.math.atan(tf.math.divide_no_nan(a[..., 2] - a[..., 0],
                                                   a[..., 3] - a[..., 1])))

    v = tf.math.square(2 / np.pi) * tf.square(arctan)
    alpha = v / ((1 - iou) + v)
    w_temp = 2 * (a[..., 2] - a[..., 0])

    ar = (8 / tf.square(np.pi)) * arctan * \
        ((a[..., 2] - a[..., 0] - w_temp) * (a[..., 3] - a[..., 1]))

    ciou = iou - (inter_diag / outer_diag) - (alpha * ar)

  tape.gradient(v, [a, b])
  """ [[[-0.04231081,  0.06346621,  0.04231081, -0.06346621]],
        [[-0.01833142,  0.09165711,  0.01833142, -0.09165711]]]

        [[[ 0.04400324, -0.03300243, -0.04400324,  0.03300243]],
        [[ 0.23830847, -0.23830847, -0.23830847,  0.23830847]]]"""
  tape.gradient(ciou, [a, b])
  """
        [[[ 0.06351729, -0.08257562, -0.15244228,  0.27827212]],
        [[ 0.08103281,  0.01117781, -0.07266921,  0.01513398]]],

        [[[-0.0851735 , -0.06970734,  0.17409849, -0.12598914]],
        [[-0.7246207 ,  0.6417478 ,  0.7162571 , -0.6680595 ]]]
    """

  offset = 0.
  a = tf.Variable([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
  b = tf.Variable([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
  with tf.GradientTape(True) as tape:
    a = a[..., None, :]
    b = b[..., None, :]
    tl = tf.maximum(a[..., :2], b[..., :2])
    br = tf.minimum(a[..., 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    iou = area_i / (area_a + area_b - area_i)

    outer_tl = tf.minimum(a[..., :2], b[..., :2])
    outer_br = tf.maximum(a[..., 2:4], b[..., 2:4])
    # two bbox center distance sum((b_cent-a_cent)^2)
    inter_diag = tf.reduce_sum(tf.square((b[..., :2] + b[..., 2:]) / 2
                                         - (a[..., :2] + a[..., 2:]) / 2 + offset), -1)
    # two bbox diagonal distance
    outer_diag = tf.reduce_sum(tf.square(outer_tl - outer_br + offset), -1)
    # calc ciou alpha paramter

    arctan = tf.stop_gradient(
        (tf.math.atan(tf.math.divide_no_nan(b[..., 2] - b[..., 0],
                                            b[..., 3] - b[..., 1]))
         - tf.math.atan(tf.math.divide_no_nan(a[..., 2] - a[..., 0],
                                              a[..., 3] - a[..., 1]))))

    v = tf.stop_gradient(tf.math.square(2 / np.pi) * tf.square(arctan))
    alpha = tf.stop_gradient(v / ((1 - iou) + v))
    w_temp = tf.stop_gradient(2 * (a[..., 2] - a[..., 0]))
    ar = (8 / tf.square(np.pi)) * arctan * \
        ((a[..., 2] - a[..., 0] - w_temp) * (a[..., 3] - a[..., 1]))

    ciou = iou - (inter_diag / outer_diag) - (alpha * ar)

  tape.gradient(ar, [a, b])
  """
        [[[ 0.5500405 , -0.8250607 , -0.5500405 ,  0.8250607 ]],
        [[ 0.47661695, -2.3830848 , -0.47661695,  2.3830848 ]]],

        None
     """
  tape.gradient(v, [a, b])
  """ [None, None] """
  tape.gradient(ciou, [a, b])

  """
        [[[-0.10692195,  0.08424009,  0.01162432,  0.12420168]],
        [[-0.08888854,  0.27500343,  0.09725215, -0.24869165]]]

        [[[ 0.03184488, -0.16596799,  0.06345274, -0.04247379]],
        [[-0.03867403, -0.0441989 ,  0.03031043,  0.01788712]]]
    """


def test_yolo_map():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor_v3.npy',
                 [416, 416], [[13, 13], [26, 26], [52, 52]])
  infer_model, train_model = yolo_mbv1([416, 416, 3], h.anchor_number, h.class_num, 1.0)
  infer_model.load_weights('log/default_yolov3_mbv1_ciou/infer_model_205.h5')
  h.set_dataset(32, False, True, False)
  name = ['aeroplane',
          'bicycle',
          'bird',
          'boat',
          'bottle',
          'bus',
          'car',
          'cat',
          'chair',
          'cow',
          'diningtable',
          'dog',
          'horse',
          'motorbike',
          'person',
          'pottedplant',
          'sheep',
          'sofa',
          'train',
          'tvmonitor', ]
  maps = YOLOMap(h, h.test_dataset.take(10), tf.Variable(0., False), 1, name, 0.1, 0.3, 'iou', 0.5)
  maps.set_model(infer_model)
  self = maps
  maps.on_train_end()

  # maps.on_epoch_begin(0)
