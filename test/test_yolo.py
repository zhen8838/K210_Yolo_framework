""" yolo单元测试文件
"""
from tools.yolo import YOLOHelper, YOLOLoss
from tools.bbox_utils import corner_to_center
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.networks import yolo_mbv1
np.set_printoptions(suppress=True)

k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


def test_resize_img():
    """ 测试resize和darw NOTE 主要为了检验resize是否正确"""
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                   [224, 320], [[7, 10], [14, 20]])
    i = 0
    for i in range(100, 120):
        path, ann, hw = h.train_list[i]
        img = h.read_img(path)
        img, ann = h.resize_img(img, h.org_in_hw, np.copy(ann))
        h.draw_image(img, ann)


def test_augment_img():
    """ 测试augment和darw NOTE 主要为了检验augment是否正确"""
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                   [224, 320], [[7, 10], [14, 20]])
    i = 213
    for i in range(120, 140):
        path, ann, hw = h.train_list[i]
        img = h.read_img(path)
        img, ann = h.resize_img(img, h.org_in_hw, np.copy(ann))
        img, ann = h.augment_img(img, ann)
        h.draw_image(img, ann)


def test_process_img():
    """ 测试处理图像流程,并绘制 NOTE 主要为了检验整个流程是否正确"""
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy',
                   [224, 320], [[7, 10], [14, 20]])
    i = 213
    for i in range(120, 140):
        path, ann, hw = h.train_list[i]
        img = h.read_img(path)
        img, ann = h.process_img(img, ann, h.org_in_hw, is_augment=True, is_resize=True, is_normlize=False)
        h.draw_image(img, ann)


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
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy', [224, 320], [[7, 10], [14, 20]])
    i = 213
    for i in range(156, 200):
        path, ann, hw = h.train_list[i][0], np.copy(h.train_list[i][1]), h.train_list[i][2]
        img = h.read_img(path)
        img, ann = h.process_img(img, ann, h.org_in_hw, True, True, False)
        labels = h.ann_to_label(h.org_in_hw, h.org_out_hw, np.copy(ann))
        new_ann = h.label_to_ann(labels)
        h.draw_image(img, new_ann)


def test_label_to_ann_compare():
    """ 处理图像,并且从label转换为ann 并对比 NOTE 主要为了检验整个处理以及label生成是否正确"""
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy', [224, 320], [[7, 10], [14, 20]])
    i = 213
    for i in range(156, 200):
        path, ann, hw = h.train_list[i][0], np.copy(h.train_list[i][1]), h.train_list[i][2]
        img = h.read_img(path)
        img, ann = h.process_img(img, ann, h.org_in_hw, True, True, False)
        labels = h.ann_to_label(h.org_in_hw, h.org_out_hw, np.copy(ann))
        new_ann = h.label_to_ann(labels)
        try:
            ann = np.array(sorted(ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
            new_ann = np.array(sorted(new_ann, key=lambda x: (-x[0], -x[1], -x[3], -x[4])))
            assert np.allclose(ann, new_ann)
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
    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor.npy', [608, 608], [[19, 19], [38, 38]])
    h.batch_size = 16

    """ 可以测试1 或 2 都是相同的~ 应该是没有错误了 """
    # with np.load('tmp/out.npz', allow_pickle=True) as npz:
    with np.load('tmp/out1.npz', allow_pickle=True) as npz:
        outputs = [npz['out0'], npz['out1']]
        target = npz['target']
        anchor = npz['anchor']
        losses = npz['loss']

    anchor = anchor / 608
    anchor = np.array([anchor[[3, 4, 5]],
                       anchor[[0, 1, 2]]])
    h.anchors = anchor
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

    y_true0 = y_trues[0]
    y_pred0 = np.reshape(np.transpose(outputs[0], [0, 2, 3, 1]), (16, 19, 19, 3, 25))
    loss0 = fn0.call(tf.convert_to_tensor(y_true0), tf.convert_to_tensor(y_pred0))
    loss0 = tf.reduce_sum(loss0)

    y_true1 = y_trues[1]
    y_pred1 = np.reshape(np.transpose(outputs[1], [0, 2, 3, 1]), (16, 38, 38, 3, 25))
    loss1 = fn1.call(tf.convert_to_tensor(y_true1), tf.convert_to_tensor(y_pred1))
    loss1 = tf.reduce_sum(loss1)

    assert np.allclose(losses[0], loss0.numpy())  # 231.03738
    assert np.allclose(losses[1], loss1.numpy())  # 66.90617


test_yolo_loss_compare()


def test_yolo_loss_clac():
    """ 用于测试yolo loss中的计算细节
    """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    h = YOLOHelper('data/voc_img_ann.npy', 20, 'data/voc_anchor_v3.npy', [416, 416], [[13, 13], [26, 26], [52, 52]])
    # yloss = YOLOLoss(h, , ,
    _, net = yolo_mbv1([416, 416, 3], 3, 20, 1.0)
    # net.load_weights('log/20191024-214827/train_model_19.h5')
    idx = 12

    imgs = []
    y_true = []
    for idx in range(12, 12 + 16):
        img_path, ann = h.test_list[idx][0].copy(), h.test_list[idx][1].copy()
        # load image
        raw_img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
        # resize image -> image augmenter
        raw_img, ann = h.process_img(raw_img.numpy(), ann, h.org_in_hw, False, True, False)
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
    xy_offset = YOLOLoss.calc_xy_offset(out_hw, y_pred)

    pred_xy, pred_wh = YOLOLoss.xywh_to_all(grid_pred_xy, grid_pred_wh,
                                            out_hw, xy_offset, h.anchors[layer])

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
    ignore_mask, layer_ignore_mask50, layer_ignore_mask75 = tf.map_fn(lmba, tf.range(16), dtype=(tf.float32, tf.float32, tf.float32))

    # ! FP=将负样本预测为正样本， 原本位置要为负样本且预测结果为正样本（预测置信度大于阈值且没有预测到任何一个有效目标）
    fp50 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) > obj_thresh, tf.float32) * layer_ignore_mask50) * (1 - obj_mask))
    fp75 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) > obj_thresh, tf.float32) * layer_ignore_mask75) * (1 - obj_mask))
    # ! FN=将正样本预测为负样本， 原本位置要为正样本且预测结果为负样本（预测置信度小于阈值或者没有预测到任何一个有效目标）
    fn50 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) < obj_thresh, tf.float32) + layer_ignore_mask50) * obj_mask)
    fn75 = tf.reduce_sum((tf.cast(tf.sigmoid(pred_confidence) < obj_thresh, tf.float32) + layer_ignore_mask75) * obj_mask)

    precision50 = tp50 / (tp50 + fp50)
    precision75 = tp75 / (tp75 + fp75)

    recall50 = tp50 / (tp50 + fn50)
    recall75 = tp75 / (tp75 + fn75)

    grid_true_xy, grid_true_wh = YOLOLoss.xywh_to_grid(all_true_xy, all_true_wh, layer, h)
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
