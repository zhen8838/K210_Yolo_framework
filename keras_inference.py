import tensorflow.python as tf
from tensorflow.python import keras
from pathlib import Path
from tools.utils import Helper, INFO, ERROR, NOTE, tf_xywh_to_all
from models.yolonet import *
from termcolor import colored
import argparse
import sys

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
keras.backend.set_learning_phase(0)


def tf_center_to_corner(box: tf.Tensor) -> tf.Tensor:
    """convert x,y,w,h box to y,x,y,x box"""
    x1 = (box[..., 0:1] - box[..., 2:3] / 2)
    y1 = (box[..., 1:2] - box[..., 3:4] / 2)
    x2 = (box[..., 0:1] + box[..., 2:3] / 2)
    y2 = (box[..., 1:2] + box[..., 3:4] / 2)
    box = tf.concat([y1, x1, y2, x2], -1)
    return box


def tf_corner_to_center(yxyx_box: tf.Tensor) -> tf.Tensor:
    """convert y,x,y,x box to x,y,w,h box"""
    x = (yxyx_box[..., 3:4] - yxyx_box[..., 1:2]) / 2 + yxyx_box[..., 1:2]
    y = (yxyx_box[..., 2:3] - yxyx_box[..., 0:1]) / 2 + yxyx_box[..., 0:1]
    w = yxyx_box[..., 3:4] - yxyx_box[..., 1:2]
    h = yxyx_box[..., 2:3] - yxyx_box[..., 0:1]
    box = tf.concat([x, y, w, h], axis=-1)
    return box


# obj_thresh = 0.7
# iou_thresh = 0.3
# ckpt_weights = 'log/20190709-192922/yolo_model.h5'
# image_size = [224, 320]
# output_size = [7, 10, 14, 20]
# model_def = 'yolo_mobilev2'
# class_num = 20
# depth_multiplier = 0.75
# train_set = 'voc'
# test_image = 'tmp/bb.jpg'


def main(ckpt_weights, image_size, output_size, model_def, class_num, depth_multiplier, obj_thresh, iou_thresh, train_set, test_image):
    h = Helper(None, class_num, f'data/{train_set}_anchor.npy',
               [[image_size[0], image_size[1]]], [[output_size[0], output_size[1]],
                                                  [output_size[2], output_size[3]]])
    network = eval(model_def)  # type :yolo_mobilev2
    yolo_model, yolo_model_warpper = network([image_size[0], image_size[1], 3], len(h.anchors[0]), class_num, alpha=depth_multiplier)

    yolo_model_warpper.load_weights(str(ckpt_weights))
    print(INFO, f' Load CKPT {str(ckpt_weights)}')

    img, _ = h._process_img(h._read_img(str(test_image)), true_box=None, is_training=False, is_resize=True)

    """ load images """
    img = tf.expand_dims(img, 0)
    y_pred = yolo_model_warpper.predict(img)

    """ box list """
    _yxyx_box = []
    _yxyx_box_scores = []
    """ preprocess label """
    for l, pred_label in enumerate(y_pred):
        """ split the label """
        pred_xy = pred_label[..., 0:2]
        pred_wh = pred_label[..., 2:4]
        pred_confidence = pred_label[..., 4:5]
        pred_cls = pred_label[..., 5:]
        # box_scores = obj_score * class_score
        box_scores = tf.sigmoid(pred_cls) * tf.sigmoid(pred_confidence)
        # obj_mask = pred_confidence_score[..., 0] > obj_thresh
        """ reshape box  """
        # NOTE tf_xywh_to_all will auto use sigmoid function
        pred_xy_A, pred_wh_A = tf_xywh_to_all(pred_xy, pred_wh, l, h)
        xywh_box = tf.reshape(tf.concat([pred_xy_A, pred_wh_A], -1), (-1, 4))
        box_scores = tf.reshape(box_scores, (-1, class_num))
        """ append box and scores to global list """
        _yxyx_box.append(tf_center_to_corner(xywh_box))
        _yxyx_box_scores.append(box_scores)

    yxyx_box = tf.concat(_yxyx_box, axis=0)
    yxyx_box_scores = tf.concat(_yxyx_box_scores, axis=0)

    mask = yxyx_box_scores >= obj_thresh

    """ do nms for every classes"""
    _boxes = []
    _scores = []
    _classes = []
    for c in range(class_num):
        class_boxes = tf.boolean_mask(yxyx_box, mask[:, c])
        class_box_scores = tf.boolean_mask(yxyx_box_scores[:, c], mask[:, c])
        select = tf.image.non_max_suppression(
            class_boxes, scores=class_box_scores, max_output_size=30, iou_threshold=iou_thresh)
        class_boxes = tf.gather(class_boxes, select)
        class_box_scores = tf.gather(class_box_scores, select)
        _boxes.append(class_boxes)
        _scores.append(class_box_scores)
        _classes.append(tf.ones_like(class_box_scores) * c)

    boxes = tf.concat(_boxes, axis=0)
    classes = tf.concat(_classes, axis=0)
    scores = tf.concat(_scores, axis=0)

    box = tf.concat([tf.reshape(classes, (-1, 1)), tf_corner_to_center(boxes)], axis=-1)

    """ show result """
    if box.shape[0] > 0:
        h.draw_box(img[0].numpy(), box.numpy())
    else:
        print(NOTE, ' no boxes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, help='trian file lists', default='voc')
    parser.add_argument('--class_num', type=int, help='trian class num', default=20)
    parser.add_argument('--model_def', type=str, help='Model definition.', default='yolo_mobilev2')
    parser.add_argument('--depth_multiplier', type=float, help='mobilenet depth_multiplier', choices=[0.5, 0.75, 1.0], default=1.0)
    parser.add_argument('--image_size', type=int, help='net work input image size', default=(224, 320), nargs='+')
    parser.add_argument('--output_size', type=int, help='net work output image size', default=(7, 10, 14, 20), nargs='+')
    parser.add_argument('--obj_thresh', type=float, help='obj mask thresh', default=0.7)
    parser.add_argument('--iou_thresh', type=float, help='iou mask thresh', default=0.3)
    parser.add_argument('pre_ckpt', type=str, help='pre-train weights path')
    parser.add_argument('test_image', type=str, help='test image path')
    args = parser.parse_args(sys.argv[1:])
    main(args.pre_ckpt,
         args.image_size,
         args.output_size,
         args.model_def,
         args.class_num,
         args.depth_multiplier,
         args.obj_thresh,
         args.iou_thresh,
         args.train_set,
         args.test_image)
