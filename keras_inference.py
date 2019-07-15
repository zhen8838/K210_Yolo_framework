import tensorflow.python as tf
from tensorflow.python import keras
from pathlib import Path
from tools.utils import Helper, INFO, ERROR, NOTE, tf_xywh_to_all
from models.yolonet import *
from termcolor import colored
from PIL import Image, ImageFont, ImageDraw
import argparse
import sys
import numpy as np

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
keras.backend.set_learning_phase(0)


# obj_thresh = 0.7
# iou_thresh = 0.3
# ckpt_weights = 'log/20190709-203142/yolo_model.h5'
# image_size = [224, 320]
# output_size = [7, 10, 14, 20]
# model_def = 'yolo_mobilev2'
# class_num = 20
# depth_multiplier = 0.75
# train_set = 'voc'
# test_image = 'tmp/bb.jpg'


def correct_box(box_xy: tf.Tensor, box_wh: tf.Tensor, input_shape: list, image_shape: list) -> tf.Tensor:
    """rescae predict box to orginal image scale

    Parameters
    ----------
    box_xy : tf.Tensor
        box xy
    box_wh : tf.Tensor
        box wh
    input_shape : list
        input shape
    image_shape : list
        image shape

    Returns
    -------
    tf.Tensor
        new boxes
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


def main(ckpt_weights, image_size, output_size, model_def, class_num, depth_multiplier, obj_thresh, iou_thresh, train_set, test_image):
    h = Helper(None, class_num, f'data/{train_set}_anchor.npy', np.reshape(np.array(image_size), (-1, 2)), np.reshape(np.array(output_size), (-1, 2)))
    network = eval(model_def)  # type :yolo_mobilev2
    yolo_model, yolo_model_warpper = network([image_size[0], image_size[1], 3], len(h.anchors[0]), class_num, alpha=depth_multiplier)

    yolo_model_warpper.load_weights(str(ckpt_weights))
    print(INFO, f' Load CKPT {str(ckpt_weights)}')
    orig_img = h._read_img(str(test_image))
    image_shape = orig_img.shape[0:2]
    img, _ = h._process_img(orig_img, true_box=None, is_training=False, is_resize=True)

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
        boxes = correct_box(pred_xy_A, pred_wh_A, image_size, image_shape)
        boxes = tf.reshape(boxes, (-1, 4))
        box_scores = tf.reshape(box_scores, (-1, class_num))
        """ append box and scores to global list """
        _yxyx_box.append(boxes)
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

    """ draw box  """
    font = ImageFont.truetype(font='asset/FiraMono-Medium.otf',
                              size=tf.cast(tf.floor(3e-2 * image_shape[0] + 0.5), tf.int32).numpy())

    thickness = (image_shape[0] + image_shape[1]) // 300

    """ show result """
    if len(classes) > 0:
        pil_img = Image.fromarray(orig_img)
        print(f'[top\tleft\tbottom\tright\tscore\tclass]')
        for i, c in enumerate(classes):
            box = boxes[i]
            score = scores[i]
            label = '{:2d} {:.2f}'.format(int(c.numpy()), score.numpy())
            draw = ImageDraw.Draw(pil_img)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            print(f'[{top:.1f}\t{left:.1f}\t{bottom:.1f}\t{right:.1f}\t{score:.2f}\t{int(c):2d}]')
            top = max(0, tf.cast(tf.floor(top + 0.5), tf.int32))
            left = max(0, tf.cast(tf.floor(left + 0.5), tf.int32))
            bottom = min(image_shape[0], tf.cast(tf.floor(bottom + 0.5), tf.int32))
            right = min(image_shape[1], tf.cast(tf.floor(right + 0.5), tf.int32))

            if top - image_shape[0] >= 0:
                text_origin = tf.convert_to_tensor([left, top - label_size[1]])
            else:
                text_origin = tf.convert_to_tensor([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=h.colormap[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=h.colormap[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        pil_img.show()
    else:
        print(NOTE, ' no boxes detected')


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
