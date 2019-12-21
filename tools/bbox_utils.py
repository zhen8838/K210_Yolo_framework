import numpy as np
import tensorflow as tf


def center_to_corner(bbox: np.ndarray, to_all_scale=True, in_hw=None) -> np.ndarray:
    """convert box coordinate from center to corner

    Parameters
    ----------
    bbox : np.ndarray
        bbox [c_x,c_y,w,h]
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : np.ndarray, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        bbox [x1,y1,x2,y2]
    """
    if to_all_scale:
        x1 = (bbox[:, 0:1] - bbox[:, 2:3] / 2) * in_hw[1]
        y1 = (bbox[:, 1:2] - bbox[:, 3:4] / 2) * in_hw[0]
        x2 = (bbox[:, 0:1] + bbox[:, 2:3] / 2) * in_hw[1]
        y2 = (bbox[:, 1:2] + bbox[:, 3:4] / 2) * in_hw[0]
    else:
        x1 = (bbox[:, 0:1] - bbox[:, 2:3] / 2)
        y1 = (bbox[:, 1:2] - bbox[:, 3:4] / 2)
        x2 = (bbox[:, 0:1] + bbox[:, 2:3] / 2)
        y2 = (bbox[:, 1:2] + bbox[:, 3:4] / 2)

    xyxy = np.hstack([x1, y1, x2, y2])
    return xyxy


def tf_center_to_corner(bbox: tf.Tensor, to_all_scale=True, in_hw=None) -> tf.Tensor:
    """convert box coordinate from center to corner

    Parameters
    ----------
    bbox : tf.Tensor
        bbox [c_x,c_y,w,h]
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : tf.Tensor, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        bbox [x1,y1,x2,y2]
    """
    if to_all_scale:
        x1 = (bbox[..., 0:1] - bbox[..., 2:3] / 2) * in_hw[1]
        y1 = (bbox[..., 1:2] - bbox[..., 3:4] / 2) * in_hw[0]
        x2 = (bbox[..., 0:1] + bbox[..., 2:3] / 2) * in_hw[1]
        y2 = (bbox[..., 1:2] + bbox[..., 3:4] / 2) * in_hw[0]
    else:
        x1 = (bbox[..., 0:1] - bbox[..., 2:3] / 2)
        y1 = (bbox[..., 1:2] - bbox[..., 3:4] / 2)
        x2 = (bbox[..., 0:1] + bbox[..., 2:3] / 2)
        y2 = (bbox[..., 1:2] + bbox[..., 3:4] / 2)

    xyxy = tf.concat([x1, y1, x2, y2], -1)
    return xyxy


def corner_to_center(bbox: np.ndarray, from_all_scale=True, in_hw=None) -> np.ndarray:
    """convert box coordinate from corner to center

    Parameters
    ----------
    bbox : np.ndarray
        bbox [x1,y1,x2,y2]
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : np.ndarray, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        bbox [c_x,c_y,w,h]
    """
    if from_all_scale:
        x = ((bbox[..., 2:3] + bbox[..., 0:1]) / 2) / in_hw[1]
        y = ((bbox[..., 3:4] + bbox[..., 1:2]) / 2) / in_hw[0]
        w = (bbox[..., 2:3] - bbox[..., 0:1]) / in_hw[1]
        h = (bbox[..., 3:4] - bbox[..., 1:2]) / in_hw[0]
    else:
        x = ((bbox[..., 2:3] + bbox[..., 0:1]) / 2)
        y = ((bbox[..., 3:4] + bbox[..., 1:2]) / 2)
        w = (bbox[..., 2:3] - bbox[..., 0:1])
        h = (bbox[..., 3:4] - bbox[..., 1:2])

    xywh = np.hstack([x, y, w, h])
    return xywh


def tf_corner_to_center(bbox: tf.Tensor, from_all_scale=True, in_hw=None) -> tf.Tensor:
    """convert box coordinate from corner to center

    Parameters
    ----------
    bbox : tf.Tensor
        bbox [x1,y1,x2,y2]
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : tf.Tensor, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        bbox [c_x,c_y,w,h]
    """
    if from_all_scale:
        x = ((bbox[..., 2:3] + bbox[..., 0:1]) / 2) / in_hw[1]
        y = ((bbox[..., 3:4] + bbox[..., 1:2]) / 2) / in_hw[0]
        w = (bbox[..., 2:3] - bbox[..., 0:1]) / in_hw[1]
        h = (bbox[..., 3:4] - bbox[..., 1:2]) / in_hw[0]
    else:
        x = ((bbox[..., 2:3] + bbox[..., 0:1]) / 2)
        y = ((bbox[..., 3:4] + bbox[..., 1:2]) / 2)
        w = (bbox[..., 2:3] - bbox[..., 0:1])
        h = (bbox[..., 3:4] - bbox[..., 1:2])

    xywh = np.concatenate([x, y, w, h], -1)
    return xywh


def bbox_iou(a: np.ndarray, b: np.ndarray, offset: int = 0) -> np.ndarray:
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    a : np.ndarray

        (n,4) x1,y1,x2,y2

    b : np.ndarray

        (m,4) x1,y1,x2,y2


    offset : int, optional
        by default 0

    Returns
    -------
    np.ndarray

        iou (n,m)
    """
    tl = np.maximum(a[:, None, :2], b[:, :2])
    br = np.minimum(a[:, None, 2:4], b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, :2] + offset, axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def tf_bbox_iou(a: tf.Tensor, b: tf.Tensor, offset: int = 0) -> tf.Tensor:
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    a : tf.Tensor

        (n,4) x1,y1,x2,y2

    b : tf.Tensor

        (m,4) x1,y1,x2,y2


    offset : int, optional
        by default 0

    Returns
    -------
    tf.Tensor

        iou (n,m)
    """

    tl = tf.maximum(a[..., None, :2], b[..., :2])
    br = tf.minimum(a[..., None, 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    return area_i / (area_a[..., None] + area_b - area_i)


def tf_bbox_diou(a: tf.Tensor, b: tf.Tensor, offset: int = 0) -> tf.Tensor:
    """Calculate DIoU of two bounding boxes.

    Parameters
    ----------
    a : tf.Tensor

        (n,4) x1,y1,x2,y2

    b : tf.Tensor

        (m,4) x1,y1,x2,y2

    offset : int, optional
        by default 0

    Returns
    -------
    tf.Tensor

        diou (n,m)
    """

    tl = tf.maximum(a[..., None, :2], b[..., :2])
    br = tf.minimum(a[..., None, 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    iou = area_i / (area_a[..., None] + area_b - area_i)

    # two bbox diagonal distance
    diag = tf.math.reduce_sum(tf.math.square(br - tl + offset), axis=-1)
    # two bbox center distance sum((b_cent-a_cent)^2)
    cent = tf.reduce_sum(tf.square(((b[..., :2] + b[..., 2:]) - (a[..., :2] + a[..., 2:])[..., None, :]) / 2) + offset, -1)

    return iou - cent / diag


def tf_bbox_ciou(a: tf.Tensor, b: tf.Tensor, offset: int = 0) -> tf.Tensor:
    """Calculate CIoU of two bounding boxes.

    Parameters
    ----------
    a : tf.Tensor

        (n,4) x1,y1,x2,y2

    b : tf.Tensor

        (m,4) x1,y1,x2,y2

    offset : int, optional
        by default 0

    Returns
    -------
    tf.Tensor

        diou (n,m)
    """

    tl = tf.maximum(a[..., None, :2], b[..., :2])
    br = tf.minimum(a[..., None, 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    iou = area_i / (area_a[..., None] + area_b - area_i)

    # two bbox diagonal distance
    diag = tf.math.reduce_sum(tf.math.square(br - tl), axis=-1)
    # two bbox center distance sum((b_cent-a_cent)^2)
    cent = tf.reduce_sum(tf.square(((b[..., :2] + b[..., 2:]) - (a[..., :2] + a[..., 2:])[..., None, :]) / 2), -1)
    # calc ciou alpha paramter
    pi = tf.constant(np.pi, tf.float32)
    v = tf.math.square(2 / pi) * tf.math.square(
        tf.math.atan((b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1]))
        - tf.math.atan((a[:, 2] - a[:, 0]) / (a[:, 3] - a[:, 1]))[:, None])

    return iou - (cent / diag + tf.square(v) / (1 - iou + v))  # CIoU


def bbox_iof(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate Intersection-Over-Foreground(IOF) of two bounding boxes.

    Parameters
    ----------
    a : np.ndarray

        (n,4) x1,y1,x2,y2

    b : np.ndarray

        (m,4) x1,y1,x2,y2


    offset : int, optional
        by default 0

    Returns
    -------
    np.ndarray

        iof (n,m)
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float) -> np.ndarray:
    """Pure Python NMS oneclass baseline. 

    Parameters
    ----------
    bbox : np.ndarray

        bbox, n*(x1,y1,x2,y2)

    score : np.ndarray

        confidence score (n,)

    thresh : float

        nms thresh

    Returns
    -------
    np.ndarray
        keep index
    """
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
