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


def bbox_iou(a: np.ndarray, b: np.ndarray, offset: int = 0, method='iou') -> np.ndarray:
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    a : np.ndarray

        (n,4) x1,y1,x2,y2

    b : np.ndarray

        (m,4) x1,y1,x2,y2

    offset : int, optional
        by default 0

    method : str, optional
        by default 'iou', can choice ['iou','giou','diou','ciou']

    Returns
    -------
    np.ndarray

        iou (n,m)
    """
    a = a[..., None, :]
    tl = np.maximum(a[..., :2], b[..., :2])
    br = np.minimum(a[..., 2:4], b[..., 2:4])

    area_i = np.prod(np.maximum(br - tl, 0) + offset, axis=-1)
    area_a = np.prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = np.prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)

    if method == 'iou':
        return area_i / (area_a + area_b - area_i)
    elif method in ['ciou', 'diou']:
        iou = area_i / (area_a + area_b - area_i)
        outer_tl = np.minimum(a[..., :2], b[..., :2])
        outer_br = np.maximum(a[..., 2:4], b[..., 2:4])
        # two bbox center distance sum((b_cent-a_cent)^2)
        inter_diag = np.sum(np.square((b[..., :2] + b[..., 2:]) / 2
                                      - (a[..., :2] + a[..., 2:]) / 2 + offset), -1)
        # two bbox diagonal distance
        outer_diag = np.sum(np.square(outer_tl - outer_br + offset), -1)
        if method == 'diou':
            return iou - inter_diag / outer_diag
        else:
            # calc ciou alpha paramter
            arctan = ((np.math.atan((b[..., 2] - b[..., 0]) / (b[..., 3] - b[..., 1]))
                       - np.math.atan((a[..., 2] - a[..., 0]) / (a[..., 3] - a[..., 1]))))
            v = np.square(2 / np.pi * arctan)
            alpha = v / ((1 - iou) + v)
            w_temp = 2 * (a[..., 2] - a[..., 0])
            ar = (8 / np.square(np.pi)) * arctan * ((a[..., 2] - a[..., 0] - w_temp) * (a[..., 3] - a[..., 1]))
            return np.clip(iou - inter_diag / outer_diag - alpha * ar, -1., 1.)

    elif method in 'giou':
        outer_tl = np.minimum(a[..., :2], b[..., :2])
        outer_br = np.maximum(a[..., 2:4], b[..., 2:4])
        area_o = np.prod(np.maximum(outer_br - outer_tl, 0) + offset, axis=-1)
        union = (area_a + area_b - area_i)
        return (area_i / union) - ((area_o - union) / area_o)


def _get_v(b1_height, b1_width, b2_height, b2_width):
    @tf.custom_gradient
    def _get_grad_v(height, width):
        arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
            tf.math.divide_no_nan(width, height))
        v = 4 * ((arctan / np.pi)**2)

        def _grad_v(dv):
            gdw = dv * 8 * arctan * height / (np.pi**2)
            gdh = -dv * 8 * arctan * width / (np.pi**2)
            return [gdh, gdw]

        return v, _grad_v

    return _get_grad_v(b2_height, b2_width)


def tf_bbox_iou(a: tf.Tensor, b: tf.Tensor, offset: int = 0, method='iou') -> tf.Tensor:
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    a : tf.Tensor

        (n,4) x1,y1,x2,y2

    b : tf.Tensor

        (m,4) x1,y1,x2,y2


    offset : int, optional
        by default 0

    method : str, optional
        by default 'iou', can choice ['iou','giou','diou','ciou']

    Returns
    -------
    tf.Tensor

        iou (n,m)
    """
    a = a[..., None, :]
    tl = tf.maximum(a[..., :2], b[..., :2])
    br = tf.minimum(a[..., 2:4], b[..., 2:4])

    area_i = tf.reduce_prod(tf.maximum(br - tl, 0) + offset, axis=-1)
    area_a = tf.reduce_prod(a[..., 2:4] - a[..., :2] + offset, axis=-1)
    area_b = tf.reduce_prod(b[..., 2:4] - b[..., :2] + offset, axis=-1)
    if method == 'iou':
        return area_i / (area_a + area_b - area_i)
    elif method in ['ciou', 'diou']:
        iou = area_i / (area_a + area_b - area_i)

        outer_tl = tf.minimum(a[..., :2], b[..., :2])
        outer_br = tf.maximum(a[..., 2:4], b[..., 2:4])
        # two bbox center distance sum((b_cent-a_cent)^2)
        inter_diag = tf.reduce_sum(tf.square((b[..., :2] + b[..., 2:]) / 2
                                             - (a[..., :2] + a[..., 2:]) / 2 + offset), -1)
        # two bbox diagonal distance
        outer_diag = tf.reduce_sum(tf.square(outer_tl - outer_br + offset), -1)
        if method == 'diou':
            return iou - inter_diag / outer_diag
        else:
            # # calc ciou alpha paramter
            # arctan = tf.stop_gradient(
            #     (tf.math.atan(tf.math.divide_no_nan(b[..., 2] - b[..., 0],
            #                                         b[..., 3] - b[..., 1]))
            #      - tf.math.atan(tf.math.divide_no_nan(a[..., 2] - a[..., 0],
            #                                           a[..., 3] - a[..., 1]))))

            # v = tf.stop_gradient(tf.math.square(2 / np.pi * arctan))
            # alpha = tf.stop_gradient(v / ((1 - iou) + v))
            # w_temp = tf.stop_gradient(2 * (a[..., 2] - a[..., 0]))

            # ar = (8 / tf.square(np.pi)) * arctan * ((a[..., 2] - a[..., 0] - w_temp) * (a[..., 3] - a[..., 1]))
            v = _get_v(a[..., 3] - a[..., 1], a[..., 2] - a[..., 0],
                       b[..., 3] - b[..., 1], b[..., 2] - b[..., 0])
            alpha = tf.math.divide_no_nan(v, ((1 - iou) + v))

            return iou - inter_diag / outer_diag - alpha * v

    elif method in 'giou':
        outer_tl = tf.minimum(a[..., :2], b[..., :2])
        outer_br = tf.maximum(a[..., 2:4], b[..., 2:4])
        area_o = tf.reduce_prod(tf.maximum(outer_br - outer_tl, 0) + offset, axis=-1)
        union = (area_a + area_b - area_i)
        return (area_i / union) - ((area_o - union) / area_o)


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


def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float, method='iou') -> np.ndarray:
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
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        iou = bbox_iou(bbox[i], bbox[order[1:]], method=method)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep
