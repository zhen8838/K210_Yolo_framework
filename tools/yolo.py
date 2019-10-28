import numpy as np
import os
import cv2
from skimage.draw import rectangle_perimeter, circle
from skimage.io import imshow, imread, imsave, show
from skimage.color import gray2rgb
from skimage.transform import AffineTransform, warp
from math import cos, sin
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras as k
from tensorflow.contrib.data import assert_element_shape
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from matplotlib.pyplot import text
from PIL import Image, ImageFont, ImageDraw
from tools.base import BaseHelper, INFO, ERROR, NOTE
from pathlib import Path
from tqdm import trange
from termcolor import colored


def fake_iou(a: np.ndarray, b: np.ndarray) -> float:
    """set a,b center to same,then calc the iou value

    Parameters
    ----------
    a : np.ndarray
        shape = [n,1,2]
    b : np.ndarray
        shape = [m,2]

    Returns
    -------
    float
        iou value
        shape = [n,m]
    """
    a_maxes = a / 2.
    a_mins = -a_maxes

    b_maxes = b / 2.
    b_mins = -b_maxes

    iner_mins = np.maximum(a_mins, b_mins)
    iner_maxes = np.minimum(a_maxes, b_maxes)
    iner_wh = np.maximum(iner_maxes - iner_mins, 0.)
    iner_area = iner_wh[..., 0] * iner_wh[..., 1]

    s1 = a[..., 0] * a[..., 1]
    s2 = b[..., 0] * b[..., 1]

    return iner_area / (s1 + s2 - iner_area)


def coordinate_offset(anchors: np.ndarray, out_hw: np.ndarray) -> np.array:
    """construct the anchor coordinate offset array , used in convert scale

    Parameters
    ----------
    anchors : np.ndarray
        anchors shape = [n,] = [ n x [m,2]]
    out_hw : np.ndarray
        output height width shape = [n,2]

    Returns
    -------
    np.array
        scale shape = [n,] = [n x [h_n,w_n,m,2]]
    """
    if len(anchors) != len(out_hw):
        raise ValueError(f'anchors len {len(anchors)} is not equal out_hw len {len(out_hw)}')
    grid = []
    for l in range(len(anchors)):
        grid_y = np.tile(np.reshape(np.arange(0, stop=out_hw[l][0]), [-1, 1, 1, 1]), [1, out_hw[l][1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(0, stop=out_hw[l][1]), [1, -1, 1, 1]), [out_hw[l][0], 1, 1, 1])
        grid.append(np.concatenate([grid_x, grid_y], axis=-1))
    return np.array(grid)


def anchor_scale(anchors: np.ndarray, grid_wh: np.ndarray) -> np.array:
    """construct the anchor scale array , used in convert label to annotation

    Parameters
    ----------
    anchors : np.ndarray
        anchors shape = [n,] = [ n x [m,2]]
    out_hw : np.ndarray
        output height width shape = [n,2]

    Returns
    -------
    np.array
        scale shape = [n,] = [n x [m,2]]
    """
    return np.array([anchors[i] * grid_wh[i] for i in range(len(anchors))])


def center_to_corner(xywh_ann: np.ndarray, to_all_scale=True, in_hw=None) -> np.ndarray:
    """convert box coordinate from center to corner

    Parameters
    ----------
    xywh_ann : np.ndarray
        true box
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : np.ndarray, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        xyxy annotation
    """
    if to_all_scale:
        x1 = (xywh_ann[:, 0:1] - xywh_ann[:, 2:3] / 2) * in_hw[1]
        y1 = (xywh_ann[:, 1:2] - xywh_ann[:, 3:4] / 2) * in_hw[0]
        x2 = (xywh_ann[:, 0:1] + xywh_ann[:, 2:3] / 2) * in_hw[1]
        y2 = (xywh_ann[:, 1:2] + xywh_ann[:, 3:4] / 2) * in_hw[0]
    else:
        x1 = (xywh_ann[:, 0:1] - xywh_ann[:, 2:3] / 2)
        y1 = (xywh_ann[:, 1:2] - xywh_ann[:, 3:4] / 2)
        x2 = (xywh_ann[:, 0:1] + xywh_ann[:, 2:3] / 2)
        y2 = (xywh_ann[:, 1:2] + xywh_ann[:, 3:4] / 2)

    xyxy_ann = np.hstack([x1, y1, x2, y2])
    return xyxy_ann


def corner_to_center(xyxy_ann: np.ndarray, from_all_scale=True, in_hw=None) -> np.ndarray:
    """convert box coordinate from corner to center

    Parameters
    ----------
    xyxy_ann : np.ndarray
        xyxy box (upper left,bottom right)
    to_all_scale : bool, optional
        weather to all image scale, by default True
    in_hw : np.ndarray, optional
        in hw, by default None

    Returns
    -------
    np.ndarray
        xywh annotation
    """
    if from_all_scale:
        x = ((xyxy_ann[:, 2:3] + xyxy_ann[:, 0:1]) / 2) / in_hw[1]
        y = ((xyxy_ann[:, 3:4] + xyxy_ann[:, 1:2]) / 2) / in_hw[0]
        w = (xyxy_ann[:, 2:3] - xyxy_ann[:, 0:1]) / in_hw[1]
        h = (xyxy_ann[:, 3:4] - xyxy_ann[:, 1:2]) / in_hw[0]
    else:
        x = ((xyxy_ann[:, 2:3] + xyxy_ann[:, 0:1]) / 2)
        y = ((xyxy_ann[:, 3:4] + xyxy_ann[:, 1:2]) / 2)
        w = (xyxy_ann[:, 2:3] - xyxy_ann[:, 0:1])
        h = (xyxy_ann[:, 3:4] - xyxy_ann[:, 1:2])

    xywh_ann = np.hstack([x, y, w, h])
    return xywh_ann


class YOLOHelper(BaseHelper):
    def __init__(self, image_ann: str, class_num: int, anchors: str,
                 in_hw: tuple, out_hw: tuple, validation_split=0.1):
        super().__init__(image_ann, validation_split)
        self.in_hw = np.array(in_hw)
        assert self.in_hw.ndim == 1
        self.out_hw = np.array(out_hw)
        assert self.out_hw.ndim == 2
        self.grid_wh = (1 / self.out_hw)[:, [1, 0]]  # hw 转 wh 需要交换两列
        if class_num:
            self.class_num = class_num  # type:int
        if anchors:
            self.anchors = np.load(anchors)  # type:np.ndarray
            self.anchor_number = len(self.anchors[0])
            self.output_number = len(self.anchors)
            self.__flatten_anchors = np.reshape(self.anchors, (-1, 2))
            self.xy_offset = coordinate_offset(self.anchors, self.out_hw)  # type:np.ndarray
            self.wh_scale = anchor_scale(self.anchors, self.grid_wh)  # type:np.ndarray

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.OneOf

        self.colormap = [
            (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
            (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
            (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
            (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
            (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
            (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
            (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214),
            (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
            (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), (255, 122, 8),
            (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
            (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), (20, 255, 0),
            (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
            (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200)]

    def _xy_to_grid(self, xy: np.ndarray, layer: int) -> np.ndarray:
        """ convert true label xy to grid scale

        Parameters
        ----------
        xy : np.ndarray
            label xy shape = [out h, out w,anchor num,2]
        layer : int
            layer index

        Returns
        -------
        np.ndarray
            label xy grid scale, shape = [out h, out w,anchor num,2]
        """
        return (xy * self.out_hw[layer][::-1]) - self.xy_offset[layer]

    def _xy_grid_index(self, box_xy: np.ndarray, layer: int) -> [np.ndarray, np.ndarray]:
        """ get xy index in grid scale

        Parameters
        ----------
        box_xy : np.ndarray
            value = [x,y]
        layer  : int
            layer index

        Returns
        -------
        [np.ndarray,np.ndarray]

            index xy : = [idx,idy]
        """
        return np.floor(box_xy * self.out_hw[layer][::-1]).astype('int')

    def _get_anchor_index(self, wh: np.ndarray) -> [np.ndarray, np.ndarray]:
        """get the max iou anchor index

        Parameters
        ----------
        wh : np.ndarray
            shape = [num_box,2]
            value = [w,h]

        Returns
        -------
        np.ndarray, np.ndarray
            max iou anchor index
            layer_idx shape = [num_box,1]
            anchor_idx shape = [num_box,1]
        """
        iou = fake_iou(np.expand_dims(wh, -2), self.__flatten_anchors)
        best_anchor = np.argmax(iou, -1)
        return np.divmod(best_anchor, self.anchor_number)

    def ann_to_label(self, ann: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 label~

        Parameters
        ----------
        ann : np.ndarray
            annotation shape :[n,5] value :[n*[p,x,y,w,h]]

        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1], len(self.anchors[i]),
                            5 + self.class_num), dtype='float32') for i in range(self.output_number)]

        layer_idx, anchor_idx = self._get_anchor_index(ann[:, 3:5])
        for box, l, n in zip(ann, layer_idx, anchor_idx):
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            idx, idy = self._xy_grid_index(box[1:3], l)  # [x index , y index]
            # Note clip box in [1e-8,1.] avoid width or heigh == 0 ====> loss = inf
            labels[l][idy, idx, n, 0:4] = np.clip(box[1:5], 1e-8, 1.)
            labels[l][idy, idx, n, 4] = 1.
            labels[l][idy, idx, n, 5 + int(box[0])] = 1.

        return labels

    def _xy_to_all(self, labels: tuple):
        """convert xy scale from grid to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 0:2] = labels[i][..., 0:2] * self.grid_wh[i] + self.xy_offset[i]

    def _wh_to_all(self, labels: tuple):
        """convert wh scale to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 2:4] = np.exp(labels[i][..., 2: 4]) * self.anchors[i]

    def label_to_ann(self, labels: tuple, thersh=.7) -> np.ndarray:
        """reverse the labels to annotation

        Parameters
        ----------
        labels : np.ndarray

        Returns
        -------
        np.ndarray
            annotaions
        """
        new_ann = np.vstack([label[np.where(label[..., 4] > thersh)] for label in labels])
        new_ann = np.c_[np.argmax(new_ann[:, 5:], axis=-1), new_ann[:, :4]]
        return new_ann

    def data_augmenter(self, img: np.ndarray, ann: np.ndarray) -> tuple:
        """ augmenter for image

        Parameters
        ----------
        img : np.ndarray
            img src
        ann : np.ndarray
            one annotation

        Returns
        -------
        tuple
            [image src,box] after data augmenter
            image src dtype is uint8
        """
        p = ann[:, 0:1]
        xywh_box = ann[:, 1:]

        bbs = BoundingBoxesOnImage.from_xyxy_array(
            center_to_corner(xywh_box, in_hw=img.shape[0:2]), shape=img.shape)

        image_aug, bbs_aug = self.iaaseq(image=img, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        xyxy_box = bbs_aug.to_xyxy_array()
        new_ann = corner_to_center(xyxy_box, in_hw=img.shape[0:2])
        new_ann = np.hstack((p[0:new_ann.shape[0], :], new_ann))

        return image_aug, new_ann

    def resize_img(self, img: np.ndarray, ann: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        resize image and keep ratio

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray


        Returns
        -------
        [np.ndarray, np.ndarray]
            img, ann [uint8,float64]
        """
        img_wh = np.array(img.shape[1::-1])
        in_wh = self.in_hw[::-1]

        """ calculate the affine transform factor """
        scale = in_wh / img_wh  # NOTE affine tranform sacle is [w,h]
        scale[:] = np.min(scale)
        # NOTE translation is [w offset,h offset]
        translation = ((in_wh - img_wh * scale) / 2).astype(int)

        """ calculate the box transform matrix """
        if isinstance(ann, np.ndarray):
            ann[:, 1:3] = (ann[:, 1:3] * img_wh * scale + translation) / in_wh
            ann[:, 3:5] = (ann[:, 3:5] * img_wh * scale) / in_wh
        elif isinstance(ann, tf.Tensor):
            # NOTE use concat replace item assign
            ann = tf.concat((ann[:, 0:1],
                             (ann[:, 1:3] * img_wh * scale + translation) / in_wh,
                             (ann[:, 3:5] * img_wh * scale) / in_wh), axis=1)

        """ apply Affine Transform """
        aff = AffineTransform(scale=scale, translation=translation)
        img = warp(img, aff.inverse, output_shape=self.in_hw, preserve_range=True).astype('uint8')
        return img, ann

    def read_img(self, img_path: str) -> np.ndarray:
        """ read image

        Parameters
        ----------
        img_path : str


        Returns
        -------
        np.ndarray
            image src
        """
        img = imread(img_path)
        if len(img.shape) != 3:
            img = gray2rgb(img)
        return img[..., :3]

    def _compute_dataset_shape(self) -> tuple:
        """ compute dataset shape to avoid keras check shape error

        Returns
        -------
        tuple
            dataset shape lists
        """
        output_shapes = [tf.TensorShape([None] + list(self.out_hw[i]) + [len(self.anchors[i]), self.class_num + 5])
                         for i in range(len(self.anchors))]
        dataset_shapes = (tf.TensorShape([None] + list(self.in_hw) + [3]), tuple(output_shapes))
        return dataset_shapes

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       rand_seed: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))

        def _parser_wrapper(i: tf.Tensor):
            # NOTE use wrapper function and dynamic list construct (x,(y_1,y_2,...))
            img_path, ann = tf.numpy_function(lambda idx: (image_ann_list[idx][0].copy(), image_ann_list[idx][1].copy()),
                                              [i], [tf.dtypes.string, tf.float64])
            # load image
            raw_img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
            # resize image -> image augmenter
            raw_img, ann = tf.numpy_function(self.process_img,
                                             [raw_img, ann, is_augment, True, False],
                                             [tf.uint8, tf.float64])
            # make labels
            labels = tf.numpy_function(self.ann_to_label, [ann], [tf.float32] * len(self.anchors))

            # normlize image
            if is_normlize:
                img = self.normlize_img(raw_img)
            else:
                img = tf.cast(raw_img, tf.float32)

            return img, tuple(labels)

        dataset_shapes = self._compute_dataset_shape()
        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                       shuffle(batch_size * 500 if is_training == True else batch_size * 50, rand_seed).
                       repeat().
                       map(_parser_wrapper, -1).
                       batch(batch_size, True).prefetch(-1).
                       apply(assert_element_shape(dataset_shapes)))
        else:
            dataset = (tf.data.Dataset.from_tensor_slices(
                tf.range(len(image_ann_list))).
                map(_parser_wrapper, -1).
                batch(batch_size, True).prefetch(-1).
                apply(assert_element_shape(dataset_shapes)))

        return dataset

    def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True, scores=None):
        """ draw img and show bbox , set ann = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

           shape : [p,x,y,w,h]

        is_show : bool

            show image
        """
        if isinstance(ann, np.ndarray):
            p = ann[:, 0]
            xyxybox = center_to_corner(ann[:, 1:], in_hw=img.shape[:2])
            for i, a in enumerate(xyxybox):
                classes = int(p[i])
                r_top = tuple(a[0:2].astype(int))
                l_bottom = tuple(a[2:].astype(int))
                r_bottom = (r_top[0], l_bottom[1])
                org = (np.maximum(np.minimum(r_bottom[0], img.shape[1] - 12), 0),
                       np.maximum(np.minimum(r_bottom[1], img.shape[0] - 12), 0))
                cv2.rectangle(img, r_top, l_bottom, self.colormap[classes])
                if isinstance(scores, np.ndarray):
                    cv2.putText(img, f'{classes} {scores[i]:.2f}',
                                org, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                0.5, self.colormap[classes], thickness=1)
                else:
                    cv2.putText(img, f'{classes}', org,
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                self.colormap[classes], thickness=1)

        if is_show:
            imshow((img).astype('uint8'))
            show()


def tf_xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor,
                   layer: int, h: YOLOHelper) -> [tf.Tensor, tf.Tensor]:
    """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

    Parameters
    ----------
    grid_pred_xy : tf.Tensor

    grid_pred_wh : tf.Tensor

    layer : int
        the output layer
    h : YOLOHelper


    Returns
    -------
    tuple

        after process, [all_pred_xy, all_pred_wh]
    """
    with tf.name_scope('xywh_to_all_%d' % layer):
        all_pred_xy = (tf.sigmoid(grid_pred_xy) + h.xy_offset[layer]) / h.out_hw[layer][::-1]
        all_pred_wh = tf.exp(grid_pred_wh) * h.anchors[layer]
    return all_pred_xy, all_pred_wh


def tf_xywh_to_grid(all_true_xy: tf.Tensor, all_true_wh: tf.Tensor, layer: int, h: YOLOHelper) -> [tf.Tensor, tf.Tensor]:
    """convert true label xy wh to grid scale

    Parameters
    ----------
    all_true_xy : tf.Tensor

    all_true_wh : tf.Tensor

    layer : int
        layer index
    h : YOLOHelper


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        grid_true_xy, grid_true_wh shape = [out h ,out w,anchor num , 2 ]
    """
    with tf.name_scope('xywh_to_grid_%d' % layer):
        grid_true_xy = (all_true_xy * h.out_hw[layer][::-1]) - h.xy_offset[layer]
        grid_true_wh = tf.math.log(all_true_wh / h.anchors[layer])
    return grid_true_xy, grid_true_wh


def tf_iou(pred_xy: tf.Tensor, pred_wh: tf.Tensor,
           vaild_xy: tf.Tensor, vaild_wh: tf.Tensor) -> tf.Tensor:
    """ calc the iou form pred box with vaild box

    Parameters
    ----------
    pred_xy : tf.Tensor
        pred box shape = [out h, out w, anchor num, 2]

    pred_wh : tf.Tensor
        pred box shape = [out h, out w, anchor num, 2]

    vaild_xy : tf.Tensor
        vaild box shape = [? , 2]

    vaild_wh : tf.Tensor
        vaild box shape = [? , 2]

    Returns
    -------
    tf.Tensor
        iou value shape = [out h, out w, anchor num ,?]
    """
    b1_xy = tf.expand_dims(pred_xy, -2)
    b1_wh = tf.expand_dims(pred_wh, -2)
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = tf.expand_dims(vaild_xy, 0)
    b2_wh = tf.expand_dims(vaild_wh, 0)
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def calc_ignore_mask(t_xy_A: tf.Tensor, t_wh_A: tf.Tensor, p_xy: tf.Tensor,
                     p_wh: tf.Tensor, obj_mask: tf.Tensor, iou_thresh: float,
                     layer: int, helper: YOLOHelper) -> tf.Tensor:
    """clac the ignore mask

    Parameters
    ----------
    t_xy_A : tf.Tensor
        raw ture xy,shape = [batch size,h,w,anchors,2]
    t_wh_A : tf.Tensor
        raw true wh,shape = [batch size,h,w,anchors,2]
    p_xy : tf.Tensor
        raw pred xy,shape = [batch size,h,w,anchors,2]
    p_wh : tf.Tensor
        raw pred wh,shape = [batch size,h,w,anchors,2]
    obj_mask : tf.Tensor
        old obj mask,shape = [batch size,h,w,anchors]
    iou_thresh : float
        iou thresh
    helper : YOLOHelper
        YOLOHelper obj

    Returns
    -------
    tf.Tensor
    ignore_mask :
        ignore_mask, shape = [batch size, h, w, anchors, 1]
    """
    with tf.name_scope('calc_mask_%d' % layer):
        pred_xy, pred_wh = tf_xywh_to_all(p_xy, p_wh, layer, helper)

        def lmba(bc):
            vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
            vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
            iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
            best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
            return tf.cast(best_iou < iou_thresh, tf.float32)

    return tf.map_fn(lmba, tf.range(helper.batch_size), dtype=tf.float32)


class YOLO_Loss(Loss):
    def __init__(self, h: YOLOHelper, obj_thresh: float, iou_thresh: float,
                 obj_weight: float, noobj_weight: float, wh_weight: float,
                 layer: int, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """ yolo loss obj

        Parameters
        ----------
        h : YOLOHelper

        obj_thresh : float

        iou_thresh : float

        obj_weight : float

        noobj_weight : float

        wh_weight : float

        layer : int
            the current layer index

        """
        super().__init__(reduction=reduction, name=name)
        self.h = h
        self.obj_thresh = obj_thresh
        self.iou_thresh = iou_thresh
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.wh_weight = wh_weight
        self.layer = layer

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ split the label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        pred_cls = y_pred[..., 5:]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        true_confidence = y_true[..., 4:5]
        true_cls = y_true[..., 5:]

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh
        obj_mask_bool = y_true[..., 4] > self.obj_thresh

        """ calc the ignore mask  """

        ignore_mask = calc_ignore_mask(all_true_xy, all_true_wh, grid_pred_xy,
                                       grid_pred_wh, obj_mask_bool,
                                       self.iou_thresh, self.layer, self.h)

        grid_true_xy, grid_true_wh = tf_xywh_to_grid(all_true_xy, all_true_wh, self.layer, self.h)
        # NOTE When wh=0 , tf.log(0) = -inf, so use tf.where to avoid it
        grid_true_wh = tf.where(tf.tile(obj_mask_bool[..., tf.newaxis], [1, 1, 1, 1, 2]),
                                grid_true_wh, tf.zeros_like(grid_true_wh))

        """ define loss """
        coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

        xy_loss = tf.reduce_sum(
            obj_mask * coord_weight * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=grid_true_xy, logits=grid_pred_xy), [1, 2, 3, 4])

        wh_loss = tf.reduce_sum(
            obj_mask * coord_weight * self.wh_weight * tf.square(tf.subtract(
                x=grid_true_wh, y=grid_pred_wh)), [1, 2, 3, 4])

        obj_loss = self.obj_weight * tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

        noobj_loss = self.noobj_weight * tf.reduce_sum(
            (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence), [1, 2, 3, 4])

        cls_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_cls, logits=pred_cls), [1, 2, 3, 4])

        total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss

        return total_loss


def correct_box(box_xy: tf.Tensor, box_wh: tf.Tensor,
                input_hw: list, img_hw: list) -> tf.Tensor:
    """rescae predict box to orginal image scale

    Parameters
    ----------
    box_xy : tf.Tensor
        box xy
    box_wh : tf.Tensor
        box wh
    input_hw : list
        input shape
    img_hw : list
        image shape

    Returns
    -------
    tf.Tensor
        new boxes
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_hw = tf.cast(input_hw, tf.float32)
    img_hw = tf.cast(img_hw, tf.float32)
    new_shape = tf.round(img_hw * tf.reduce_min(input_hw / img_hw))
    offset = (input_hw - new_shape) / 2. / input_hw
    scale = input_hw / new_shape
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
    boxes *= tf.concat([img_hw, img_hw], axis=-1)
    return boxes


def yolo_parser_one(img: tf.Tensor, img_hw: np.ndarray, infer_model: k.Model,
                    obj_thresh: float, iou_thresh: float, h: YOLOHelper
                    ) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ yolo parser one image output

    Parameters
    ----------
    img : tf.Tensor

        image src, shape = [1,in h,in w,3]

    img_hw : np.ndarray

        image orginal hw, shape = [2]

    infer_model : k.Model

        infer model

    obj_thresh : float

    iou_thresh : float

    h : YOLOHelper

    Returns
    -------
    [np.ndarray, np.ndarray, np.ndarray]

        box : [y1, x1, y2, y2]
        clss : [class]
        score : [score]
    """
    y_pred = infer_model.predict(img)
    # NOTE because yolo train model and infer model is same,
    # In order to ensure the consistency of the framework code reshape here.
    y_pred = [np.reshape(pred, list(pred.shape[:-1]) + [h.anchor_number, 5 + h.class_num])
              for pred in y_pred]
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
        # obj_mask = pred_confidence_score[..., 0] > model.obj_thresh
        """ reshape box  """
        # NOTE tf_xywh_to_all will auto use sigmoid function
        pred_xy_A, pred_wh_A = tf_xywh_to_all(pred_xy, pred_wh, l, h)
        boxes = correct_box(pred_xy_A, pred_wh_A, h.in_hw, img_hw)
        boxes = tf.reshape(boxes, (-1, 4))
        box_scores = tf.reshape(box_scores, (-1, h.class_num))
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
    for c in range(h.class_num):
        class_boxes = tf.boolean_mask(yxyx_box, mask[:, c])
        class_box_scores = tf.boolean_mask(yxyx_box_scores[:, c], mask[:, c])
        select = tf.image.non_max_suppression(
            class_boxes, scores=class_box_scores, max_output_size=30, iou_threshold=iou_thresh)
        class_boxes = tf.gather(class_boxes, select)
        class_box_scores = tf.gather(class_box_scores, select)
        _boxes.append(class_boxes)
        _scores.append(class_box_scores)
        _classes.append(tf.ones_like(class_box_scores) * c)

    box = tf.concat(_boxes, axis=0)
    clss = tf.concat(_classes, axis=0)
    score = tf.concat(_scores, axis=0)
    return box.numpy(), clss.numpy(), score.numpy()


def yolo_infer(img_path: Path, infer_model: k.Model,
               result_path: Path, h: YOLOHelper,
               obj_thresh: float = .7, iou_thresh: float = .3):
    """ yolo infer function

    Parameters
    ----------
    img_path : Path

        image path or image dir path

    infer_model : k.Model

        infer model

    result_path : Path

        result path dir

    h : YOLOHelper

    obj_thresh : float, optional

        object detection thresh, by default .7

    iou_thresh : float, optional

        iou thresh , by default .3

    """

    """ load images """
    orig_img = h.read_img(str(img_path))
    img_hw = orig_img.shape[0:2]
    img, _ = h.process_img(orig_img, None, False, True, True)
    img = tf.expand_dims(img, 0)
    """ get output """
    boxes, classes, scores = yolo_parser_one(img, img_hw, infer_model, obj_thresh, iou_thresh, h)
    """ draw box  """
    font = ImageFont.truetype(font='asset/FiraMono-Medium.otf',
                              size=(np.floor(3e-2 * img_hw[0] + 0.5)).astype(np.int))

    thickness = (img_hw[0] + img_hw[1]) // 300

    """ show result """
    if len(classes) > 0:
        pil_img = Image.fromarray(orig_img)
        print(f'[top\tleft\tbottom\tright\tscore\tclass]')
        for i, c in enumerate(classes):
            c = int(c)
            box = boxes[i]
            score = scores[i]
            label = '{:2d} {:.2f}'.format(c, score)
            draw = ImageDraw.Draw(pil_img)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            print(f'[{top:.1f}\t{left:.1f}\t{bottom:.1f}\t{right:.1f}\t{score:.2f}\t{c:2d}]')
            top = max(0, (np.floor(top + 0.5)).astype(np.int))
            left = max(0, (np.floor(left + 0.5)).astype(np.int))
            bottom = min(img_hw[0], (np.floor(bottom + 0.5)).astype(np.int))
            right = min(img_hw[1], (np.floor(right + 0.5)).astype(np.int))

            if top - img_hw[0] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

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


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """ 
        Compute VOC AP given precision and recall

    Parameters
    ----------
    recall : np.ndarray

        recall, shape = [len(imgs)]

    precision : np.ndarray

        precison, shape = [len(imgs)]


    Returns
    -------

    float

        ap

    """

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    #  where X axis (recall) changes value

    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    Ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return Ap


def yolo_eval(infer_model: k.Model, h: YOLOHelper, det_obj_thresh: float,
              det_iou_thresh: float, mAp_iou_thresh: float):
    """ calc yolo pre-class Ap and mAp

    Parameters
    ----------
    infer_model : k.Model

    h : YOLOHelper

    det_obj_thresh : float

        detection obj thresh

    det_iou_thresh : float

        detection iou thresh

    mAp_iou_thresh : float

        mAp iou thresh
    """
    p_img_ids = [[] for i in range(h.class_num)]
    p_scores = [[] for i in range(h.class_num)]
    p_bboxes = [[] for i in range(h.class_num)]
    t_res = [{} for i in range(h.class_num)]  # type:list[dict]
    t_npos = np.zeros((h.class_num, 1))
    for i in trange(len(h.test_list)):
        img_path, true_ann, img_hw = h.test_list[i]
        img_name = Path(img_path).stem
        raw_img = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        img, _ = h.process_img(raw_img.numpy(), None, False, True, True)
        img = img[tf.newaxis, ...]

        p_yxyx, p_clas, p_score = yolo_parser_one(img, img_hw,
                                                  infer_model, det_obj_thresh,
                                                  det_iou_thresh, h)

        for j, c in enumerate(p_clas.astype(np.int)):
            p_img_ids[c].append(img_name)
            p_scores[c].append(p_score[j])
            p_bboxes[c].append(np.array(
                [np.maximum(p_yxyx[j, 1], 0),
                 np.maximum(p_yxyx[j, 0], 0),
                 np.minimum(p_yxyx[j, 3], img_hw[1]),
                 np.minimum(p_yxyx[j, 2], img_hw[0])]))

        true_clas, true_box = np.split(true_ann, [1], -1)
        true_xyxy = center_to_corner(true_box, in_hw=img_hw)
        true_clas = np.ravel(true_clas)
        for c in true_clas.astype(np.int):
            t_res[c][img_name] = {
                'bbox': true_xyxy,
                'det': [False] * len(true_xyxy)}
            t_npos[c] = t_npos[c] + 1

    p_img_ids = np.array([np.array(i, dtype=np.str) for i in p_img_ids])
    p_scores = np.array([np.array(i) for i in p_scores])
    p_bboxes = np.array([np.stack(i) for i in p_bboxes])

    """ sorted pre-classes by scores """
    sorted_ind = np.array([np.argsort(-s) for s in p_scores])
    sorted_scores = np.array([np.sort(-s) for s in p_scores])
    p_bboxes = np.array([p_bboxes[i][sorted_ind[i]] for i, b in enumerate(p_bboxes)])
    p_img_ids = np.array([p_img_ids[i][sorted_ind[i]] for i, b in enumerate(p_img_ids)])

    """ calc pre-classes tp and fp """
    nd = [len(i) for i in p_img_ids]
    tp = np.array([np.zeros(nd[i]) for i in range(h.class_num)])
    fp = np.array([np.zeros(nd[i]) for i in range(h.class_num)])
    for c in range(h.class_num):
        for d in range(nd[c]):
            if p_img_ids[c][d] in t_res[c]:
                gt = t_res[c].get(p_img_ids[c][d])  # type:dict
            else:
                continue
            bb = p_bboxes[c][d]
            gtbb = gt['bbox']
            if len(gtbb) > 0:
                ixmin = np.maximum(gtbb[:, 0], bb[0])
                iymin = np.maximum(gtbb[:, 1], bb[1])
                ixmax = np.minimum(gtbb[:, 2], bb[2])
                iymax = np.minimum(gtbb[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) *
                       (bb[3] - bb[1] + 1.) +
                       (gtbb[:, 2] - gtbb[:, 0] + 1.) *
                       (gtbb[:, 3] - gtbb[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            else:
                continue

            if ovmax > mAp_iou_thresh:
                if not gt['det'][jmax]:
                    tp[c][d] = 1.  # tp + 1
                    gt['det'][jmax] = True  # detectioned = true
                else:
                    fp[c][d] = 1.
            else:
                fp[c][d] = 1.

    Ap = np.zeros(h.class_num)
    for c in range(h.class_num):
        fpint = np.cumsum(fp[c])
        tpint = np.cumsum(tp[c])
        recall = fpint / t_npos[c]
        precision = tpint / np.maximum(tpint + fpint,
                                       np.finfo(np.float64).eps)
        Ap[c] = voc_ap(recall, precision)

    mAp = np.mean(Ap)
    print('~~~~~~~~')
    for c in range(len(Ap)):
        print(f'AP for Class {c} =', colored(f'{Ap[c]:.4f}', 'blue'))
    print(f'mAP =', colored(f'{mAp:.4f}', 'red'))
    print('~~~~~~~~')
