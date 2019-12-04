import numpy as np
import os
import cv2
from matplotlib.pyplot import imshow, show
from math import cos, sin
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras as k
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from matplotlib.pyplot import text
from PIL import Image, ImageFont, ImageDraw
from tools.base import BaseHelper, INFO, ERROR, NOTE
from tools.bbox_utils import bbox_iou, center_to_corner
from pathlib import Path
import shutil
from tqdm import trange
from termcolor import colored
from typing import List, Tuple, AnyStr, Iterable


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


def bbox_crop(ann: np.ndarray, crop_box=None, allow_outside_center=True) -> np.ndarray:
    """ Crop bounding boxes according to slice area.

    Parameters
    ----------
    ann : np.ndarray

        (n,5) [p,x1,y1,x2,y1]

    crop_box : optional

        crop_box [x1,y1,x2,y2] , by default None

    allow_outside_center : bool, optional

        by default True

    Returns
    -------
    np.ndarray

        ann
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    ann = ann.copy()
    if crop_box is None:
        return ann
    if sum([int(c is None) for c in crop_box]) == 4:
        return ann

    l, t, r, b = crop_box

    left = l if l else 0
    top = t if t else 0
    right = r if r else np.inf
    bottom = b if b else np.inf
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(ann.shape[0], dtype=bool)
    else:
        centers = (ann[:, 1:3] + ann[:, 3:5]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    ann[:, 1:3] = np.maximum(ann[:, 1:3], crop_bbox[:2])
    ann[:, 3:5] = np.minimum(ann[:, 3:5], crop_bbox[2:4])
    ann[:, 1:3] -= crop_bbox[:2]
    ann[:, 3:5] -= crop_bbox[:2]

    mask = np.logical_and(mask, (ann[:, 1:3] < ann[:, 3:5]).all(axis=1))
    ann = ann[mask]
    return ann


def random_crop_with_constraints(ann: np.ndarray, im_wh: np.ndarray,
                                 min_scale: float = 0.3, max_scale: float = 1,
                                 max_aspect_ratio: float = 2,
                                 constraints: float = None,
                                 max_trial: float = 50) -> [np.ndarray, list]:
    """
        Crop an image randomly with bounding box constraints.

        [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Parameters
    ----------
    ann : np.ndarray

        (n,5) [p,x1,y1,x2,y1]

    im_wh : np.ndarray

        im wh

    min_scale : float, optional

        The minimum ratio between a cropped region and the original image. by default 0.3

    max_scale : float, optional

        The maximum ratio between a cropped region and the original image. by default 1

    max_aspect_ratio : float, optional

        The maximum aspect ratio of cropped region. by default 2

    constraints : float, optional

        by default None

    max_trial : float, optional

        Maximum number of trials for each constraint before exit no matter what. by default 50

    Returns
    -------
    [np.ndarray, list]

        new ann, crop idx : (0, 0, w, h)

    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = im_wh

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = np.random.uniform(min_scale, max_scale)
            aspect_ratio = np.random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = np.random.randint(h - crop_h)
            crop_l = np.random.randint(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(ann) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return ann, (left, top, right, bottom)

            iou = bbox_iou(ann[:, 1:], crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right, bottom))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_ann = bbox_crop(ann, crop, allow_outside_center=False)
        if new_ann.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_ann, new_crop
    return ann, (0, 0, w, h)


class YOLOHelper(BaseHelper):
    def __init__(self, image_ann: str, class_num: int, anchors: str,
                 in_hw: tuple, out_hw: tuple, validation_split: float = 0.1):
        """ yolo helper

        Parameters
        ----------
        image_ann : str

            image ann `.npy` file path

        class_num : int

        anchors : str

            anchor `.npy` file path

        in_hw : tuple

            default input image height width

        out_hw : tuple

            default output height width

        validation_split : float, optional

            validation split, by default 0.1

        """
        super().__init__(image_ann, validation_split)
        self.org_in_hw = np.array(in_hw)
        self.org_out_hw = np.array(out_hw)
        assert self.org_in_hw.ndim == 1
        assert self.org_out_hw.ndim == 2
        self.in_hw = tf.Variable(self.org_in_hw, trainable=False)
        self.out_hw = tf.Variable(self.org_out_hw, trainable=False)
        if class_num:
            self.class_num = class_num  # type:int
        if anchors:
            self.anchors = np.load(anchors)  # type:np.ndarray
            self.anchor_number = len(self.anchors[0])
            self.output_number = len(self.anchors)
            self.__flatten_anchors = np.reshape(self.anchors, (-1, 2))
            self.grid_wh = (1 / self.org_out_hw)[:, [1, 0]]  # hw => wh
            self.xy_offset = coordinate_offset(self.anchors, self.org_out_hw)  # type:np.ndarray

        self.iaaseq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.SomeOf([1, 4], [
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Affine(rotate=(-30, 30),
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))
            ], True)
        ])  # type: iaa.meta.Sequential

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
        return (xy * self.org_out_hw[layer][:: -1]) - self.xy_offset[layer]

    def _xy_grid_index(self, out_hw: np.ndarray, box_xy: np.ndarray, layer: int) -> [np.ndarray, np.ndarray]:
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
        return np.floor(box_xy * out_hw[layer][:: -1]).astype('int')

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
            layer_idx shape = [num_box, num_anchor]
            anchor_idx shape = [num_box, num_anchor]
        """
        iou = fake_iou(np.expand_dims(wh, -2), self.__flatten_anchors)
        # sort iou score in decreasing order
        best_anchor = np.argsort(-iou, -1)  # shape = [num_box, num_anchor]
        return np.divmod(best_anchor, self.anchor_number)

    @staticmethod
    def corner_to_center(ann: np.ndarray, in_hw: np.ndarray) -> np.ndarray:
        """ conrner ann to center ann
            xyxy ann to xywh ann

        Parameters
        ----------
        ann : np.ndarray

            xyxyann n*[p,x1,y1,x2,y2]
            NOTE all pixel scale

        in_hw : np.ndarray

        Returns
        -------
        np.ndarray

            xywhann n*[p,x,y,w,h]

            NOTE scale = [0~1] x,y is center point
        """
        return np.hstack([
            ann[:, 0:1],
            ((ann[:, 1:2] + ann[:, 3:4]) / 2) / in_hw[1],
            ((ann[:, 2:3] + ann[:, 4:5]) / 2) / in_hw[0],
            (ann[:, 3:4] - ann[:, 1:2]) / in_hw[1],
            (ann[:, 4:5] - ann[:, 2:3]) / in_hw[0]])

    @staticmethod
    def center_to_corner(ann: np.ndarray, in_hw: np.ndarray) -> np.ndarray:
        """ center ann to conrner ann
            xywh ann to xyxy ann

        Parameters
        ----------
        ann : np.ndarray

            xywhann n*[p,x,y,w,h]
            NOTE scale = [0~1] x,y is center point

        in_hw : np.ndarray

        Returns
        -------
        np.ndarray

            xyxyann n*[p,x1,y1,x2,y2]
            NOTE all pixel scale

        """
        return np.hstack([
            ann[:, 0:1],
            (ann[:, 1:2] - ann[:, 3:4] / 2) * in_hw[1],
            (ann[:, 2:3] - ann[:, 4:5] / 2) * in_hw[0],
            (ann[:, 1:2] + ann[:, 3:4] / 2) * in_hw[1],
            (ann[:, 2:3] + ann[:, 4:5] / 2) * in_hw[0]])

    def ann_to_label(self, in_hw: np.ndarray, out_hw: np.ndarray, ann: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 label~

        Parameters
        ----------
        ann : np.ndarray
            annotation shape :[n,5] value : n*[p,x1,y1,x2,y2]

        Returns
        -------
        tuple
            labels list value :[output_number*[ out_h, out_w, anchor_num, class +5 +1 ]]

            NOTE The last element of the last dimension is the allocation flag,
             which means that there is ground truth at this position.
            Can use only one output label find all ground truth~
            ```python
            new_anns = []
            for i in range(h.output_number):
                new_ann = labels[i][np.where(labels[i][..., -1] == 1)]
                new_anns.append(np.c_[np.argmax(new_ann[:, 5:], axis=-1), new_ann[:, :4]])
            np.allclose(new_anns[0], new_anns[1])  # True
            np.allclose(new_anns[1], new_anns[2])  # True
            ```

        """
        labels = [np.zeros((out_hw[i][0], out_hw[i][1], len(self.anchors[i]),
                            5 + self.class_num + 1), dtype='float32') for i in range(self.output_number)]

        ann = self.corner_to_center(ann, in_hw)
        layer_idx, anchor_idx = self._get_anchor_index(ann[:, 3: 5])
        for box, l, n in zip(ann, layer_idx, anchor_idx):
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            # clip box avoid width or heigh == 0 ====> loss = inf
            bb = np.clip(box[1: 5], 1e-8, 0.99999999)
            cnt = np.zeros(self.output_number, np.bool)  # assigned flag
            for i in range(len(l)):
                x, y = self._xy_grid_index(out_hw, bb[0: 2], l[i])  # [x index , y index]
                if cnt[l[i]] or labels[l[i]][y, x, n[i], 4] == 1.:
                    # 1. when this output layer already have ground truth, skip
                    # 2. when this grid already being assigned, skip
                    continue
                labels[l[i]][y, x, n[i], 0: 4] = bb
                labels[l[i]][y, x, n[i], 4] = (0. if cnt.any() else 1.)
                labels[l[i]][y, x, n[i], 5 + int(box[0])] = 1.
                labels[l[i]][y, x, n[i], -1] = 1.  # set gt flag = 1
                cnt[l[i]] = True  # output layer ground truth flag
                if cnt.all():
                    # when all output layer have ground truth, exit
                    break
        return labels

    def _xy_to_all(self, labels: tuple):
        """convert xy scale from grid to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 0: 2] = labels[i][..., 0: 2] * self.grid_wh[i] + self.xy_offset[i]

    def _wh_to_all(self, labels: tuple):
        """convert wh scale to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 2: 4] = np.exp(labels[i][..., 2: 4]) * self.anchors[i]

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
        new_ann = self.center_to_corner(new_ann, self.org_in_hw)
        return new_ann

    def augment_img(self, img: np.ndarray, ann: np.ndarray) -> tuple:
        """ augmenter for image

        Parameters
        ----------
        img : np.ndarray

            img src

        ann : np.ndarray

            one annotation
            [p,x1,y1,x2,y2]

        Returns
        -------
        tuple

            [image src,ann] after data augmenter
            image src dtype is uint8
        """
        p = ann[:, 0: 1]
        box = ann[:, 1:]

        bbs = BoundingBoxesOnImage.from_xyxy_array(box, shape=img.shape)

        image_aug, bbs_aug = self.iaaseq(image=img, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        new_box = bbs_aug.to_xyxy_array()
        new_ann = np.hstack((p[0: len(new_box), :], new_box))

        return image_aug, new_ann

    def resize_train_img(self, img: np.ndarray, in_hw: np.ndarray, ann: np.ndarray
                         ) -> [np.ndarray, np.ndarray]:
        """ when training first crop image and resize image and keep ratio

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

        Returns
        -------
        [np.ndarray, np.ndarray]
            img, ann
        """
        if np.random.uniform(0, 1) > 0.5:
            ann, (x1, y1, x2, y2) = random_crop_with_constraints(ann, img.shape[1::-1])
            img = img[y1:y2, x1:x2, :]

        return self.resize_img(img, in_hw, ann)

    def resize_img(self, img: np.ndarray, in_hw: np.ndarray,
                   ann: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        resize image and keep ratio

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray


        Returns
        -------
        [np.ndarray, np.ndarray]
            img, ann [ uint8 , float64 ]
        """
        im_in = np.zeros((in_hw[0], in_hw[1], 3), np.uint8)

        """ transform factor """
        img_hw = np.array(img.shape[:2])
        scale = np.min(in_hw / img_hw)

        # NOTE xy_off is [x offset,y offset]
        x_off, y_off = ((in_hw[::-1] - img_hw[::-1] * scale) / 2).astype(int)
        img = cv2.resize(img, None, fx=scale, fy=scale)

        im_in[y_off:y_off + img.shape[0],
              x_off:x_off + img.shape[1], :] = img[...]

        """ calculate the box transform matrix """
        if isinstance(ann, np.ndarray):
            ann[:, 1:3] = ann[:, 1:3] * scale + [x_off, y_off]
            ann[:, 3:5] = ann[:, 3:5] * scale + [x_off, y_off]

        return im_in, ann

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
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    def process_img(self, img: np.ndarray, ann: np.ndarray, in_hw: np.ndarray,
                    is_augment: bool, is_resize: bool,
                    is_normlize: bool) -> [tf.Tensor, tf.Tensor]:
        """ process image and true box , if is training then use data augmenter

        Parameters
        ----------
        img : np.ndarray
            image srs
        ann : np.ndarray
            one annotation
        is_augment : bool
            wether to use data augmenter
        is_resize : bool
            wether to resize the image
        is_normlize : bool
            wether to normlize the image

        Returns
        -------
        tuple
            image src , true box
        """
        if is_resize and is_augment:
            img, ann = self.resize_train_img(img, in_hw, ann)
        elif is_resize:
            img, ann = self.resize_img(img, in_hw, ann)
        if is_augment:
            img, ann = self.augment_img(img, ann)
        if is_normlize:
            img = self.normlize_img(img)
        return img, ann

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))

        def _parser_wrapper(i: tf.Tensor):
            # NOTE use wrapper function and dynamic list construct (x,(y_1,y_2,...))
            img_path, ann = tf.numpy_function(lambda idx:
                                              (image_ann_list[idx][0].copy(),
                                               image_ann_list[idx][1].copy()),
                                              [i], [tf.dtypes.string, tf.float64])
            # tf.numpy_function(lambda x: print('img id:', x), [i],[])
            # load image
            raw_img = tf.image.decode_jpeg(tf.io.read_file(img_path),
                                           channels=3)
            # resize image -> image augmenter
            raw_img, ann = tf.numpy_function(self.process_img,
                                             [raw_img, ann, self.in_hw,
                                              is_augment, True, False],
                                             [tf.uint8, tf.float64],
                                             name='process_img')
            # make labels
            labels = tf.numpy_function(self.ann_to_label, [self.in_hw, self.out_hw, ann],
                                       [tf.float32] * len(self.anchors),
                                       name='ann_to_label')

            # normlize image
            if is_normlize:
                img = self.normlize_img(raw_img)
            else:
                img = tf.cast(raw_img, tf.float32)

            # set shape
            img.set_shape((None, None, 3))
            for i in range(len(self.anchors)):
                labels[i].set_shape((None, None, len(self.anchors[i]), self.class_num + 5 + 1))
            return img, tuple(labels)

        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                       shuffle(batch_size * 500).
                       repeat().
                       map(_parser_wrapper, -1).
                       batch(batch_size, True).
                       prefetch(-1))
        else:
            dataset = (tf.data.Dataset.from_tensor_slices(
                tf.range(len(image_ann_list))).
                map(_parser_wrapper, -1).
                batch(batch_size, True).
                prefetch(-1))

        return dataset

    def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True, scores=None):
        """ draw img and show bbox , set ann = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

            scale is all image pixal scale
            shape : [p,x1,y1,x2,y2]

        is_show : bool

            show image
        """
        if isinstance(ann, np.ndarray):
            p = ann[:, 0]
            xyxybox = ann[:, 1:]
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


class MultiScaleTrain(Callback):
    def __init__(self, h: YOLOHelper, interval: int = 10, scale_range: list = [-3, 3]):
        """ Multi-scale training callback

            NOTE This implementation will lead to the lack of multi-scale
                 training in several batches after the end of validation

        Parameters
        ----------
        h : YOLOHelper

        interval : int, optional

            change scale batch interval, by default 10

        scale_range : list, optional

            change scale range, by default [-3, 3]
            eg.
            ```
                org_input_size = 416
                x = 2 # in range(-3,3)
                input_size = org_input_size + (x * 32)
                           = 416 + (2 * 32)
                           = 480
            ```
        """
        super().__init__()
        self.h = h
        self.interval = interval
        self.scale_range = np.arange(scale_range[0], scale_range[1])
        self.cur_scl = scale_range[1]  # default max size
        self.flag = True  # change flag
        self.count = 1

    def on_train_begin(self, logs=None):
        K.set_value(self.h.in_hw, self.h.org_in_hw + 32 * self.cur_scl)
        K.set_value(self.h.out_hw, self.h.org_out_hw + np.power(2, np.arange(self.h.output_number))[:, None] * self.cur_scl)
        print(f'\n {NOTE} : Train input image size : [{self.h.in_hw[0]},{self.h.in_hw[1]}]')

    def on_epoch_begin(self, epoch, logs=None):
        self.flag = True

    def on_train_batch_end(self, batch, logs=None):
        if self.flag == True:
            if self.count % self.interval == 0:
                # random choice resize scale
                self.cur_scl = np.random.choice(self.scale_range)
                K.set_value(self.h.in_hw, self.h.org_in_hw + 32 * self.cur_scl)
                K.set_value(self.h.out_hw, self.h.org_out_hw + np.power(2, np.arange(self.h.output_number))[:, None] * self.cur_scl)
                self.count = 1
                print(f'\n {NOTE} : Train input image size : [{self.h.in_hw[0]},{self.h.in_hw[1]}]')
            else:
                self.count += 1

    def on_test_begin(self, logs=None):
        """ change to orginal image size """
        if self.flag == True:
            self.flag = False
            K.set_value(self.h.in_hw, self.h.org_in_hw)
            K.set_value(self.h.out_hw, self.h.org_out_hw)


class YOLOLoss(Loss):
    def __init__(self, h: YOLOHelper, iou_thresh: float,
                 obj_weight: float, noobj_weight: float, wh_weight: float,
                 xy_weight: float, cls_weight: float, layer: int, verbose=1,
                 reduction=losses_utils.ReductionV2.AUTO, name=None):
        """ yolo loss obj

        Parameters
        ----------
        h : YOLOHelper

        iou_thresh : float

        obj_weight : float

        noobj_weight : float

        wh_weight : float

        layer : int
            the current layer index

        """
        super().__init__(reduction=reduction, name=name)
        self.h = h
        self.iou_thresh = iou_thresh
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.wh_weight = wh_weight
        self.xy_weight = xy_weight
        self.cls_weight = cls_weight
        self.layer = layer
        self.anchors = np.copy(self.h.anchors[self.layer])  # type:np.ndarray
        self.verbose = verbose
        self.op_list = []
        with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            self.r50: ResourceVariable = tf.compat.v1.get_variable(
                'r50', (), tf.float32,
                tf.zeros_initializer(),
                trainable=False)

            self.r75: ResourceVariable = tf.compat.v1.get_variable(
                'r75', (), tf.float32,
                tf.zeros_initializer(),
                trainable=False)

            names = ['r50', 'r75']
            self.lookups: Iterable[Tuple[ResourceVariable, AnyStr]] = [
                (tf.compat.v1.get_variable(name, (), tf.float32,
                                           tf.zeros_initializer(),
                                           trainable=False),
                    name)
                for name in names]

        if self.verbose == 2:
            with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                             reuse=tf.compat.v1.AUTO_REUSE):
                names = ['xy', 'wh', 'obj', 'noobj', 'cls']
                self.lookups.extend([
                    (tf.compat.v1.get_variable(name, (), tf.float32,
                                               tf.zeros_initializer(),
                                               trainable=False),
                     name)
                    for name in names])

    def calc_xy_offset(self, out_hw: tf.Tensor, feature: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """ for dynamic sacle get xy offset tensor for loss calc

        Parameters
        ----------
        feature : tf.Tensor

            featrue tensor, shape = [batch,out h,out w,anchor num,class num+5]

        Returns
        -------

        [tf.Tensor, tf.Tensor]

            out hw    : shape []
            xy offset : shape [out h , out w , 1 , 2]

        """
        grid_y = tf.tile(tf.reshape(tf.range(0, out_hw[0]),
                                    [-1, 1, 1, 1]), [1, out_hw[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(0, out_hw[1]),
                                    [1, -1, 1, 1]), [out_hw[0], 1, 1, 1])
        xy_offset = tf.concat([grid_x, grid_y], -1)
        return xy_offset

    @staticmethod
    def xywh_to_grid(all_true_xy: tf.Tensor, all_true_wh: tf.Tensor,
                     out_hw: tf.Tensor, xy_offset: tf.Tensor,
                     anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """convert true label xy wh to grid scale

        Returns
        -------
        [tf.Tensor, tf.Tensor]

            grid_true_xy, grid_true_wh shape = [out h ,out w,anchor num , 2 ]

        """
        grid_true_xy = (all_true_xy * out_hw[::-1]) - xy_offset
        grid_true_wh = tf.math.log(all_true_wh / anchors)
        return grid_true_xy, grid_true_wh

    @staticmethod
    def xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor,
                    out_hw: tf.Tensor, xy_offset: tf.Tensor,
                    anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

        Returns
        -------
        [tf.Tensor, tf.Tensor]

            [all_pred_xy, all_pred_wh]
        """
        all_pred_xy = (tf.sigmoid(grid_pred_xy) + xy_offset) / out_hw[::-1]
        all_pred_wh = tf.exp(grid_pred_wh) * anchors
        return all_pred_xy, all_pred_wh

    @staticmethod
    def iou(pred_xy: tf.Tensor, pred_wh: tf.Tensor,
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

    def smoothl1loss(self, labels: tf.Tensor, predictions: tf.Tensor, delta=1.0):
        error = tf.math.subtract(predictions, labels)
        abs_error = tf.math.abs(error)
        quadratic = tf.math.minimum(abs_error, delta)
        # The following expression is the same in value as
        # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
        # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
        # This is necessary to avoid doubling the gradient, since there is already a
        # nonzero contribution to the gradient from the quadratic term.
        linear = tf.math.subtract(abs_error, quadratic)
        return tf.math.add(tf.math.multiply(0.5, tf.math.multiply(quadratic, quadratic)),
                           tf.math.multiply(delta, linear))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
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
        xy_offset = self.calc_xy_offset(out_hw, y_pred)

        pred_xy, pred_wh = self.xywh_to_all(grid_pred_xy, grid_pred_wh,
                                            out_hw, xy_offset, self.anchors)

        obj_cnt = tf.reduce_sum(obj_mask)

        def lmba(bc):
            # bc=1
            # NOTE use location_mask find all ground truth
            gt_xy = tf.boolean_mask(all_true_xy[bc], location_mask[bc])
            gt_wh = tf.boolean_mask(all_true_wh[bc], location_mask[bc])
            iou_score = self.iou(pred_xy[bc], pred_wh[bc], gt_xy, gt_wh)  # [h,w,anchor,box_num]
            # NOTE this layer gt and pred iou score
            mask_iou_score = tf.reduce_max(tf.boolean_mask(iou_score, obj_mask_bool[bc]), -1)

            with tf.control_dependencies(
                    [self.r50.assign_add(tf.reduce_sum(tf.cast(mask_iou_score > .5, tf.float32))),
                     self.r75.assign_add(tf.reduce_sum(tf.cast(mask_iou_score > .75, tf.float32)))]):
                # if iou for any ground truth larger than iou_thresh, the pred is true.
                match_num = tf.reduce_sum(tf.cast(iou_score > self.iou_thresh, tf.float32),
                                          -1, keepdims=True)
            return tf.cast(match_num < 1, tf.float32)

        ignore_mask = tf.map_fn(lmba, tf.range(self.h.batch_size), dtype=tf.float32)

        """ calc the loss dynamic weight """
        grid_true_xy, grid_true_wh = self.xywh_to_grid(all_true_xy, all_true_wh,
                                                       out_hw, xy_offset, self.anchors)
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

        """ sum loss """
        self.op_list.extend([
            self.lookups[0][0].assign(tf.math.divide_no_nan(self.r50, obj_cnt)),
            self.lookups[1][0].assign(tf.math.divide_no_nan(self.r75, obj_cnt)),
            self.r50.assign(0),
            self.r75.assign(0)
        ])
        if self.verbose == 2:
            self.op_list.extend([self.lookups[2][0].assign(tf.reduce_mean(xy_loss)),
                                 self.lookups[3][0].assign(tf.reduce_mean(wh_loss)),
                                 self.lookups[4][0].assign(tf.reduce_mean(obj_loss)),
                                 self.lookups[5][0].assign(tf.reduce_mean(noobj_loss)),
                                 self.lookups[6][0].assign(tf.reduce_mean(cls_loss))])

        with tf.control_dependencies(self.op_list):
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
    box_yx = box_xy[..., :: -1]
    box_hw = box_wh[..., :: -1]
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
    y_pred = [np.reshape(pred, list(pred.shape[:-1]) + [h.anchor_number,
                                                        5 + h.class_num])
              for pred in y_pred]
    """ box list """
    _yxyx_box = []
    _yxyx_box_scores = []
    """ preprocess label """
    for l, pred_label in enumerate(y_pred):
        """ split the label """
        pred_xy = pred_label[..., 0: 2]
        pred_wh = pred_label[..., 2: 4]
        pred_confidence = pred_label[..., 4: 5]
        pred_cls = pred_label[..., 5:]
        if h.class_num > 1:
            box_scores = tf.sigmoid(pred_cls) * tf.sigmoid(pred_confidence)
        else:
            box_scores = tf.sigmoid(pred_confidence)

        """ reshape box  """
        # NOTE tf_xywh_to_all will auto use sigmoid function
        pred_xy_A, pred_wh_A = YOLOLoss.xywh_to_all(pred_xy, pred_wh, h.org_out_hw[l],
                                                    h.xy_offset[l], h.anchors[l])
        boxes = correct_box(pred_xy_A, pred_wh_A, h.org_in_hw, img_hw)
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
    img_hw = orig_img.shape[0: 2]
    img, _ = h.process_img(orig_img, None, h.org_in_hw, False, True, True)
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

    i = np.where(mrec[1:] != mrec[: -1])[0]
    # and sum (\Delta recall) * prec
    Ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return Ap


def yolo_eval(infer_model: k.Model, h: YOLOHelper, det_obj_thresh: float,
              det_iou_thresh: float, mAp_iou_thresh: float, class_name: list,
              save_result: bool = False, save_result_dir: str = 'tmp'):
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

    save_result : bool

        when save result, while save `tmp/detection-results` and `tmp/ground-truth`.

    save_result_dir : str

        default `tmp`
    """
    p_img_ids = [[] for i in range(h.class_num)]
    p_scores = [[] for i in range(h.class_num)]
    p_bboxes = [[] for i in range(h.class_num)]
    t_res = [{} for i in range(h.class_num)]  # type:list[dict]
    class_name = np.array(class_name)
    t_npos = np.zeros((h.class_num, 1))
    if save_result == True:
        res_path = Path(save_result_dir + '/detection-results')
        gt_path = Path(save_result_dir + '/ground-truth')
        if gt_path.exists():
            shutil.rmtree(str(gt_path))
        if res_path.exists():
            shutil.rmtree(str(res_path))
        gt_path.mkdir(parents=True)
        res_path.mkdir(parents=True)

    for i in trange(len(h.test_list)):
        img_path, true_ann, img_hw = h.test_list[i]
        img_name = Path(img_path).stem
        raw_img = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        img, _ = h.process_img(raw_img.numpy(), None, h.org_in_hw, False, True, True)
        img = img[tf.newaxis, ...]

        p_yxyx, p_clas, p_score = yolo_parser_one(img, img_hw,
                                                  infer_model, det_obj_thresh,
                                                  det_iou_thresh, h)
        if save_result == True:
            p_xyxy = np.concatenate([np.maximum(p_yxyx[:, 1:2], 0),
                                     np.maximum(p_yxyx[:, 0:1], 0),
                                     np.minimum(p_yxyx[:, 3:4], img_hw[1]),
                                     np.minimum(p_yxyx[:, 2:3], img_hw[0])], -1)

            p_s = p_score[:, None].astype('<U7')
            p_c = class_name[p_clas[:, None].astype(np.int)]
            p_x = p_xyxy.astype(np.int32).astype('<U6')
            res_arr = np.concatenate([p_c, p_s, p_x], -1)
            np.savetxt(str(res_path / f'{img_name}.txt'), res_arr, fmt='%s')

            true_clas, true_xyxy = np.split(true_ann, [1], -1)
            t_c = class_name[true_clas.astype(np.int)]
            t_x = true_xyxy.astype(np.int32).astype('<U6')
            t_arr = np.concatenate([t_c, t_x], -1)
            np.savetxt(str(gt_path / f'{img_name}.txt'), t_arr, fmt='%s')

        else:
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

    if save_result == False:
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
        for c, name in enumerate(class_name):
            print(f'AP for {name} =', colored(f'{Ap[c]:.4f}', 'blue'))
        print(f'mAP =', colored(f'{mAp:.4f}', 'red'))
        print('~~~~~~~~')
