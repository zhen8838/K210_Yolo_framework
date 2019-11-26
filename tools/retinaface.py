import tensorflow as tf
import numpy as np
from numpy import random
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from tools.yolo import center_to_corner, bbox_iou
from tools.base import BaseHelper
from typing import List, Iterable, Tuple


def encode(matches, anchors, variances):
    g_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 - anchors[:, :2]
    g_cxcy /= (variances[0] * anchors[:, 2:])
    g_wh = (matches[:, 2:] - matches[:, :2]) / anchors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(matched, anchors, variances):
    matched = matched.reshape((-1, 5, 2))
    anchors = np.concatenate([np.tile(anchors[:, 0:1, None], [1, 5, 1]),
                              np.tile(anchors[:, 1:2, None], [1, 5, 1]),
                              np.tile(anchors[:, 2:3, None], [1, 5, 1]),
                              np.tile(anchors[:, 3:4, None], [1, 5, 1])], 2)
    g_cxcy = matched[:, :, :2] - anchors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(-1, 5 * 2)
    # return target for smooth_l1_loss
    return g_cxcy


def bbox_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


class RetinaFaceHelper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: tuple,
                 anchor_min_size: list,
                 anchor_steps: list,
                 pos_thresh: float,
                 variances: float):
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None

        img_ann_list = np.load(image_ann, allow_pickle=True)[()]

        # NOTE can use dict set trian and test dataset
        self.train_list: Iterable[Tuple[np.ndarray, np.ndarray]] = img_ann_list['train']
        self.val_list: Iterable[Tuple[np.ndarray, np.ndarray]] = img_ann_list['val']
        self.test_list: Iterable[Tuple[np.ndarray, np.ndarray]] = img_ann_list['test']
        self.train_total_data: int = len(self.train_list)
        self.val_total_data: int = len(self.val_list)
        self.test_total_data: int = len(self.test_list)

        self.anchors = self._get_anchors(in_hw, anchor_min_size, anchor_steps)
        self.org_in_hw: np.ndarray = np.array(in_hw)
        self.in_hw = tf.Variable(self.org_in_hw, trainable=False)
        self.pos_thresh: float = pos_thresh
        self.variances: np.ndarray = variances

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

    @staticmethod
    def _get_anchors(image_size=[640, 640],
                     min_sizes=[[16, 32], [64, 128], [256, 512]],
                     steps=[8, 16, 32]) -> np.ndarray:
        """ get anchors """
        feature_maps = [[int(image_size[0] / step), int(image_size[1] / step)] for step in steps]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizess = min_sizes[k]
            feature = np.empty((f[0], f[1], len(min_sizess), 4))
            for i in range(f[0]):
                for j in range(f[1]):
                    for n, min_size in enumerate(min_sizess):
                        s_kx = min_size / image_size[1]
                        s_ky = min_size / image_size[0]
                        cx = (j + 0.5) * steps[k] / image_size[1]
                        cy = (i + 0.5) * steps[k] / image_size[0]
                        feature[i, j, n, :] = cx, cy, s_kx, s_ky
            anchors.append(feature)

        anchors = np.concatenate([
            np.reshape(anchors[0], (-1, 4)),
            np.reshape(anchors[1], (-1, 4)),
            np.reshape(anchors[2], (-1, 4))], 0)

        anchors = np.clip(anchors, 0, 1)
        return anchors

    @staticmethod
    def _crop_with_constraints(img: np.ndarray,
                               ann: List[np.ndarray],
                               in_hw: np.ndarray
                               ) -> [np.ndarray, List[np.ndarray]]:
        """ random crop with constraints

            make sure that the cropped img contains at least one face > 16 pixel at training image scale
        """
        bbox, landm, clses = ann
        im_h, im_w, _ = img.shape
        in_h, in_w = in_hw

        for _ in range(250):
            scale = random.choice([0.3, 0.45, 0.6, 0.8, 1.0])
            # short_side = min(im_w, im_h)
            new_w = int(scale * im_w)
            new_h = int(scale * im_h)

            if im_w == new_w:
                l = 0
            else:
                l = random.randint(im_w - new_w)
            if im_h == new_h:
                t = 0
            else:
                t = random.randint(im_h - new_h)
            roi = np.array((l, t, l + new_w, t + new_h))

            value = bbox_iof(bbox, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (bbox[:, :2] + bbox[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            bbox_t = bbox[mask_a].copy()
            clses_t = clses[mask_a].copy()
            landm_t = landm[mask_a].copy().reshape([-1, 5, 2])

            if bbox_t.shape[0] == 0:
                continue

            image_t = img[roi[1]:roi[3], roi[0]:roi[2]]

            bbox_t[:, :2] = np.maximum(bbox_t[:, :2], roi[:2])
            bbox_t[:, :2] -= roi[:2]
            bbox_t[:, 2:] = np.minimum(bbox_t[:, 2:], roi[2:])
            bbox_t[:, 2:] -= roi[:2]

            # landm
            landm_t[:, :, :2] = landm_t[:, :, :2] - roi[:2]
            landm_t[:, :, :2] = np.maximum(landm_t[:, :, :2], np.array([0, 0]))
            landm_t[:, :, :2] = np.minimum(landm_t[:, :, :2], roi[2:] - roi[:2])
            landm_t = landm_t.reshape([-1, 10])

            # make sure that the cropped img contains at least one face > 16 pixel at training image scale
            b_w_t = (bbox_t[:, 2] - bbox_t[:, 0] + 1) / new_w * in_w
            b_h_t = (bbox_t[:, 3] - bbox_t[:, 1] + 1) / new_h * in_h
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            bbox_t = bbox_t[mask_b]
            clses_t = clses_t[mask_b]
            landm_t = landm_t[mask_b]

            if bbox_t.shape[0] == 0:
                continue

            return image_t, [bbox_t, landm_t, clses_t]

        return img, [bbox, landm, clses]

    def read_img(self, img_path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    def resize_img(self, img: np.ndarray,
                   ann: List[np.ndarray],
                   in_hw: np.ndarray
                   ) -> [np.ndarray, List[np.ndarray]]:
        bbox, landm, clses = ann
        im_in = np.zeros((in_hw[0], in_hw[1], 3), np.uint8)

        """ transform factor """
        img_hw = np.array(img.shape[:2])
        img_wh = img_hw[::-1]
        in_wh = in_hw[::-1]
        scale = np.min(in_hw / img_hw)

        # NOTE calc the x,y offset
        y_off, x_off = ((in_hw - img_hw * scale) / 2).astype(int)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=np.random.choice(
                             [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]))

        im_in[y_off:y_off + img.shape[0],
              x_off:x_off + img.shape[1], :] = img[...]

        """ calc the point transform """
        bbox[:, 0::2] = bbox[:, 0::2] * scale + x_off
        bbox[:, 1::2] = bbox[:, 1::2] * scale + y_off
        landm[:, 0::2] = landm[:, 0::2] * scale + x_off
        landm[:, 1::2] = landm[:, 1::2] * scale + y_off

        return im_in, [bbox, landm, clses]

    def resize_train_img(self, img: np.ndarray,
                         ann: List[np.ndarray],
                         in_hw: np.ndarray
                         ) -> [np.ndarray, List[np.ndarray]]:
        img, ann = self._crop_with_constraints(img, ann, in_hw)
        return self.resize_img(img, ann, in_hw)

    def augment_img(self, img: np.ndarray,
                    ann: List[np.ndarray],
                    ) -> [np.ndarray, List[np.ndarray]]:
        bbox, landm, clses = ann
        bbs = ia.BoundingBoxesOnImage.from_xyxy_array(bbox, shape=img.shape)
        kps = ia.KeypointsOnImage.from_xy_array(landm.reshape(-1, 2), shape=img.shape)

        image_aug, bbs_aug, kps_aug = self.iaaseq(image=img, bounding_boxes=bbs,
                                                  keypoints=kps)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        new_bbox = bbs_aug.to_xyxy_array()
        new_landm = np.reshape(kps_aug.to_xy_array(), (-1, 5, 2))

        return image_aug, new_bbox, new_landm, clses

    @staticmethod
    def match(ann: List[np.ndarray],
              anchors: np.ndarray, pos_thresh: float, variances: list
              ) -> List[np.ndarray]:
        bbox, clses, landms = ann
        overlaps = bbox_iou(bbox, center_to_corner(anchors, False))
        best_prior_overlap = np.max(overlaps, 1)
        best_prior_idx = np.argmax(overlaps, 1)

        valid_gt_idx = best_prior_overlap >= 0.2  # 有效的gt
        best_prior_idx_filter = best_prior_idx[valid_gt_idx]
        if len(best_prior_idx_filter) <= 0:
            return np.zeros((len(anchors), 4)), np.zeros((len(anchors), 4)), np.zeros((len(anchors), 10))

        best_truth_overlap = np.max(overlaps, 0)
        best_truth_idx = np.argmax(overlaps, 0)

        best_truth_overlap[best_prior_idx_filter] = 2

        matches = bbox[best_truth_idx]
        conf = clses[best_truth_idx]
        conf[best_truth_overlap < pos_thresh] = 0    # label as background
        loc = encode(matches, anchors, variances)

        matches_landm = landms[best_truth_idx]
        matches_landm.shape

        landm = encode_landm(matches_landm, anchors, variances)
        return loc, landm, conf

    def process_img(self, img: np.ndarray, ann: np.ndarray, in_hw: np.ndarray,
                    is_augment: bool, is_resize: bool, is_normlize: bool
                    ) -> [np.ndarray, List[np.ndarray]]:
        bbox, landm, clses = np.split(ann, [4, -1], 1)
        if is_resize and is_augment:
            img, ann = self.resize_train_img(img, [bbox, landm, clses], in_hw)
        elif is_resize:
            img, ann = self.resize_img(img, ann, in_hw)
        if is_augment:
            img, ann = self.augment_img(img, ann)
        if is_normlize:
            img = self.normlize_img(img)
        return img, ann

    def ann_to_label(self, ann: List[np.ndarray]) -> List[np.ndarray]:
        label_loc, label_landm, label_conf = self.match(ann,
                                                        self.anchors, self.pos_thresh,
                                                        self.variances)
        return label_loc, label_landm, label_conf

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:
        pass
