import tensorflow as tf
import numpy as np
from numpy import random
import cv2
from tools.yolo import center_to_corner, bbox_iou
from tools.base import BaseHelper


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

        img_ann_list = np.load(image_ann, allow_pickle=True)

        if isinstance(img_ann_list[()], dict):
            # NOTE can use dict set trian and test dataset
            self.train_list = img_ann_list[()]['train']  # type:np.ndarray
            self.val_list = img_ann_list[()]['val']  # type:np.ndarray
            self.test_list = img_ann_list[()]['test']  # type:np.ndarray
        elif isinstance(img_ann_list[()], np.ndarray):
            self.train_list, self.val_list, self.test_list = np.split(
                img_ann_list,
                [int((1 - self.validation_split) * len(img_ann_list)),
                    int((1 - self.validation_split / 2) * len(img_ann_list))])
        else:
            raise ValueError(f'{image_ann} data format error!')
        self.train_total_data = len(self.train_list)
        self.val_total_data = len(self.val_list)
        self.test_total_data = len(self.test_list)

        self.anchors = self._get_anchors(in_hw, anchor_min_size, anchor_steps)
        self.in_hw: np.ndarray = in_hw
        self.pos_thresh: float = pos_thresh
        self.variances: np.ndarray = variances

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
    def _crop_with_constraints(img, bbox, landm, clses, in_hw):
        """ random crop with constraints

            make sure that the cropped img contains at least one face > 16 pixel at training image scale
        """
        height, width, _ = img.shape
        in_h, in_w = in_hw

        for _ in range(250):
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            # short_side = min(width, height)
            w = int(scale * width)
            h = int(scale * height)

            if width == w:
                l = 0
            else:
                l = random.randint(width - w)
            if height == h:
                t = 0
            else:
                t = random.randint(height - h)
            roi = np.array((l, t, l + w, t + h))

            value = bbox_iof(bbox, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (bbox[:, :2] + bbox[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            bbox_t = bbox[mask_a].copy()
            clses_t = clses[mask_a].copy()
            landm_t = landm[mask_a].copy()
            landm_t = landm_t.reshape([-1, 5, 2])

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
            b_w_t = (bbox_t[:, 2] - bbox_t[:, 0] + 1) / w * in_w
            b_h_t = (bbox_t[:, 3] - bbox_t[:, 1] + 1) / h * in_h
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            bbox_t = bbox_t[mask_b]
            clses_t = clses_t[mask_b]
            landm_t = landm_t[mask_b]

            if bbox_t.shape[0] == 0:
                continue

            return image_t, bbox_t, landm_t, clses_t

        return img, bbox, landm, clses

    def reszie_img(self, img, bbox, landm, clses, in_hw):
        im_in = np.zeros((in_hw[0], in_hw[1], 3), np.uint8)

        """ transform factor """
        img_hw = np.array(img.shape[:2])
        img_wh = img_hw[::-1]
        in_wh = in_hw[::-1]
        scale = np.min(in_hw / img_hw)

        # NOTE hw_off is [h offset,w offset]
        hw_off = ((in_hw - img_hw * scale) / 2).astype(int)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=np.random.choice(
                             [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]))

        im_in[hw_off[0]:hw_off[0] + img.shape[0],
              hw_off[1]:hw_off[1] + img.shape[1], :] = img[...]

        """ calc the point transform """
        bbox[:, 0::2] = bbox[:, 0::2] * scale + hw_off[1]
        bbox[:, 1::2] = bbox[:, 1::2] * scale + hw_off[0]
        landm[:, 0::2] = landm[:, 0::2] * scale + hw_off[1]
        landm[:, 1::2] = landm[:, 1::2] * scale + hw_off[0]

        return im_in, bbox, landm, clses

    def resize_train_img(self, img, bbox, landm, clses, in_hw):
        img, bbox, landm, clses = self._crop_with_constraints(img, bbox, landm, clses, in_hw)
        return self.reszie_img(img, bbox, landm, clses)

    @staticmethod
    def match(bbox, clses, landms, anchors, pos_thresh, variances):
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
                    is_augment: bool, is_resize: bool, is_normlize: bool):
        bbox, landm, clses = np.split(ann, [4, -1], 1)
        if is_resize and is_augment:
            img, bbox, landm, clses = self.resize_train_img(img, bbox, landm, clses, in_hw)
        elif is_resize:
            img, bbox, landm, clses = self.resize_img(img, bbox, landm, clses, in_hw)
        if is_augment:
            img, ann = self.augment_img(img, ann)
        if is_normlize:
            img = self.normlize_img(img)
        return img, ann

    def ann_to_train_label(self, ann: np.ndarray):
        bbox = ann[:, :4]
        landms = ann[:, 4:14]
        clses = ann[:, -1]
        label_loc, label_landm, label_conf = self.match(bbox, clses, landms,
                                                        self.anchors, self.pos_thresh,
                                                        self.variances)
        return label_loc, label_conf, label_landm

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:
        pass
