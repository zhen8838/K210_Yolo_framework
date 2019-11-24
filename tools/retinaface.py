import tensorflow as tf
import numpy as np
from numpy import random
import cv2
from tools.yolo import center_to_corner, bbox_iou


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


class RetinaFaceHelper():
    def __init__(self, image_ann: str, in_hw: tuple,
                 anchor_min_size: list,
                 anchor_steps: list):
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None

        img_ann_list = np.load(image_ann, allow_pickle=True)

        if isinstance(img_ann_list[()], dict):
            # NOTE can use dict set trian and test dataset
            self.train_list = img_ann_list[()]['train_data']  # type:np.ndarray
            self.val_list = img_ann_list[()]['val_data']  # type:np.ndarray
            self.test_list = img_ann_list[()]['test_data']  # type:np.ndarray
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
    def _crop(image, boxes, labels, landm, img_dim):
        height, width, _ = image.shape
        pad_image_flag = True

        for _ in range(250):
            """
            if random.uniform(0, 1) <= 0.2:
                scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
            """
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                l = 0
            else:
                l = random.randint(width - w)
            if height == h:
                t = 0
            else:
                t = random.randint(height - h)
            roi = np.array((l, t, l + w, t + h))

            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()
            landms_t = landm[mask_a].copy()
            landms_t = landms_t.reshape([-1, 5, 2])

            if boxes_t.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            # landm
            landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
            landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
            landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
            landms_t = landms_t.reshape([-1, 10])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]
            landms_t = landms_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            pad_image_flag = False

            return image_t, boxes_t, labels_t, landms_t, pad_image_flag
        return image, boxes, labels, landm, pad_image_flag

    @staticmethod
    def _distort(image):

        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        if random.randint(2):

            # brightness distortion
            if random.randint(2):
                _convert(image, beta=random.uniform(-32, 32))

            # contrast distortion
            if random.randint(2):
                _convert(image, alpha=random.uniform(0.5, 1.5))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # saturation distortion
            if random.randint(2):
                _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randint(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:

            # brightness distortion
            if random.randint(2):
                _convert(image, beta=random.uniform(-32, 32))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # saturation distortion
            if random.randint(2):
                _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randint(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # contrast distortion
            if random.randint(2):
                _convert(image, alpha=random.uniform(0.5, 1.5))

        return image

    @staticmethod
    def _expand(image, boxes, fill, p):
        if random.randint(2):
            return image, boxes

        height, width, depth = image.shape

        scale = random.uniform(1, p)
        w = int(scale * width)
        h = int(scale * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)
        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t

    @staticmethod
    def _mirror(image, boxes, landms):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

            # landm
            landms = landms.copy()
            landms = landms.reshape([-1, 5, 2])
            landms[:, :, 0] = width - landms[:, :, 0]
            tmp = landms[:, 1, :].copy()
            landms[:, 1, :] = landms[:, 0, :]
            landms[:, 0, :] = tmp
            tmp1 = landms[:, 4, :].copy()
            landms[:, 4, :] = landms[:, 3, :]
            landms[:, 3, :] = tmp1
            landms = landms.reshape([-1, 10])

        return image, boxes, landms

    @staticmethod
    def _pad_to_square(image, rgb_mean, pad_image_flag):
        if not pad_image_flag:
            return image
        height, width, _ = image.shape
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        image_t[:, :] = rgb_mean
        image_t[0:0 + height, 0:0 + width] = image
        return image_t

    @staticmethod
    def _resize(image, insize):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                          cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randint(5)]
        image = cv2.resize(image, (insize, insize), interpolation=interp_method)
        image = image.astype(np.float32)
        image = (image / 255. - 0.5) / 1
        return image

    @staticmethod
    def match(threshold, bbox, anchors, variances, clses, landms):
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
        conf[best_truth_overlap < threshold] = 0    # label as background
        loc = encode(matches, anchors, variances)

        matches_landm = landms[best_truth_idx]
        matches_landm.shape

        landm = encode_landm(matches_landm, anchors, variances)
        return loc, conf, landm

    def process_train_img(self, img: np.ndarray, ann: np.ndarray):
        rgb_means = (104, 117, 123)
        boxes = ann[:, :4].copy()
        landm = ann[:, 4:-1].copy()
        labels = ann[:, -1].copy()
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = self._crop(img, boxes, labels, landm, self.in_hw)
        image_t = self._distort(image_t)
        image_t = self._pad_to_square(image_t, rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = self._mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        new_img = self._resize(image_t, self.in_hw)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        new_ann = np.hstack((boxes_t, landm_t, labels_t))
        return new_img, new_ann

    def ann_to_label(self, ann: np.ndarray):
        num_anchors = len(self.anchors)
        bbox = ann[:, :4]
        landms = ann[:, 4:14]
        clses = ann[:, -1]

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:
        pass
        
