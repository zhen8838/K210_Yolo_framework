import tensorflow as tf
from tensorflow.python.keras.losses import huber_loss
import numpy as np
from numpy import random
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from matplotlib.pyplot import imshow, show
from tools.bbox_utils import center_to_corner, bbox_iou, bbox_iof
from tools.base import BaseHelper
from typing import List, Iterable, Tuple


def encode_bbox(matches, anchors, variances):
    g_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 - anchors[:, :2]
    g_cxcy /= (variances[0] * anchors[:, 2:])
    g_wh = (np.clip(matches[:, 2:] - matches[:, :2], 1e-6, 0.999999)) / anchors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(matches, anchors, variances):
    matches = matches.reshape((-1, 5, 2))
    anchors = np.concatenate([np.tile(anchors[:, 0:1, None], [1, 5, 1]),
                              np.tile(anchors[:, 1:2, None], [1, 5, 1]),
                              np.tile(anchors[:, 2:3, None], [1, 5, 1]),
                              np.tile(anchors[:, 3:4, None], [1, 5, 1])], 2)
    g_cxcy = matches[:, :, :2] - anchors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(-1, 5 * 2)
    # return target for smooth_l1_loss
    return g_cxcy


def decode_bbox(bbox, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """

    boxes = np.concatenate((
        anchors[:, :2] + bbox[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(landm, anchors, variances):
    """Decode landm from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """
    landms = np.concatenate(
        (anchors[:, :2] + landm[:, :2] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 2:4] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 4:6] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 6:8] * variances[0] * anchors[:, 2:],
         anchors[:, :2] + landm[:, 8:10] * variances[0] * anchors[:, 2:]), 1)
    return landms


class RetinaFaceHelper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: tuple,
                 anchor_widths: list,
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
        self.anchor_widths = anchor_widths
        self.anchor_steps = anchor_steps
        self.anchors = self._get_anchors(in_hw, anchor_widths, anchor_steps)
        self.corner_anchors = center_to_corner(self.anchors, False)
        self.anchors_num: int = len(self.anchors)
        self.org_in_hw: np.ndarray = np.array(in_hw)
        self.in_hw = tf.Variable(self.org_in_hw, trainable=False)
        self.pos_thresh: float = pos_thresh
        self.variances: np.ndarray = variances

        self.iaaseq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.SomeOf([1, 3], [
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Affine(rotate=(-30, 30),
                           backend='cv2', order=[0, 1], cval=(0, 255), mode=ia.ALL),
                iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))
            ], True),
            iaa.SomeOf([1, 3], [
                iaa.LinearContrast((0.5, 1.5)),  # contrast distortion
                iaa.AddToHue((-18, 18)),  # hue distortion
                iaa.MultiplySaturation((0.5, 1.5)),  # saturation distortion
                iaa.AddToSaturation((-32, 32))  # brightness distortion
            ], True),
        ], random_order=True)

    @staticmethod
    def _get_anchors(in_hw: List[int],
                     anchor_widths: Iterable[Tuple[int, int]],
                     anchor_steps: List[int]) -> np.ndarray:
        """ get anchors """
        feature_maps = [[int(in_hw[0] / step), int(in_hw[1] / step)] for step in anchor_steps]
        anchors = []
        for k, f in enumerate(feature_maps):
            anchor_width = anchor_widths[k]
            feature = np.empty((f[0], f[1], len(anchor_width), 4))
            for i in range(f[0]):
                for j in range(f[1]):
                    for n, width in enumerate(anchor_width):
                        s_kx = width
                        s_ky = width
                        cx = (j + 0.5) * anchor_steps[k] / in_hw[1]
                        cy = (i + 0.5) * anchor_steps[k] / in_hw[0]
                        feature[i, j, n, :] = cx, cy, s_kx, s_ky
            anchors.append(feature)

        anchors = np.concatenate([
            np.reshape(anchors[0], (-1, 4)),
            np.reshape(anchors[1], (-1, 4)),
            np.reshape(anchors[2], (-1, 4))], 0)

        return np.clip(anchors, 0, 1).astype('float32')

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
        return cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

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

        image_aug, bbs_aug, kps_aug = self.iaaseq(image=img,
                                                  bounding_boxes=bbs,
                                                  keypoints=kps)
        new_bbox = bbs_aug.to_xyxy_array()
        new_landm = np.reshape(kps_aug.to_xy_array(), (-1, 5 * 2))
        # remove out of bound bbox
        bbs_xy = (new_bbox[:, :2] + new_bbox[:, 2:]) / 2
        bbs_x, bbs_y = bbs_xy[:, 0], bbs_xy[:, 1]
        mask_t = bbs_y < img.shape[0]
        mask_b = bbs_y > 0
        mask_r = bbs_x < img.shape[1]
        mask_l = bbs_x > 0

        mask = np.logical_and(np.logical_and(mask_t, mask_b),
                              np.logical_and(mask_r, mask_l))
        clses = clses[mask]
        new_bbox = new_bbox[mask]
        new_landm = new_landm[mask]
        return image_aug, [new_bbox, new_landm, clses]

    def normlize_img(self, img: np.ndarray) -> tf.Tensor:
        return (img.astype(np.float32) / 255. - 0.5) / 1

    def process_img(self, img: np.ndarray, ann: np.ndarray, in_hw: np.ndarray,
                    is_augment: bool, is_resize: bool, is_normlize: bool
                    ) -> [np.ndarray, List[np.ndarray]]:
        temp_ann = np.split(ann, [4, -1], 1)
        if is_resize and is_augment:
            img, temp_ann = self.resize_train_img(img, temp_ann, in_hw)
        elif is_resize:
            img, temp_ann = self.resize_img(img, temp_ann, in_hw)
        if is_augment:
            img, temp_ann = self.augment_img(img, temp_ann)
        if is_normlize:
            img = self.normlize_img(img)
        return img, temp_ann

    def ann_to_label(self, ann: List[np.ndarray], in_hw: np.ndarray
                     ) -> List[np.ndarray]:
        bbox, landm, clses = ann

        # convert bbox and landmark scale to 0~1
        bbox[:, 0::2] /= in_hw[1]
        bbox[:, 1::2] /= in_hw[0]
        landm[:, 0::2] /= in_hw[1]
        landm[:, 1::2] /= in_hw[0]

        # find vaild bbox
        overlaps = bbox_iou(bbox, self.corner_anchors)
        best_prior_overlap = np.max(overlaps, 1)
        best_prior_idx = np.argmax(overlaps, 1)
        valid_gt_idx = best_prior_overlap >= 0.2
        best_prior_idx_filter = best_prior_idx[valid_gt_idx]
        if len(best_prior_idx_filter) <= 0:
            label_loc = np.zeros((self.anchors_num, 4), np.float32)
            label_landm = np.zeros((self.anchors_num, 10), np.float32)
            label_conf = np.zeros((self.anchors_num, 1), np.float32)
            return label_loc, label_landm, label_conf

        # calc best gt for each anchors.
        best_truth_overlap = np.max(overlaps, 0)
        best_truth_idx = np.argmax(overlaps, 0)
        best_truth_overlap[best_prior_idx_filter] = 2
        for j in range(len(best_prior_idx)):
            best_truth_idx[best_prior_idx[j]] = j
        matches = bbox[best_truth_idx]
        label_conf = clses[best_truth_idx]
        # filter gt and anchor overlap less than pos_thresh, set as background
        label_conf[best_truth_overlap < self.pos_thresh] = 0

        # encode matches gt to network label
        label_loc = encode_bbox(matches, self.anchors, self.variances)
        matches_landm = landm[best_truth_idx]
        label_landm = encode_landm(matches_landm, self.anchors, self.variances)

        return label_loc, label_landm, label_conf

    def draw_image(self, img: np.ndarray, ann: List[np.ndarray],
                   is_show: bool = True):
        bbox, landm, clses = ann
        for i, flag in enumerate(clses):
            if flag == 1:
                cv2.rectangle(img, tuple(bbox[i][:2].astype(int)),
                              tuple(bbox[i][2:].astype(int)), (255, 0, 0))
                for ldx, ldy, color in zip(landm[i][0::2].astype(int),
                                           landm[i][1::2].astype(int),
                                           [(255, 0, 0), (0, 255, 0),
                                            (0, 0, 255), (255, 255, 0),
                                            (255, 0, 255)]):
                    cv2.circle(img, (ldx, ldy), 1, color, 1)
        if is_show:
            imshow(img)
            show()

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:

        def _py_wrapper(i: int, in_hw: np.ndarray,
                        is_augment: bool, is_resize: bool,
                        is_normlize: bool) -> np.ndarray:
            path, ann = np.copy(image_ann_list[i])
            img = self.read_img(path)
            new_img, new_ann = self.process_img(
                img, ann, in_hw,
                is_augment=is_augment,
                is_resize=is_resize, is_normlize=is_normlize)
            return np.transpose(new_img, [2, 0, 1]), np.concatenate(self.ann_to_label(new_ann, in_hw), -1)

        def _wrapper(i: tf.Tensor) -> tf.Tensor:
            img, label = tf.numpy_function(_py_wrapper,
                                           [i, self.in_hw, is_augment,
                                            True, is_normlize],
                                           [tf.float32, tf.float32])
            img.set_shape((3, None, None))
            label.set_shape((None, 15))
            return img, label

        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                       shuffle(batch_size * 500).
                       repeat().
                       map(_wrapper, -1).
                       batch(batch_size, True).
                       prefetch(-1))
        else:
            dataset = (tf.data.Dataset.from_tensor_slices(
                tf.range(len(image_ann_list))).
                map(_wrapper, -1).
                batch(batch_size, True).
                prefetch(-1))

        return dataset


class RetinaFaceLoss(tf.keras.losses.Loss):
    def __init__(self, h: RetinaFaceHelper, negpos_ratio=7, reduction='auto', name=None):
        super().__init__(reduction=reduction, name=name)
        """ RetinaFace Loss is from SSD Weighted Loss
            See: https://arxiv.org/pdf/1512.02325.pdf for more details.
        """
        self.negpos_ratio = negpos_ratio
        self.h = h
        self.anchors = self.h.anchors
        self.anchors_num = self.h.anchors_num

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        bc_num = tf.shape(y_pred)[0]
        loc_data, landm_data, conf_data = tf.split(y_pred, [4, 10, 2], -1)
        loc_t, landm_t, conf_t = tf.split(y_true, [4, 10, 1], -1)
        # landmark loss
        pos_landm_mask = tf.greater(conf_t, 0.)  # get valid landmark num
        num_pos_landm = tf.maximum(tf.reduce_sum(tf.cast(pos_landm_mask, tf.float32)), 1)  # sum pos landmark num
        pos_landm_mask = tf.tile(pos_landm_mask, [1, 1, 10])  # 10, 16800, 10
        # filter valid lanmark
        landm_p = tf.reshape(tf.boolean_mask(landm_data, pos_landm_mask), (-1, 10))
        landm_t = tf.reshape(tf.boolean_mask(landm_t, pos_landm_mask), (-1, 10))
        loss_landm = tf.reduce_sum(huber_loss(landm_t, landm_p))

        # find have bbox but no landmark location
        pos_conf_mask = tf.not_equal(conf_t, 0)
        # agjust conf_t, calc (have bbox,have landmark) and (have bbox,no landmark) location loss
        conf_t = tf.where(pos_conf_mask, tf.ones_like(conf_t, tf.int32), tf.cast(conf_t, tf.int32))

        # Localization Loss (Smooth L1)
        pos_loc_mask = tf.tile(pos_conf_mask, [1, 1, 4])
        loc_p = tf.reshape(tf.boolean_mask(loc_data, pos_loc_mask), (-1, 4))  # 792,4
        loc_t = tf.reshape(tf.boolean_mask(loc_t, pos_loc_mask), (-1, 4))
        loss_loc = tf.reduce_sum(huber_loss(loc_p, loc_t))

        # Compute max conf across batch for hard negative mining
        batch_conf = tf.reshape(conf_data, (-1, 2))  # 10,16800,2 -> 10*16800,2
        loss_conf = (tf.reduce_logsumexp(batch_conf, 1, True) -
                     tf.gather_nd(batch_conf,
                                  tf.concat([tf.range(tf.shape(batch_conf)[0])[:, None],
                                             tf.reshape(conf_t, (-1, 1))], 1))[:, None])

        # Hard Negative Mining
        loss_conf = loss_conf * tf.reshape(tf.cast(tf.logical_not(pos_conf_mask), tf.float32), (-1, 1))
        loss_conf = tf.reshape(loss_conf, (bc_num, -1))
        idx_rank = tf.argsort(tf.argsort(loss_conf, 1, direction='DESCENDING'), 1)

        num_pos_conf = tf.reduce_sum(tf.cast(pos_conf_mask, tf.float32), 1)
        num_neg_conf = tf.minimum(self.negpos_ratio * num_pos_conf,
                                  tf.cast(tf.shape(pos_conf_mask)[1], tf.float32) - 1.)
        neg_conf_mask = tf.less(tf.cast(idx_rank, tf.float32),
                                tf.tile(num_neg_conf, [1, tf.shape(pos_conf_mask)[1]]))[..., None]

        # calc pos , neg confidence loss
        pos_idx = tf.tile(pos_conf_mask, [1, 1, 2])
        neg_idx = tf.tile(neg_conf_mask, [1, 1, 2])

        conf_p = tf.reshape(tf.boolean_mask(
            conf_data,
            tf.equal(tf.logical_or(pos_idx, neg_idx), True)), (-1, 2))
        conf_t = tf.boolean_mask(conf_t, tf.equal(tf.logical_or(pos_conf_mask, neg_conf_mask), True))

        loss_conf = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(conf_t, conf_p))

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / num_pos_conf
        num_pos_conf = tf.maximum(tf.reduce_sum(num_pos_conf), 1)  # 正样本个数
        loss_loc /= num_pos_conf
        loss_conf /= num_pos_conf
        loss_landm /= num_pos_landm
        return loss_loc + loss_conf + loss_landm
