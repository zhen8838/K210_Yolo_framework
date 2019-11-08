import numpy as np
import tensorflow as tf
from cv2 import resize, imdecode, IMREAD_COLOR
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
from tools.base import BaseHelper, INFO
from typing import List
import matplotlib.pyplot as plt


class LFFDHelper(BaseHelper):
    def __init__(self, image_ann: np.ndarray, featuremap_size: np.ndarray, in_hw: np.ndarray,
                 neg_resize_factor: np.ndarray, validation_split: float, neg_sample_ratio: float):
        """ LFFD model helper

        Parameters
        ----------
        image_ann : str

            dataset meta file path,
            value =
            `{'train_pos': 'xxx/train_pos.tfrecords',
            'train_pos_num': xx,
            'val_pos': 'xxx/val_pos.tfrecords',
            'val_pos_num': xx,
            'train_neg': 'xxx/train_neg.tfrecords',
            'train_neg_num': xx,
            'val_neg': 'xxx/val_neg.tfrecords',
            'val_neg_num': xx}`

        featuremap_size : np.ndarray

            featruemap size array

        in_hw : np.ndarray

            network input image height and weight

        neg_resize_factor : np.ndarray

            negative sample random resize factor

        validation_split : float


        neg_sample_ratio : float

            negative sample in training ratio

        """
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.meta: dict = np.load(image_ann, allow_pickle=True)[()]
        self.data = np.load(self.meta['data_path'], allow_pickle=True)
        self.train_pos = self.meta['train_pos']  # type:list
        self.train_neg = self.meta['train_neg']  # type:list
        self.val_pos = self.meta['val_pos']  # type:list
        self.val_neg = self.meta['val_neg']  # type:list
        self.train_total_data = self.meta['train_num']  # type:int
        self.val_total_data = self.meta['val_num']  # type:int
        self.train_epoch_step: int = None
        self.val_epoch_step: int = None

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.Augmenter

        # set paramter
        self.featuremap_size: np.ndarray = np.array(featuremap_size)
        self.scale_num: int = self.featuremap_size.size
        self.neg_resize_factor: np.ndarray = np.array(neg_resize_factor)
        self.in_hw: np.ndarray = np.array(in_hw)
        self.small_list: np.ndarray = np.flip(featuremap_size) + 1
        self.large_list: np.ndarray = self.small_list * 2
        self.small_weak_list: np.ndarray = np.floor(self.small_list * 0.9).astype(np.int)
        self.large_weak_list: np.ndarray = np.ceil(self.large_list * 1.1).astype(np.int)
        self.stride_list: np.ndarray = in_hw[0] // (self.featuremap_size + 1)
        self.center_list: np.ndarray = self.stride_list - 1
        self.out_channels = 6
        self.normal_para = self.large_list // 2  # Normalization parameters
        self.neg_sample_ratio: float = neg_sample_ratio  # neg sample ratio

    def _resize_neg_img(self, im_in: np.ndarray, img: np.ndarray):
        """ resize negative image

        Parameters
        ----------
        im_in : np.ndarray

        img : np.ndarray

        """
        in_h, in_w = self.in_hw[0], self.in_hw[1]
        # random resize neg image
        resize_factor = np.random.uniform(self.neg_resize_factor[0],
                                          self.neg_resize_factor[1])

        img = resize(img, None, fx=resize_factor, fy=resize_factor)

        im_h, im_w = img.shape[0], img.shape[1]  # new h,w

        # put neg image into the placeholder
        h_gap = im_h - in_h
        w_gap = im_w - in_w
        if h_gap >= 0:
            y_top = np.random.randint(0, h_gap + 1)
        else:
            y_pad = int(-h_gap / 2)
        if w_gap >= 0:
            x_left = np.random.randint(0, w_gap + 1)
        else:
            x_pad = int(-w_gap / 2)

        if h_gap >= 0 and w_gap >= 0:
            im_in[...] = img[y_top:y_top + in_h, x_left:x_left + in_w]
        elif h_gap >= 0 and w_gap < 0:
            im_in[:, x_pad:x_pad + im_w] = img[y_top:y_top + in_h]
        elif h_gap < 0 and w_gap >= 0:
            im_in[y_pad:y_pad + im_h] = img[:, x_left:x_left + in_w]
        else:
            im_in[y_pad:y_pad + im_h, x_pad:x_pad + im_w] = img

    def _resize_pos_img(self, im_in: np.ndarray, img: np.ndarray,
                        boxes: np.ndarray) -> [np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
        """ resize positive image

        Parameters
        ----------
        im_in : np.ndarray

        img : np.ndarray

        boxes : np.ndarray

        Returns
        -------

        [np.ndarray, np.ndarray, np.ndarray, np.ndarray]

            [boxes, strong_fit, weak_fit, valid]

        """
        target_idx = np.random.randint(len(boxes))
        # select bbox scale
        longer_side = max(boxes[target_idx, 2:])
        if longer_side <= self.small_list[0]:
            scale_idx = 0
        elif longer_side <= self.small_list[1]:
            scale_idx = np.random.randint(2)
        else:
            if np.random.random() > 0.9:
                scale_idx = np.random.randint(self.scale_num + 1)
            else:
                scale_idx = np.random.randint(self.scale_num)

        # random select scale
        if scale_idx == self.scale_num:
            scale_idx -= 1
            side_length = np.random.randint(
                self.large_list[scale_idx],
                self.small_list[scale_idx] + self.large_list[scale_idx])
        else:
            side_length = np.random.randint(
                self.small_list[scale_idx], self.large_list[scale_idx])

        target_scale = side_length / longer_side
        # calculate scale
        boxes = boxes * target_scale

        # init state array
        strong_fit = np.zeros((self.scale_num, len(boxes)), dtype=np.bool)
        weak_fit = np.zeros((self.scale_num, len(boxes)), dtype=np.bool)
        valid = np.zeros((self.scale_num, len(boxes)), dtype=np.bool)

        for i, box in enumerate(boxes):
            longer_side = max(box[2:])
            for j in range(self.scale_num):
                if self.small_list[j] <= longer_side <= self.large_list[j]:
                    strong_fit[j, i] = True
                    valid[j, i] = True
                elif self.small_weak_list[j] <= longer_side <= self.large_weak_list[j]:
                    weak_fit[j, i] = True
                    valid[j, i] = True
        # rescale
        img = resize(img, None, fx=target_scale, fy=target_scale)
        # crop and place the input image centered on the selected box
        vibr = self.stride_list[scale_idx] // 2  # add vibrate
        offset_x = np.random.randint(-vibr, vibr)
        offset_y = np.random.randint(-vibr, vibr)

        center_x = boxes[target_idx, 0] + boxes[target_idx, 2] / 2 + offset_x
        center_y = boxes[target_idx, 1] + boxes[target_idx, 3] / 2 + offset_y
        left = int(center_x - self.in_hw[1] / 2)
        top = int(center_y - self.in_hw[0] / 2)
        right = int(center_x + self.in_hw[1] / 2)
        bottom = int(center_y + self.in_hw[0] / 2)

        if left < 0:
            left_pad = -left
            left = 0
        else:
            left_pad = 0

        if top < 0:
            top_pad = -top
            top = 0
        else:
            top_pad = 0

        img = img[top:bottom, left:right]
        im_in[top_pad:top_pad + img.shape[0],
              left_pad:left_pad + img.shape[1]] = img[...]
        # adjust boxes
        boxes[:, 0] = boxes[:, 0] + left_pad - left
        boxes[:, 1] = boxes[:, 1] + top_pad - top
        return boxes, strong_fit, weak_fit, valid

    def resize_img(self, img: np.ndarray, boxes: np.ndarray = 0.) -> [np.ndarray, list]:
        """ resize image

        Parameters
        ----------
        img : np.ndarray

        boxes : np.ndarray, optional

            when annoation is 0., mean this sampe is negative, by default None

            annoation = [num_box * [left_x, top_y, width, height]]

        Returns
        -------

        [np.ndarray, list]

            image, state_list
            when boxes is not **0.**, state_list contain :
                [boxes, strong_fit, weak_fit, valid]

        """
        im_in = np.zeros([self.in_hw[0], self.in_hw[1], 3], dtype=np.uint8)

        if isinstance(boxes, np.ndarray):
            state_list = self._resize_pos_img(im_in, img, boxes.copy())
            return im_in, state_list
        else:
            self._resize_neg_img(im_in, img)
            return im_in, boxes

    def data_augmenter(self, img: np.ndarray,
                       boxes: tuple = 0.) -> [np.ndarray, np.ndarray]:
        """ data augmenter

        Parameters
        ----------
        img : np.ndarray

        boxes : tuple, optional

            when boxes is **0.** , mean this sample is negative, by default 0.
            when boxes is **tuple** , mean this sample is postive ,
            And contains :
                [boxes, strong_fit, weak_fit, valid]

        Returns
        -------

        [np.ndarray, np.ndarray]

            img_aug , ann_aug
        """
        if isinstance(boxes, tuple):
            # TODO add augment
            return img, boxes
        else:
            image_aug = self.iaaseq(image=img)
            return image_aug, boxes

    def _neg_ann_to_label(self, labels: list, prob_axis: int, bbox_axis: int):
        """ make negative annotation to label

        Parameters
        ----------
        labels : list

        prob_axis : int

        bbox_axis : int

        """
        for label in labels:
            # all negative porb = 1
            label[..., 1] = 1
            # all location porb is valid, all bbox regression is invalid
            label[..., prob_axis] = 1

    def _pos_ann_to_label(self, labels: list, prob_axis: int, bbox_axis: int,
                          boxes: np.ndarray, strong_fit: np.ndarray,
                          weak_fit: np.ndarray, valid: np.ndarray):
        """ make positive annotation to label

        Parameters
        ----------
        labels : list

        prob_axis : int

        bbox_axis : int

        boxes : np.ndarray

            boxes array, [box_num * [x0,y0,x1,y1]]

        strong_fit : np.ndarray

            strong fit area array

        weak_fit : np.ndarray

            weak fit area array

        valid : np.ndarray

            valid area array

        """
        # compute the center coordinates of all receptive fields
        for i in range(self.scale_num):
            # init state
            rf_centers = np.array([
                self.center_list[i] + w * self.stride_list[i]
                for w in range(self.featuremap_size[i])])

            labels[i][..., 1] = 1  # all is negative
            labels[i][..., prob_axis] = 1  # all location porb is valid
            count_strong_fit = np.zeros((self.featuremap_size[i],
                                         self.featuremap_size[i]), dtype=np.int32)
            count_weak_fit = np.zeros((self.featuremap_size[i],
                                       self.featuremap_size[i]), dtype=np.int32)

            for j, (tmp_x0, tmp_y0, tmp_w, tmp_h) in enumerate(boxes):
                if valid[i][j] is False:
                    continue
                tmp_x1 = tmp_x0 + tmp_w
                tmp_y1 = tmp_y0 + tmp_h
                # skip if this bbox is not in the image
                if tmp_x1 <= 0 or tmp_x0 >= self.in_hw[1] \
                        or tmp_y1 <= 0 or tmp_y1 >= self.in_hw[0]:
                    continue

                # calculation of bbox's receptive field coordinates
                rf_x0 = max(0, int((tmp_x0 - self.center_list[i]) /
                                   self.stride_list[i]) + 1)
                rf_x1 = min(self.featuremap_size[i] - 1,
                            int((tmp_x1 - self.center_list[i]) / self.stride_list[i]))
                rf_y0 = max(0, int((tmp_y0 - self.center_list[i]) /
                                   self.stride_list[i]) + 1)
                rf_y1 = min(self.featuremap_size[i] - 1,
                            int((tmp_y1 - self.center_list[i]) / self.stride_list[i]))

                # skip if this receptive field coordinates is wrong
                if rf_x1 < rf_x0 or rf_y1 < rf_y0:
                    continue

                if weak_fit[i][j]:
                    count_weak_fit[rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1] = 1
                else:
                    count_strong_fit[rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1] += 1

                    x_centers = rf_centers[rf_x0:rf_x1 + 1]
                    y_centers = rf_centers[rf_y0:rf_y1 + 1]
                    x0 = (x_centers - tmp_x0) / self.normal_para[i]
                    y0 = (y_centers - tmp_y0) / self.normal_para[i]
                    x1 = (x_centers - tmp_x1) / self.normal_para[i]
                    y1 = (y_centers - tmp_y1) / self.normal_para[i]

                    labels[i][rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1, 2] = x0
                    labels[i][rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1, 3] = y0[:, None]
                    labels[i][rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1, 4] = x1
                    labels[i][rf_y0:rf_y1 + 1, rf_x0:rf_x1 + 1, 5] = y1[:, None]

                # filter some overlap points
                # and some points that are only weakly coincident
                weak_fit_flag = np.logical_or(count_strong_fit > 1, count_weak_fit > 0)
                strong_fit_flag = count_strong_fit == 1
                # strong_fit location is positive
                labels[i][..., 0][strong_fit_flag] = 1  # pos prob is 1
                labels[i][..., 1][strong_fit_flag] = 0  # neg prob is 0
                # filter weak_fit location
                labels[i][..., prob_axis][weak_fit_flag] = 0
                # for bbox regression, only strong_fit area is available
                labels[i][..., bbox_axis][strong_fit_flag] = 1

    def ann_to_label(self, boxes: tuple = 0.) -> (list, list):
        """ convert annotation to label

        Parameters
        ----------
        boxes : tuple, optional

            when boxes is **0.** , mean this sample is negative, by default None
            when boxes is **tuple** , mean this sample is postive ,
            And contains :
                [boxes, strong_fit, weak_fit, valid]

        Returns
        -------

        tuple

            m = scale num
            labels = ([label_1, label_2, ..., label_m, mask_score, mask_bbox]
            label shape = [featuremap_size, featuremap_size, output_channels + 2]
            NOTE when debug need split labels -> [labels, masks]
        """
        labels = [np.zeros((v, v, self.out_channels + 2), dtype=np.float32)
                  for v in self.featuremap_size]
        prob_axis = self.out_channels
        bbox_axis = self.out_channels + 1

        if isinstance(boxes, tuple):
            # boxes, strong_fit, weak_fit, valid = boxes
            self._pos_ann_to_label(labels, prob_axis, bbox_axis, *boxes)
        else:
            self._neg_ann_to_label(labels, prob_axis, bbox_axis)
        return labels

    def build_datapipe(self, pos_list: list, neg_list: list,
                       batch_size: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:

        def _wapper(idx: int, is_augment: bool) -> [np.ndarray, tuple]:
            """ wapper for process image and ann to label """
            raw_img = imdecode(self.data[idx][0], IMREAD_COLOR)[:, :, ::-1]  # BGR to RGB
            raw_img, ann = self.process_img(raw_img, self.data[idx][2], is_augment, True, False)
            labels = self.ann_to_label(ann)
            return (raw_img, *labels)

        def _parser(idx: tf.Tensor):
            # read -> resize -> augmenter -> make labels
            raw_img, *labels = tf.numpy_function(
                _wapper, [idx, is_augment],
                [tf.uint8] + [tf.float32] * self.scale_num, name='process_img')

            # normlize image
            if is_normlize:
                img = self.normlize_img(raw_img)
            else:
                img = tf.cast(raw_img, tf.float32)

            for i, v in enumerate(self.featuremap_size):
                labels[i].set_shape((v, v, self.out_channels + 2))
            img.set_shape((self.in_hw[0], self.in_hw[1], 3))

            return img, tuple(labels)

        if is_training:
            pos_ds = (tf.data.Dataset.range(len(pos_list)).
                      shuffle(batch_size * 500).repeat())
            neg_ds = (tf.data.Dataset.range(len(neg_list)).
                      shuffle(batch_size * 500).repeat())
            ds = (tf.data.experimental.sample_from_datasets(
                [pos_ds, neg_ds], [1 - self.neg_sample_ratio,
                                   self.neg_sample_ratio]).
                map(_parser).batch(batch_size, True).prefetch(-1))
        else:
            raise NotImplementedError('No support to test eval')

        return ds

    def set_dataset(self, batch_size: int, is_augment: bool = True,
                    is_normlize: bool = True, is_training: bool = True):
        self.batch_size = batch_size
        if is_training:
            self.train_dataset = self.build_datapipe(
                self.train_pos, self.train_neg,
                batch_size, is_augment, is_normlize, is_training)
            self.val_dataset = self.build_datapipe(
                self.val_pos, self.val_neg,
                batch_size, False, is_normlize, is_training)

            self.train_epoch_step = self.train_total_data // self.batch_size
            self.val_epoch_step = self.val_total_data // self.batch_size
        else:
            self.test_dataset = self.build_datapipe(
                self.val_list, batch_size, False, is_normlize, is_training)
            self.test_epoch_step = self.test_total_data // self.batch_size

    def draw_image(self, img: np.ndarray, labels: list, is_show: bool = True):
        """ darw image with label~

        Parameters
        ----------
        img : np.ndarray

        labels : list

            labels list

        is_show : bool, optional

            by default True

        """
        if labels is None:
            plt.imshow(img.astype(np.uint8))
        else:
            fig, axs = plt.subplots(self.scale_num, 3, figsize=(9, 15))
            for i in range(self.scale_num):
                score_mask, bbox_mask = labels[i][..., 6:7], labels[i][..., 7:8]
                scale = self.in_hw / score_mask.shape[:2]
                img1 = resize(score_mask, None, fx=scale[1], fy=scale[0])[..., None] * np.array([150, 0, 0])
                img2 = (img * resize(bbox_mask, None, fx=scale[1], fy=scale[0])[..., None])
                imgs = [img, img1, img2]
                for j in range(3):
                    axs[i, j].imshow(imgs[j].astype(np.uint8))
                    axs[i, j].axis('off')

            plt.subplots_adjust(wspace=0.01, hspace=0.02)

        plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
        if is_show:
            plt.show()


class LFFD_Loss(tf.keras.losses.Loss):
    def __init__(self, h: LFFDHelper, hnm_ratio: int, reduction='auto', name=None):
        """ LFFD loss obj

        Parameters
        ----------
        h : LFFDHelper

        hnm_ratio : int

        """
        super().__init__(reduction=reduction, name=name)
        self.hnm_ratio = hnm_ratio

    def classify_loss(self, y_true_score: tf.Tensor,
                      y_pred_score: tf.Tensor, mask_score: tf.Tensor):
        pred_softmax = tf.nn.softmax(y_pred_score, -1)
        # y_true_score axis 0 is pos prob , axis 1 is neg prob
        pos_flag = y_true_score[..., 0] > 0.5
        pos_num = tf.reduce_sum(tf.cast(pos_flag, tf.float32))  # pos sample num

        def pos_fn():
            neg_flag = tf.cast(y_true_score[..., 1] > 0.5, tf.float32)
            neg_num = tf.reduce_sum(neg_flag)  # neg sample num
            neg_num_selected = tf.cast(tf.minimum((self.hnm_ratio * pos_num), neg_num), tf.int32)
            neg_prob = pred_softmax[..., 1] * neg_flag  # 过滤掉不需要的点
            neg_prob_sorted = tf.nn.top_k(tf.reshape(neg_prob, (1, -1)), neg_num_selected)[0]
            prob_threshold = neg_prob_sorted[0, -1]  # 这里的意思是,以neg_num_selected这个地方作为分界线,然后往上取
            neg_grad_flag = tf.less_equal(neg_prob, prob_threshold)
            # 小于阈值的,以及有正样本存在的地方需要计算梯度
            return tf.logical_or(neg_grad_flag, pos_flag)  # loss_mask

        def neg_fn():
            """ when this sample is negative sample """
            neg_choice_ratio = 0.1
            neg_num_selected = tf.cast(tf.cast(tf.size(pred_softmax[..., 1]),
                                               tf.float32) * neg_choice_ratio, tf.int32)
            neg_prob = pred_softmax[..., 1]
            neg_prob_sorted = tf.nn.top_k(tf.reshape(neg_prob, (1, -1)),
                                          neg_num_selected)[0]
            prob_threshold = neg_prob_sorted[0, -1]
            return tf.less_equal(neg_prob, prob_threshold)  # loss mask

        loss_mask = tf.cast(tf.cond(pos_num > 0, pos_fn, neg_fn), tf.float32)

        loss = (tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true_score, logits=y_pred_score, axis=-1)
            * loss_mask
            * tf.squeeze(mask_score, -1))

        return tf.math.divide_no_nan(tf.reduce_sum(loss),
                                     tf.reduce_sum(loss_mask))

    def regress_loss(self, y_true_bbox: tf.Tensor,
                     y_pred_bbox: tf.Tensor, mask_bbox: tf.Tensor):
        return tf.math.divide_no_nan(
            tf.reduce_sum(tf.math.squared_difference(y_true_bbox, y_pred_bbox) * mask_bbox),
            tf.reduce_sum(mask_bbox))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true_score, y_true_bbox, mask_score, mask_bbox = tf.split(y_true, [2, 4, 1, 1], -1)
        y_pred_score, y_pred_bbox = tf.split(y_pred, [2, 4], -1)

        loss = self.classify_loss(y_true_score, y_pred_score, mask_score)
        loss += self.regress_loss(y_true_bbox, y_pred_bbox, mask_bbox)

        return loss
