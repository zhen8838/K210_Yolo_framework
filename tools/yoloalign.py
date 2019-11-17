from tools.yolo import YOLOHelper, center_to_corner, corner_to_center
import cv2
import numpy as np
from imgaug import BoundingBoxesOnImage, KeypointsOnImage
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.python.keras.losses import Loss
import tensorflow.python.keras.backend as K
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from PIL import Image, ImageFont, ImageDraw
from tools.base import ERROR, INFO, NOTE
from pathlib import Path
from matplotlib.pyplot import imshow, show


class YOLOAlignHelper(YOLOHelper):
    def __init__(self, image_ann: str, class_num: int, anchors: np.ndarray,
                 landmark_num: int, in_hw: tuple, out_hw: tuple, validation_split=0.1):
        super().__init__(image_ann, class_num, anchors, in_hw, out_hw,
                         validation_split=validation_split)
        assert self.class_num == 1, 'The yolo algin class num must == 1'
        self.landmark_num = landmark_num  # landmark point numbers

    def ann_to_label(self, ann: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 (add alignment) label~

        Parameters
        ----------
        ann : np.ndarray
            annotation shape :[n,5+ self.landmark_num * 2]
             value :[n * [ p,x,y,w,h,landmark_num*2] ]
        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1],
                            len(self.anchors[i]), 5 + self.landmark_num * 2 + 1 + 1),
                           dtype='float32') for i in range(self.output_number)]

        layer_idx, anchor_idx = self._get_anchor_index(ann[:, 3:5])
        for box, l, n in zip(ann, layer_idx, anchor_idx):
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            # clip box avoid width or heigh == 0 ====> loss = inf
            bb = np.clip(box[1:5], 1e-8, 0.99999999)
            cnt = np.zeros(self.output_number, np.bool)  # assigned flag
            for i in range(len(l)):
                x, y = self._xy_grid_index(bb[0:2], l[i])  # [x index , y index]
                if cnt[l[i]] or labels[l[i]][y, x, n[i], 4] == 1.:
                    # 1. when this output layer already have ground truth, skip
                    # 2. when this grid already being assigned, skip
                    continue
                labels[l[i]][y, x, n[i], 0:4] = bb
                labels[l[i]][y, x, n[i], 4] = (0. if cnt.any() else 1.)
                labels[l[i]][y, x, n[i], 5:5 + self.landmark_num * 2] = box[5:5 + self.landmark_num * 2]
                labels[l[i]][y, x, n[i], 5 + self.landmark_num * 2:5 + self.landmark_num * 2 + 1] = 1  # calss num
                labels[l[i]][y, x, n[i], -1] = 1.  # set gt flag = 1
                cnt[l[i]] = True  # output layer ground truth flag
                if cnt.all():
                    # when all output layer have ground truth, exit
                    break

        return labels

    def label_to_ann(self, labels: np.ndarray, thersh=0.7) -> np.ndarray:
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
        new_ann = np.hstack([np.expand_dims(np.argmax(new_ann[:, 5 + self.landmark_num * 2:5 + self.landmark_num * 2 + 1], axis=-1), -1),
                             new_ann[:, :4],
                             new_ann[:, 5:5 + self.landmark_num * 2]])  # type:np.ndarray
        return new_ann

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
        im_in = np.zeros((self.in_hw[0], self.in_hw[1], 3), np.uint8)

        """ transform factor """
        img_hw = np.array(img.shape[:2])
        img_wh = img_hw[::-1]
        in_wh = self.in_hw[::-1]
        scale = np.min(self.in_hw / img_hw)

        # NOTE hw_off is [h offset,w offset]
        hw_off = ((self.in_hw - img_hw * scale) / 2).astype(int)
        img = cv2.resize(img, None, fx=scale, fy=scale)

        im_in[hw_off[0]:hw_off[0] + img.shape[0],
              hw_off[1]:hw_off[1] + img.shape[1], :] = img[...]
        """ calculate the box transform matrix """
        if isinstance(ann, np.ndarray):
            ann[:, 1:3] = (ann[:, 1:3] * img_wh * scale + hw_off[::-1]) / in_wh
            ann[:, 3:5] = (ann[:, 3:5] * img_wh * scale) / in_wh
            ann[:, 5:5 + self.landmark_num * 2] = ((ann[:, 5:5 + self.landmark_num * 2].reshape(
                (-1, self.landmark_num, 2)) * img_wh * scale + hw_off[::-1]) / in_wh).reshape((-1, self.landmark_num * 2))

        return im_in, ann

    def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True, scores=None):
        """ draw img and show bbox , set ann = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

           shape : [p,x,y,w,h]

        is_show : bool

            show image

        scores : None

            the confidence scores
        """
        if isinstance(ann, np.ndarray):
            p = ann[:, 0]

            left_top = ((ann[:, 1:3] - ann[:, 3:5] / 2) * img.shape[1::-1]).astype('int32')
            right_bottom = ((ann[:, 1:3] + ann[:, 3:5] / 2) * img.shape[1::-1]).astype('int32')

            # convert landmark  from [n,[x,y]]
            landmarks = ann[:, 5:5 + self.landmark_num * 2].reshape((-1, self.landmark_num, 2))
            landmarks = (landmarks * img.shape[1::-1]).astype('int32')

            for i in range(len(p)):
                classes = int(p[i])
                # draw bbox
                cv2.rectangle(img, tuple(left_top[i]),
                              tuple(right_bottom[i]), self.colormap[classes])
                for j in range(self.landmark_num):
                    # NOTE circle( y, x, radius )
                    cv2.circle(img, tuple(landmarks[i][j]), 2, self.colormap[classes])

        if is_show:
            imshow(img)
            show()

    def data_augmenter(self, img: np.ndarray, ann: np.ndarray) -> tuple:
        """ augmenter for image with bbox and landmark

        Parameters
        ----------
        img : np.ndarray
            img src
        ann : np.ndarray
            box [cls,x,y,w,h,landmark num * 2]

        Returns
        -------
        tuple
            [image src,box] after data augmenter
            image src dtype is uint8
        """
        p = ann[:, 0:1]
        xywh_box = ann[:, 1:5]
        landmarks = ann[:, 5:].reshape((len(ann) * self.landmark_num, 2))

        bbs = BoundingBoxesOnImage.from_xyxy_array(center_to_corner(xywh_box, in_hw=img.shape[0:2]), shape=img.shape)
        kps = KeypointsOnImage.from_xy_array(landmarks * img.shape[1::-1], shape=img.shape)
        image_aug, bbs_aug, kps_aug = self.iaaseq(image=img, bounding_boxes=bbs, keypoints=kps)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        xyxy_ann = bbs_aug.to_xyxy_array()
        xywh_ann = corner_to_center(xyxy_ann, in_hw=img.shape[0:2])
        new_landmarks = (kps_aug.to_xy_array() / img.shape[1::-1]).reshape((len(ann), self.landmark_num * 2))
        new_ann = np.hstack((p, xywh_ann, new_landmarks))

        return image_aug, new_ann

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
            raw_img = tf.image.decode_image(tf.io.read_file(img_path),
                                            channels=3, expand_animations=False)
            # resize image -> image augmenter
            raw_img, ann = tf.numpy_function(self.process_img,
                                             [raw_img, ann, is_augment, True, False],
                                             [tf.uint8, tf.float64],
                                             name='process_img')
            # make labels
            labels = tf.numpy_function(self.ann_to_label, [ann],
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
                labels[i].set_shape((None, None, len(self.anchors[i]),
                                     5 + self.landmark_num * 2 + 1 + 1))
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


class YOLOAlign_Loss(Loss):
    def __init__(self, h: YOLOAlignHelper, obj_thresh: float,
                 iou_thresh: float, obj_weight: float,
                 noobj_weight: float, xy_weight: float,
                 wh_weight: float, landmark_weight: float,
                 layer: int, reduction='auto', name=None):
        """  yolo align loss function

        Parameters
        ----------
        h : YOLOHelper

        obj_thresh : float

        iou_thresh : float

        obj_weight : float

        noobj_weight : float

        wh_weight : float

        landmark_weight : float

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
        self.xy_weight = xy_weight
        self.landmark_weight = landmark_weight
        self.layer = layer
        self.anchors = np.copy(self.h.anchors[self.layer])  # type:np.ndarray
        self.op_list = []
        with tf.compat.v1.variable_scope(f'lookups_{self.layer}',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            names = ['le']
            self.lookups: Iterable[Tuple[ResourceVariable, AnyStr]] = [
                (tf.compat.v1.get_variable(name, (), tf.float32,
                                           tf.zeros_initializer(),
                                           trainable=False),
                    name)
                for name in names]

    @staticmethod
    def xywh_to_grid(all_true_xy: tf.Tensor, all_true_wh: tf.Tensor,
                     all_true_landmark: tf.Tensor, out_hw: tf.Tensor,
                     xy_offset: tf.Tensor, anchors: tf.Tensor
                     ) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        """convert true label xy wh to grid scale

        Returns
        -------
        [tf.Tensor, tf.Tensor]

            grid_true_xy, grid_true_wh,grid_true_landmark
             shape = [out h ,out w,anchor num , 2 ]

        """
        grid_true_landmark = all_true_landmark - (tf.expand_dims(all_true_xy, -2) - (tf.expand_dims(all_true_wh, -2) / 2))
        grid_true_xy = (all_true_xy * out_hw[::-1]) - xy_offset
        grid_true_wh = tf.math.log(all_true_wh / anchors)
        return grid_true_xy, grid_true_wh, grid_true_landmark

    @staticmethod
    def xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor,
                    pred_bbox_landmark: tf.Tensor,
                    out_hw: tf.Tensor, xy_offset: tf.Tensor,
                    anchors: tf.Tensor) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
        """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

        Returns
        -------
        [tf.Tensor, tf.Tensor]

            [all_pred_xy, all_pred_wh]
        """
        all_pred_xy = (tf.sigmoid(grid_pred_xy) + xy_offset) / out_hw[::-1]
        all_pred_wh = tf.exp(grid_pred_wh) * anchors
        all_pred_landmark = tf.sigmoid(pred_bbox_landmark) + (
            tf.expand_dims(all_pred_xy, -2) - (tf.expand_dims(all_pred_wh, -2) / 2))
        return all_pred_xy, all_pred_wh, all_pred_landmark

    @staticmethod
    def iou(pred_xy: tf.Tensor, pred_wh: tf.Tensor,
            vaild_xy: tf.Tensor, vaild_wh: tf.Tensor) -> tf.Tensor:
        """ calc the iou form pred box with vaild box
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

    @staticmethod
    def calc_xy_offset(out_hw: tf.Tensor, feature: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """ for dynamic sacle get xy offset tensor for loss calc
        """
        grid_y = tf.tile(tf.reshape(tf.range(0, out_hw[0]),
                                    [-1, 1, 1, 1]), [1, out_hw[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(0, out_hw[1]),
                                    [1, -1, 1, 1]), [out_hw[0], 1, 1, 1])
        xy_offset = tf.concat([grid_x, grid_y], -1)
        return xy_offset

    def call(self, y_true, y_pred):
        out_hw = tf.cast(tf.shape(y_true)[1:3], tf.float32)

        y_true = tf.reshape(y_true,
                            [-1, out_hw[0], out_hw[1],
                             self.h.anchor_number,
                             5 + self.h.landmark_num * 2 + self.h.class_num + 1])
        y_pred = tf.reshape(y_pred,
                            [-1, out_hw[0], out_hw[1],
                             self.h.anchor_number,
                             5 + self.h.landmark_num * 2 + self.h.class_num])

        """ Split Label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        grid_pred_landmark = y_pred[..., 5:5 + self.h.landmark_num * 2]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        true_confidence = y_true[..., 4:5]
        all_true_landmark = y_true[..., 5:5 + self.h.landmark_num * 2]

        all_true_landmark = tf.reshape(
            all_true_landmark, [-1, out_hw[0], out_hw[1],
                                self.h.anchor_number, self.h.landmark_num, 2])
        grid_pred_landmark = tf.reshape(
            grid_pred_landmark, [-1, out_hw[0], out_hw[1],
                                 self.h.anchor_number, self.h.landmark_num, 2])

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh
        obj_mask_bool = tf.cast(y_true[..., 4], tf.bool)
        location_mask = tf.cast(y_true[..., -1], tf.bool)
        batch_obj = tf.reduce_sum(obj_mask)

        """ Calc the ignore mask  """
        xy_offset = self.calc_xy_offset(out_hw, y_pred)

        all_pred_xy, all_pred_wh, all_pred_landmark = self.xywh_to_all(
            grid_pred_xy, grid_pred_wh, grid_pred_landmark,
            out_hw, xy_offset, self.anchors)

        def lmba(bc):
            # bc=1
            # NOTE use location_mask find all ground truth
            gt_xy = tf.boolean_mask(all_true_xy[bc], location_mask[bc])
            gt_wh = tf.boolean_mask(all_true_wh[bc], location_mask[bc])
            iou_score = self.iou(all_pred_xy[bc], all_pred_wh[bc], gt_xy, gt_wh)  # [h,w,anchor,box_num]
            # NOTE this layer gt and pred iou score
            mask_iou_score = tf.reduce_max(tf.boolean_mask(iou_score, obj_mask_bool[bc]), -1)
            # if iou for any ground truth larger than iou_thresh, the pred is true.
            match_num = tf.reduce_sum(tf.cast(iou_score > self.iou_thresh,
                                              tf.float32), -1, keepdims=True)
            return tf.cast(match_num < 1, tf.float32)

        ignore_mask = tf.map_fn(lmba, tf.range(self.h.batch_size), dtype=tf.float32)

        grid_true_xy, grid_true_wh, grid_true_landmark = self.xywh_to_grid(
            all_true_xy, all_true_wh, all_true_landmark,
            out_hw, xy_offset, self.anchors)

        # NOTE When wh=0 , tf.log(0) = -inf, so use K.switch to avoid it
        grid_true_wh = tf.where(tf.tile(obj_mask_bool[..., None],
                                        [1, 1, 1, 1, 2]),
                                grid_true_wh, tf.zeros_like(grid_true_wh))

        """ define loss """
        coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

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

        landmark_loss = tf.reduce_sum(
            obj_mask[..., None] * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=grid_true_landmark, logits=grid_pred_landmark),
            [1, 2, 3, 4, 5])

        with tf.control_dependencies([self.lookups[0][0].assign(tf.reduce_sum(landmark_loss))]):
            total_loss = obj_loss + noobj_loss + xy_loss + wh_loss + landmark_loss

        return total_loss


def correct_algin_box(box_xy: tf.Tensor, box_wh: tf.Tensor,
                      landmark: tf.Tensor, input_hw: list,
                      image_hw: list) -> tf.Tensor:
    """rescae predict box to orginal image scale

    Parameters
    ----------
    box_xy : tf.Tensor
        box xy
    box_wh : tf.Tensor
        box wh
    landmark : tf.Tensor
        landmark 
    input_hw : list
        input shape
    image_hw : list
        image shape

    Returns
    -------
    tf.Tensor
        new boxes
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_hw = tf.cast(input_hw, tf.float32)
    image_hw = tf.cast(image_hw, tf.float32)
    new_shape = tf.round(image_hw * tf.reduce_min(input_hw / image_hw))
    offset = (input_hw - new_shape) / 2. / input_hw
    scale = input_hw / new_shape
    box_yx = (box_yx - offset) * scale
    # NOTE landmark is [x,y] -> new landmarkes is [x,y]
    new_landmark = ((landmark[..., ::-1] - offset) * scale)[..., ::-1]
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
    boxes *= tf.concat([image_hw, image_hw], axis=-1)
    new_landmark *= image_hw[::-1]
    return boxes, new_landmark


def yoloalgin_infer(img_path: Path, infer_model: tf.keras.Model,
                    result_path: Path, h: YOLOAlignHelper,
                    obj_thresh: float = .7, iou_thresh: float = .3):
    """ load images """
    orig_img = h.read_img(str(img_path))
    image_hw = orig_img.shape[0:2]
    img, _ = h.process_img(orig_img, true_box=None, is_augment=False, is_resize=True)
    img = tf.expand_dims(img, 0)
    """ get output """
    y_pred = infer_model.predict(img)
    """ parser output """

    landmark_num = h.landmark_num
    class_num = h.landmark_num
    in_hw = h.in_hw

    """ box list """
    _yxyx_box = []
    _yxyx_box_scores = []
    _xy_landmarks = []
    """ preprocess label """
    for l, pred_label in enumerate(y_pred):
        """ split the label """
        pred_xy = pred_label[..., 0:2]
        pred_wh = pred_label[..., 2:4]
        pred_confidence = pred_label[..., 4:5]
        pred_landmark = tf.reshape(pred_label[..., 5:5 + landmark_num * 2],
                                   pred_label.shape[:-1] + (landmark_num, 2))

        pred_cls = pred_label[..., 5 + landmark_num * 2:]
        # box_scores = obj_score * class_score
        box_scores = tf.sigmoid(pred_cls) * tf.sigmoid(pred_confidence)
        # obj_mask = pred_confidence_score[..., 0] > model.obj_thresh
        """ reshape box  """
        # NOTE tf_xywh_to_all will auto use sigmoid function
        pred_xy_A, pred_wh_A, pred_landmark_A = YOLOAlign_Loss.xywh_to_all(
            pred_xy, pred_wh, pred_landmark, h.out_hw[l],
            h.xy_offset[l], h.anchors[l])
        boxes, landmarkes = correct_algin_box(pred_xy_A, pred_wh_A, pred_landmark_A, in_hw, image_hw)
        boxes = tf.reshape(boxes, (-1, 4))
        box_scores = tf.reshape(box_scores, (-1, class_num))
        landmarkes = tf.reshape(landmarkes, (-1, landmark_num, 2))
        """ append box and scores to global list """
        _yxyx_box.append(boxes)
        _yxyx_box_scores.append(box_scores)
        _xy_landmarks.append(landmarkes)

    yxyx_box = tf.concat(_yxyx_box, axis=0)
    yxyx_box_scores = tf.concat(_yxyx_box_scores, axis=0)
    xy_landmarks = tf.concat(_xy_landmarks, axis=0)

    mask = yxyx_box_scores >= obj_thresh

    """ do nms for every classes"""
    _boxes = []
    _scores = []
    _classes = []
    _landmarkes = []
    for c in range(class_num):
        class_boxes = tf.boolean_mask(yxyx_box, mask[:, c])
        class_box_scores = tf.boolean_mask(yxyx_box_scores[:, c], mask[:, c])
        class_landmarks = tf.boolean_mask(xy_landmarks, mask[:, c])
        select = tf.image.non_max_suppression(
            class_boxes, scores=class_box_scores, max_output_size=30, iou_threshold=iou_thresh)
        class_boxes = tf.gather(class_boxes, select)
        class_box_scores = tf.gather(class_box_scores, select)
        class_landmarks = tf.gather(class_landmarks, select)
        _boxes.append(class_boxes)
        _scores.append(class_box_scores)
        _classes.append(tf.ones_like(class_box_scores) * c)
        _landmarkes.append(class_landmarks)

    boxes = tf.concat(_boxes, axis=0)
    classes = tf.concat(_classes, axis=0)
    scores = tf.concat(_scores, axis=0)
    landmarkes = tf.concat(_landmarkes, axis=0)
    """ draw box  """
    font = ImageFont.truetype(font='asset/FiraMono-Medium.otf',
                              size=tf.cast(tf.floor(3e-2 * image_hw[0] + 0.5), tf.int32).numpy())

    thickness = (image_hw[0] + image_hw[1]) // 300

    """ show result """
    if len(classes) > 0:
        pil_img = Image.fromarray(orig_img)
        print('[' + 'top\tleft\tbottom\tright\tscore\tclass\t' + '\t'.join([f'p{i//2}_{i-i//2}' for i in range(landmark_num * 2)]) + ']')
        for i, c in enumerate(classes):
            box = boxes[i]
            score = scores[i]
            label = '{:2d} {:.2f}'.format(int(c.numpy()), score.numpy())
            draw = ImageDraw.Draw(pil_img)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            strings = f'[{top:.1f}\t{left:.1f}\t{bottom:.1f}\t{right:.1f}\t{score:.2f}\t{int(c):2d}'
            top = max(0, tf.cast(tf.floor(top + 0.5), tf.int32))
            left = max(0, tf.cast(tf.floor(left + 0.5), tf.int32))
            bottom = min(image_hw[0], tf.cast(tf.floor(bottom + 0.5), tf.int32))
            right = min(image_hw[1], tf.cast(tf.floor(right + 0.5), tf.int32))

            if top - image_hw[0] >= 0:
                text_origin = tf.convert_to_tensor([left, top - label_size[1]])
            else:
                text_origin = tf.convert_to_tensor([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=h.colormap[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=h.colormap[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            r = 3
            for points in landmarkes:
                for point in points:
                    strings += f'\t{point[0]:.1f}\t{point[1]:.1f}'
                    draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r),
                                 fill=h.colormap[c])

            print(strings + ']')
            del draw
        pil_img.show()
    else:
        print(NOTE, ' no boxes detected')
