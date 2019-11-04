from tools.yolo import YOLOHelper, center_to_corner, corner_to_center
import cv2
import numpy as np
from imgaug import BoundingBoxesOnImage, KeypointsOnImage
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.python.keras.losses import Loss
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.engine.base_layer_utils import make_variable
from tensorflow.python.ops.init_ops import zeros_initializer
from tensorflow.python.keras.backend import switch
from PIL import Image, ImageFont, ImageDraw
from tools.base import ERROR, INFO, NOTE
from pathlib import Path
from matplotlib.pyplot import imshow, show


class YOLOAlignHelper(YOLOHelper):
    def __init__(self, image_ann: str, class_num: int, anchors: np.ndarray,
                 landmark_num: int, in_hw: tuple, out_hw: tuple, validation_split=0.1):
        super().__init__(image_ann, class_num, anchors, in_hw, out_hw,
                         validation_split=validation_split)
        self.landmark_num = landmark_num  # landmark point numbers

    def ann_to_label(self, ann: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 (add alignment) label~

        Parameters
        ----------
        ann : np.ndarray
            annotation shape :[n,5+ self.landmark_num * 2] value :[n * [ p,x,y,w,h,landmark_num*2] ]
        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1],
                            len(self.anchors[i]), 5 + self.landmark_num * 2 + self.class_num),
                           dtype='float32') for i in range(self.output_number)]
        for i in range(len(ann)):
            box = ann[i]
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            l, n = self._get_anchor_index(box[3:5])  # [layer index, anchor index]
            idx, idy = self._xy_grid_index(box[1:3], l)  # [x index , y index]
            labels[l][idy, idx, n, 0:4] = box[1:5]
            labels[l][idy, idx, n, 4] = 1.
            labels[l][idy, idx, n, 5:5 + self.landmark_num * 2] = box[5:5 + self.landmark_num * 2]
            labels[l][idy, idx, n, 5 + self.landmark_num * 2 + int(box[0])] = 1.

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
        new_ann = np.hstack([np.expand_dims(np.argmax(new_ann[:, 5 + self.landmark_num * 2:], axis=-1), -1),
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

        del img
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

    def _compute_dataset_shape(self) -> tuple:
        """ compute dataset shape to avoid keras check shape error

        Returns
        -------
        tuple
            dataset shape lists
        """
        output_shapes = [tf.TensorShape([None] + list(self.out_hw[i]) + [len(self.anchors[i]), self.class_num + 5 + self.landmark_num * 2])
                         for i in range(len(self.anchors))]
        dataset_shapes = (tf.TensorShape([None] + list(self.in_hw) + [3]), tuple(output_shapes))
        return dataset_shapes


def tf_grid_to_all(pred_grid_xy: tf.Tensor, pred_grid_wh: tf.Tensor, pred_bbox_landmark: tf.Tensor,
                   layer: int, h: YOLOAlignHelper) -> [tf.Tensor, tf.Tensor]:
    """ rescale the pred raw [pred_grid_xy,pred_grid_wh] to all image sclae [0~1],
        recale pred_bbox_landmark from bbox scale to all image sclae [0~1].
        NOTE : This function will auto active `pred value`, so Do Not use activation before this function
    Parameters
    ----------
    pred_grid_xy : tf.Tensor

        shape = [h, w, anchor num, 2]

    pred_grid_wh : tf.Tensor

        shape = [h, w, anchor num, 2]

    pred_bbox_landmark : tf.Tensor

        shape = [h, w, anchor num, landmark num, 2]

    layer : int
        the output layer
    h : YOLOAlignHelper


    Returns
    -------
    tuple

        after process, [all_xy, all_wh, all_landmark]
    """
    with tf.name_scope('xywh_to_all_%d' % layer):
        all_xy = (tf.sigmoid(pred_grid_xy) + h.xy_offset[layer]) / h.out_hw[layer][::-1]
        all_wh = tf.exp(pred_grid_wh) * h.anchors[layer]
        all_landmark = (tf.sigmoid(pred_bbox_landmark) - .5) * tf.expand_dims(all_wh, -2) + tf.expand_dims(all_xy, -2)
    return all_xy, all_wh, all_landmark


def tf_all_to_grid(all_xy: tf.Tensor, all_wh: tf.Tensor, all_landmark: tf.Tensor,
                   layer: int, h: YOLOAlignHelper) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """convert true label xy wh to grid scale, landmark from all image scale to bbox scale

    Parameters
    ----------
    all_xy : tf.Tensor

        shape = [h, w, anchor num, 2]

    all_wh : tf.Tensor

        shape = [h, w, anchor num, 2]

    all_landmark : tf.Tensor

        shape = [h, w, anchor num, landmark num, 2]

    layer : int
        layer index
    h : YOLOAlignHelper


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        `grid_true_xy`, `grid_true_wh` shape = [h, w, anchor num, 2]
        `bbox_true_landmark` shape = [h, w, anchor num, landmark num, 2]
    """
    with tf.name_scope('xywh_to_grid_%d' % layer):
        grid_xy = (all_xy * h.out_hw[layer][::-1]) - h.xy_offset[layer]
        grid_wh = tf.log(all_wh / h.anchors[layer])
        bbox_landmark = (all_landmark - tf.expand_dims(all_xy, -2)) / tf.expand_dims(all_wh, -2) + .5
    return grid_xy, grid_wh, bbox_landmark


def tf_iou(pred_xy: tf.Tensor, pred_wh: tf.Tensor, vaild_xy: tf.Tensor, vaild_wh: tf.Tensor) -> tf.Tensor:
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


def calc_ignore_mask(t_xy_A: tf.Tensor, t_wh_A: tf.Tensor, t_landmark_A: tf.Tensor, p_xy: tf.Tensor,
                     p_wh: tf.Tensor, p_landmark: tf.Tensor, obj_mask: tf.Tensor, iou_thresh: float,
                     layer: int, helper: YOLOAlignHelper) -> tf.Tensor:
    """clac the ignore mask

    Parameters
    ----------
    t_xy_A : tf.Tensor
        ture xy,shape = [batch size,h,w,anchors,2]
    t_wh_A : tf.Tensor
        true wh,shape = [batch size,h,w,anchors,2]
    t_wh_A : tf.Tensor
        true landmarks,shape = [batch size,h,w,anchors,landmark_num * 2]
    p_xy : tf.Tensor
        raw pred xy,shape = [batch size,h,w,anchors,2]
    p_wh : tf.Tensor
        raw pred wh,shape = [batch size,h,w,anchors,2]
    p_landmark : tf.Tensor
        raw pred landmarks,shape = [batch size,h,w,anchors,landmark_num * 2]
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
        pred_xy, pred_wh, pred_landmark = tf_grid_to_all(p_xy, p_wh,
                                                         p_landmark,
                                                         layer, helper)

        masks = []
        for bc in range(helper.batch_size):
            vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
            vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
            iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
            best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
            masks.append(tf.cast(best_iou < iou_thresh, tf.float32))

    return tf.stack(masks)


class YOLOAlign_Loss(Loss):
    def __init__(self, h: YOLOAlignHelper, obj_thresh: float,
                 iou_thresh: float, obj_weight: float,
                 noobj_weight: float, wh_weight: float,
                 landmark_weight: float, layer: int,
                 reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name=None):
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
        self.landmark_weight = landmark_weight
        self.layer = layer
        self.landmark_shape = tf.TensorShape([self.h.batch_size, self.h.out_hw[layer][0],
                                              self.h.out_hw[layer][1],
                                              self.h.anchor_number,
                                              self.h.landmark_num, 2])
        self.landmark_error = make_variable('landmark_error', shape=(),
                                            initializer=zeros_initializer,
                                            trainable=False)

    def call(self, y_true, y_pred):
        """ Split Label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        bbox_pred_landmark = y_pred[..., 5:5 + self.h.landmark_num * 2]
        pred_cls = y_pred[..., 5:]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        obj_mask_bool = y_true[..., 4] > self.obj_thresh
        true_confidence = y_true[..., 4:5]
        all_true_landmark = y_true[..., 5:5 + self.h.landmark_num * 2]
        true_cls = y_true[..., 5:]

        all_true_landmark = tf.reshape(all_true_landmark, self.landmark_shape)
        bbox_pred_landmark = tf.reshape(bbox_pred_landmark, self.landmark_shape)

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh

        """ Calc the ignore mask  """

        ignore_mask = calc_ignore_mask(all_true_xy, all_true_wh, all_true_landmark,
                                       grid_pred_xy, grid_pred_wh, bbox_pred_landmark,
                                       obj_mask_bool, self.iou_thresh, self.layer, self.h)

        grid_true_xy, grid_true_wh, bbox_true_landmark = tf_all_to_grid(
            all_true_xy, all_true_wh, all_true_landmark, self.layer, self.h)

        # NOTE When wh=0 , tf.log(0) = -inf, so use K.switch to avoid it
        grid_true_wh = tf.where(
            tf.tile(obj_mask_bool[..., tf.newaxis],
                    [1, 1, 1, 1, 2]),
            grid_true_wh, tf.zeros_like(grid_true_wh))
        bbox_true_landmark = tf.where(
            tf.tile(obj_mask_bool[..., tf.newaxis, tf.newaxis],
                    [1, 1, 1, 1, self.h.landmark_num, 2]),
            bbox_true_landmark, tf.zeros_like(bbox_true_landmark))

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

        landmark_loss = tf.reduce_sum(
            # NOTE obj_mask shape is [?,7,10,anchor,1] can't broadcast with [?,7,10,anchor,landmark,2]
            self.landmark_weight * obj_mask[..., tf.newaxis] * tf.square(tf.subtract(
                x=bbox_true_landmark, y=tf.sigmoid(bbox_pred_landmark))), [1, 2, 3, 4, 5])

        total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss + landmark_loss + \
            0 * self.landmark_error.assign(tf.reduce_sum(landmark_loss))

        return total_loss


def correct_algin_box(box_xy: tf.Tensor, box_wh: tf.Tensor, landmark: tf.Tensor, input_hw: list, image_hw: list) -> tf.Tensor:
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
        pred_xy_A, pred_wh_A, pred_landmark_A = tf_grid_to_all(pred_xy, pred_wh, pred_landmark, l, h)
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
