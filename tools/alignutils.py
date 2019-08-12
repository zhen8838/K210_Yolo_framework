from tools.utils import Helper, center_to_corner, corner_to_center
from skimage.draw import rectangle_perimeter, circle
from skimage.io import imshow, imread, imsave, show
from skimage.color import gray2rgb
from skimage.transform import AffineTransform, warp
import numpy as np
from imgaug import BoundingBoxesOnImage, KeypointsOnImage
from imgaug import augmenters as iaa
import imgaug.augmenters as ia
from tensorflow import numpy_function
import tensorflow.python as tf
from tensorflow import map_fn
from tensorflow.python.keras.backend import switch


class YOLOAlignHelper(Helper):
    def __init__(self, image_ann: str, class_num: int, anchors: np.ndarray,
                 landmark_num: int, in_hw: tuple, out_hw: tuple, validation_split=0.1):
        super().__init__(image_ann, class_num, anchors, in_hw, out_hw, validation_split=validation_split)
        self.landmark_num = landmark_num  # landmark point numbers

    def box_to_label(self, true_box: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 (add alignment) label~

        Parameters
        ----------
        true_box : np.ndarray
            annotation shape :[n,5+ self.landmark_num * 2] value :[n * [ p,x,y,w,h,landmark_num*2] ]
        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1],
                            len(self.anchors[i]), 5 + self.landmark_num * 2 + self.class_num),
                           dtype='float32') for i in range(self.output_number)]
        for i in range(len(true_box)):
            box = true_box[i]
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            l, n = self._get_anchor_index(box[3:5])  # [layer index, anchor index]
            idx, idy = self._xy_grid_index(box[1:3], l)  # [x index , y index]
            labels[l][idy, idx, n, 0:4] = box[1:5]
            labels[l][idy, idx, n, 4] = 1.
            labels[l][idy, idx, n, 5:5 + self.landmark_num * 2] = box[5:5 + self.landmark_num * 2]
            labels[l][idy, idx, n, 5 + self.landmark_num * 2 + int(box[0])] = 1.

        return labels

    def label_to_box(self, labels: np.ndarray, thersh=0.7) -> np.ndarray:
        """reverse the labels to annotation

        Parameters
        ----------
        labels : np.ndarray

        Returns
        -------
        np.ndarray
            annotaions
        """
        new_boxs = np.vstack([label[np.where(label[..., 4] > thersh)] for label in labels])
        new_boxs = np.hstack([np.expand_dims(np.argmax(new_boxs[:, 5 + self.landmark_num * 2:], axis=-1), -1),
                              new_boxs[:, :4],
                              new_boxs[:, 5:5 + self.landmark_num * 2]])  # type:np.ndarray
        return new_boxs

    def process_img(self, img: np.ndarray, true_box: np.ndarray,
                    is_training: bool, is_resize: bool) -> [np.ndarray, np.ndarray]:
        """ process image and true box , if is training then use data augmenter

        Parameters
        ----------
        img : np.ndarray
            image srs
        true_box : np.ndarray
            box , [n*[cls,x,y,w,h,landmark num * 2]]
        is_training : bool
            wether to use data augmenter
        is_resize : bool
            wether to resize the image

        Returns
        -------
        tuple
            image src , true box
        """
        if is_resize:
            """ resize image and keep ratio """
            img_wh = np.array(img.shape[1::-1])
            in_wh = self.in_hw[::-1]

            """ calculate the affine transform factor """
            scale = in_wh / img_wh  # NOTE affine tranform sacle is [w,h]
            scale[:] = np.min(scale)
            # NOTE translation is [w offset,h offset]
            translation = ((in_wh - img_wh * scale) / 2).astype(int)

            """ calculate the box transform matrix """
            true_box[:, 1:3] = (true_box[:, 1:3] * img_wh * scale + translation) / in_wh
            true_box[:, 3:5] = (true_box[:, 3:5] * img_wh * scale) / in_wh
            true_box[:, 5:5 + self.landmark_num * 2] = ((true_box[:, 5:5 + self.landmark_num * 2].reshape(
                (-1, self.landmark_num, 2)) * img_wh * scale + translation) / in_wh).reshape((-1, self.landmark_num * 2))

            """ apply Affine Transform """
            aff = AffineTransform(scale=scale, translation=translation)
            img = warp(img, aff.inverse, output_shape=self.in_hw, preserve_range=True).astype('uint8')

        if is_training:
            img, true_box = self.data_augmenter(img, true_box)

        # normlize image
        img = img / 255.
        return img, true_box

    def draw_image(self, img: np.ndarray, true_box: np.ndarray, is_show=True, scores=None):
        """ draw img and show bbox , set true_box = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        true_box : np.ndarray

           shape : [p,x,y,w,h]

        is_show : bool

            show image

        scores : None

            the confidence scores
        """
        if isinstance(true_box, np.ndarray):
            p = true_box[:, 0]

            left_top = ((true_box[:, 1:3] - true_box[:, 3:5] / 2)[:, ::-1] * img.shape[0:2]).astype('int32')
            right_bottom = ((true_box[:, 1:3] + true_box[:, 3:5] / 2)[:, ::-1] * img.shape[0:2]).astype('int32')

            # convert landmark
            landmarks = true_box[:, 5:5 + self.landmark_num * 2].reshape((-1, self.landmark_num, 2))
            # landmarks = (((landmarks * true_box[:, 3:5])[:, :, ::-1] + (true_box[:, 1:3] - true_box[:, 3:5] / 2)[:, ::-1]) * img.shape[0:2]).astype('int32')
            landmarks = (landmarks[:, :, ::-1] * img.shape[0:2]).astype('int32')

            for i in range(len(p)):
                classes = int(p[i])
                # draw bbox
                rr, cc = rectangle_perimeter(left_top[i], right_bottom[i], shape=img.shape, clip=True)
                img[rr, cc] = self.colormap[classes]
                for j in range(self.landmark_num):  # draw landmark
                    rr, cc = circle(landmarks[i][j][0], landmarks[i][j][1], 2)
                    img[rr, cc] = self.colormap[classes]

        if is_show:
            imshow((img * 255).astype('uint8'))
            show()

    def data_augmenter(self, img: np.ndarray, true_box: np.ndarray) -> tuple:
        """ augmenter for image with bbox and landmark

        Parameters
        ----------
        img : np.ndarray
            img src
        true_box : np.ndarray
            box [cls,x,y,w,h,landmark num * 2]

        Returns
        -------
        tuple
            [image src,box] after data augmenter
        """
        seq_det = self.iaaseq.to_deterministic()  # type: ia.meta.Augmenter
        p = true_box[:, 0:1]
        xywh_box = true_box[:, 1:5]
        landmarks = true_box[:, 5:].reshape((len(true_box) * self.landmark_num, 2))

        bbs = BoundingBoxesOnImage.from_xyxy_array(center_to_corner(xywh_box, in_hw=img.shape[0:2]), shape=img.shape)
        kps = KeypointsOnImage.from_xy_array(landmarks * img.shape[1::-1], shape=img.shape)

        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().clip_out_of_image()
        kps_aug = seq_det.augment_keypoints([kps])[0]  # type:KeypointsOnImage

        xyxy_box = bbs_aug.to_xyxy_array()
        xywh_box = corner_to_center(xyxy_box, in_hw=img.shape[0:2])
        new_landmarks = (kps_aug.to_xy_array() / img.shape[1::-1]).reshape((len(true_box), self.landmark_num * 2))
        new_box = np.hstack((p, xywh_box, new_landmarks))
        return image_aug, new_box

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
        all_xy = (tf.sigmoid(pred_grid_xy[..., :]) + h.xy_offset[layer]) / h.out_hw[layer][::-1]
        all_wh = tf.exp(pred_grid_wh[..., :]) * h.anchors[layer]
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
    helper : Helper
        Helper obj

    Returns
    -------
    tf.Tensor
    ignore_mask :
        ignore_mask, shape = [batch size, h, w, anchors, 1]
    """
    with tf.name_scope('calc_mask_%d' % layer):
        pred_xy, pred_wh, pred_landmark = tf_grid_to_all(p_xy, p_wh, p_landmark, layer, helper)

        masks = []
        for bc in range(helper.batch_size):
            vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
            vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
            iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
            best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
            masks.append(tf.cast(best_iou < iou_thresh, tf.float32))

    return tf.parallel_stack(masks)


def create_yoloalign_loss(h: YOLOAlignHelper, obj_thresh: float, iou_thresh: float, obj_weight: float,
                          noobj_weight: float, wh_weight: float, layer: int):
    """ create the yolo loss function

    Parameters
    ----------
    h : Helper

    obj_thresh : float

    iou_thresh : float

    obj_weight : float

    noobj_weight : float

    wh_weight : float

    layer : int
        the current layer index

    Returns
    -------
    function
        the yolo loss function

            param  : (y_true,y_pred)

            return : loss
    """
    landmark_shape = tf.TensorShape([h.batch_size, h.out_hw[layer][0], h.out_hw[layer][1], h.anchor_number, h.landmark_num, 2])

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
        """ Split Label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        bbox_pred_landmark = y_pred[..., 5:5 + h.landmark_num * 2]
        pred_cls = y_pred[..., 5:]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        obj_mask_bool = y_true[..., 4] > obj_thresh
        true_confidence = y_true[..., 4:5]
        all_true_landmark = y_true[..., 5:5 + h.landmark_num * 2]
        true_cls = y_true[..., 5:]

        all_true_landmark = tf.reshape(all_true_landmark, landmark_shape)
        bbox_pred_landmark = tf.reshape(bbox_pred_landmark, landmark_shape)

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh

        """ Calc the ignore mask  """

        ignore_mask = calc_ignore_mask(all_true_xy, all_true_wh, all_true_landmark,
                                       grid_pred_xy, grid_pred_wh, bbox_pred_landmark,
                                       obj_mask_bool, iou_thresh, layer, h)

        grid_true_xy, grid_true_wh, bbox_true_landmark = tf_all_to_grid(
            all_true_xy, all_true_wh, all_true_landmark, layer, h)

        # NOTE When wh=0 , tf.log(0) = -inf, so use K.switch to avoid it
        grid_true_wh = switch(obj_mask_bool, grid_true_wh, tf.zeros_like(grid_true_wh))
        bbox_true_landmark = switch(obj_mask_bool, bbox_true_landmark, tf.zeros_like(bbox_true_landmark))

        """ define loss """
        coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

        xy_loss = tf.reduce_sum(
            obj_mask * coord_weight * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=grid_true_xy, logits=grid_pred_xy))

        wh_loss = tf.reduce_sum(
            obj_mask * coord_weight * wh_weight * tf.square(tf.subtract(
                x=grid_true_wh, y=grid_pred_wh)))

        landmark_loss = tf.reduce_sum(
            # FIXME 这里无法broadcast，修复
            # NOTE obj_mask shape is [?,7,10,5,1] can't broadcast with [?,7,10,5,5,2]
            y_true[..., 4] * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=bbox_true_landmark, logits=bbox_pred_landmark))

        obj_loss = obj_weight * tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence))

        noobj_loss = noobj_weight * tf.reduce_sum(
            (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence))

        cls_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_cls, logits=pred_cls))

        total_loss = (obj_loss + noobj_loss +
                      cls_loss + xy_loss +
                      wh_loss + landmark_loss) / h.batch_size

        return total_loss

    return loss_fn
