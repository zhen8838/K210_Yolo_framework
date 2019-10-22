import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2
import numpy as np
from tools.base import BaseHelper
from tools.yolo import center_to_corner, corner_to_center
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage
from skimage.transform import AffineTransform, warp, resize
from skimage.io import imshow, show
from skimage.draw import rectangle_perimeter
from tools.base import INFO, ERROR, NOTE
from pathlib import Path


class CtdetHelper(BaseHelper):
    def __init__(self, image_ann: str, class_num: int, in_hw: tuple, out_hw: tuple, validation_split=0.1):
        super().__init__(image_ann, validation_split)
        self.in_hw = np.array(in_hw)
        assert self.in_hw.ndim == 1
        self.out_hw = np.array(out_hw)
        assert self.out_hw.ndim == 1
        if class_num:
            self.class_num = class_num  # type:int

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.Augmenter

        self.colormap = np.array([
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
            (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200)], dtype=np.uint8)

    def _gaussian_radius(self, w: int, h: int, min_overlap: float = 0.7) -> int:
        """ calc gaussian heatmap radius """
        a1 = 1
        b1 = (h + w)
        c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (h + w)
        c2 = (1 - min_overlap) * w * h
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (h + w)
        c3 = (min_overlap - 1) * w * h
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return max(0, int(min(r1, r2, r3)))

    def _gaussian2d(self, shape: np.ndarray, sigma: float = 1) -> np.ndarray:
        """ generate gaussian heatmap """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    def _draw_umich_gaussian(self, heatmap: np.ndarray, center: [int, int],
                             radius: int, k=1):
        diameter = 2 * radius + 1
        # get gaussian matrix
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)

        x, y = center
        h, w = self.out_hw  # out hw = heatmap hw

        # avoid gaussian matrix beyond heatmap boundary
        left, right = min(x, radius), min(w - x, radius + 1)
        top, bottom = min(y, radius), min(h - y, radius + 1)

        # select heatmap area and assignment 2d gaussian matrix
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def data_augmenter(self, img: np.ndarray, ann: np.ndarray) -> [np.ndarray, np.ndarray]:
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
        if ann is not None:
            seq_det = self.iaaseq.to_deterministic()
            p = ann[:, 0:1]
            xywh_box = ann[:, 1:]

            bbs = BoundingBoxesOnImage.from_xyxy_array(center_to_corner(xywh_box, in_hw=img.shape[0:2]), shape=img.shape)

            image_aug = seq_det.augment_images([img])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

            xyxy_box = bbs_aug.to_xyxy_array()
            new_ann = corner_to_center(xyxy_box, in_hw=img.shape[0:2])
            new_ann = np.hstack((p[0:new_ann.shape[0], :], new_ann))
            return image_aug, new_ann
        else:
            return img, ann

    def resize_img(self, img: np.ndarray, ann: np.ndarray = None) -> [np.ndarray, np.ndarray]:
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
        if ann is not None:
            ann[:, 1:3] = (ann[:, 1:3] * img_wh * scale + translation) / in_wh
            ann[:, 3:5] = (ann[:, 3:5] * img_wh * scale) / in_wh

        """ apply Affine Transform """
        aff = AffineTransform(scale=scale, translation=translation)
        img = warp(img, aff.inverse, output_shape=self.in_hw, preserve_range=True).astype('uint8')
        return img, ann

    def colors_img(self, heatmap: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        heatmap : np.ndarray

        clses : int

        Returns
        -------
        np.ndarray

            colors image
        """
        color = np.array([np.array(self.colormap, dtype=np.float32)[c][np.newaxis, np.newaxis, :] *
                          heatmap[:, :, c:c + 1] for c in range(self.class_num)])
        return resize(np.sum(color, 0), self.in_hw, preserve_range=True).astype('uint8')

    def blend_img(self, raw_img: np.ndarray, colors_img: np.ndarray, factor: float = 0.6) -> np.ndarray:
        """ blend colors image to raw image

        Parameters
        ----------
        raw_img : np.ndarray

        colors_img : np.ndarray

        factor : float, optional
            by default 0.6

        Returns
        -------
        np.ndarray
            blended image
        """

        return (np.clip(raw_img + colors_img * (1 + factor), 0, 255)).astype('uint8')

    def ann_to_label(self, ann: np.ndarray) -> [
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ center net detection annotation to label

        Parameters
        ----------
        ann : np.ndarray
            annotation [n,5]
            value : [n*[p,x,y,w,h]]

        Returns
        -------
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
           heatmap, obj_wh, ct_offset, mask
        """
        heatmap = np.zeros((self.out_hw[0], self.out_hw[1], self.class_num), dtype=np.float32)
        obj_wh = np.zeros((self.out_hw[0], self.out_hw[1], 2), dtype=np.float32)
        ct_offset = np.zeros((self.out_hw[0], self.out_hw[1], 2), dtype=np.float32)
        mask = np.zeros((self.out_hw[0], self.out_hw[1], 1), dtype=np.float32)
        """ get class and recale xywh 0~1 to out hw scale """
        clses = ann[:, 0:1].astype('uint16').ravel()
        # boxes [x,y,w,h] scale is heatmap shape
        boxes = ann[:, 1:] * np.hstack((self.out_hw[::-1], self.out_hw[::-1]))
        for i in range(len(ann)):
            box = boxes[i]
            ct = np.array([box[0], box[1]], dtype=np.float32)  # x , y
            ct_int = ct.astype(np.int32)
            offset = ct - ct_int
            radius = self._gaussian_radius(np.ceil(box[2]), np.ceil(box[3]))

            self._draw_umich_gaussian(heatmap[:, :, clses[i]], ct_int, radius)

            obj_wh[ct_int[1], ct_int[0]] = box[2:]

            ct_offset[ct_int[1], ct_int[0]] = ct - ct_int

            mask[ct_int[1], ct_int[0], 0] = 1.

        return heatmap, obj_wh, ct_offset, mask

    def label_to_ann(self, labels: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """reverse the ctdet label to annotation

        Parameters
        ----------
        labels : np.ndarray

        Returns
        -------
        np.ndarray
            annotaions
        """
        heatmap, obj_wh, offset, mask = labels

        idx = np.where(np.squeeze(mask, -1) == 1)

        clses = np.array(np.where(heatmap == 1)).T[:, -1:]

        ct_int = np.array(idx).T[:, ::-1]

        return np.concatenate((clses, (ct_int + offset[idx]) / self.out_hw, obj_wh[idx] / self.out_hw), -1)

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       rand_seed: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))

        def parser(i: tf.Tensor):

            img_path, ann = tf.numpy_function(lambda idx: (image_ann_list[idx][0].copy(),
                                                           image_ann_list[idx][1].copy()),
                                              [i], [tf.string, tf.float64])
            # load image
            raw_img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
            # resize image -> image augmenter -> normlize image
            raw_img, ann = tf.numpy_function(self.process_img,
                                             [raw_img, ann, is_augment, True, False],
                                             [tf.uint8, tf.float64])
            # make labels
            labels = tf.numpy_function(self.ann_to_label, [ann], [tf.float32] * 4)  # type:tf.Tensor
            labels = tf.concat(labels, -1)

            # normlize image
            if is_normlize is True:
                img = self.normlize_img(raw_img)  # type:tf.Tensor
            else:
                img = tf.cast(raw_img, tf.float32)

            labels.set_shape(self.out_hw.tolist() + [self.class_num + 2 + 2 + 1])
            img.set_shape(self.in_hw.tolist() + [3])

            return img, labels

        if is_training:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                  shuffle(batch_size * 500 if is_training == True else batch_size * 50, rand_seed).repeat().
                  map(parser, -1).
                  batch(batch_size, True).prefetch(-1))
        else:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                  map(parser, -1).
                  batch(batch_size, True).prefetch(-1))

        return ds

    def draw_image(self, img: np.ndarray, ann: np.ndarray, heatmap: np.ndarray = None, is_show=True):
        """ draw img and show bbox , set ann = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

           shape : [p,x,y,w,h]

        heatmap : np.ndarray

           shape : [self.out_h,self.out_w,self.calss_num] NOTE default is None

        is_show : bool

            show image
        """
        img_hw = img.shape[:2]
        p, boxes = ann[:, 0], ann[:, 1:]
        img = img.astype('uint8')

        for i, box in enumerate(boxes):
            yx, hw = box[1::-1], box[:-3:-1]

            classes = int(p[i])
            s = ((yx - hw / 2) * img_hw).astype(np.int)
            e = s + (hw * img_hw).astype(np.int)
            rr, cc = rectangle_perimeter(s, e, shape=img.shape)
            img[rr, cc] = self.colormap[classes]

        if heatmap is not None:
            img = self.blend_img(img, self.colors_img(heatmap))

        if is_show:
            imshow(img)
            show()


class Ctdet_Loss(tf.keras.losses.Loss):
    def __init__(self, h: CtdetHelper, obj_thresh: float, hm_weight: float,
                 wh_weight: float, offset_weight: float,
                 reduction=ReductionV2.AUTO, name=None):
        """ centernet detection loss obj

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
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offset_weight

    def focal_loss(self, true_hm: tf.Tensor, pred_hm: tf.Tensor) -> tf.Tensor:
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory

        Parameters
        ----------
        true_hm : tf.Tensor
            shape : [batch, out_h , out_w, calss_num]
        pred_hm : tf.Tensor
            shape : [batch, out_h , out_w, calss_num]

        Returns
        -------
        tf.Tensor
            heatmap loss
            shape : [1,]
        """
        pred_hm = tf.sigmoid(pred_hm)
        pred_hm = tf.clip_by_value(pred_hm, 1e-4, 1.0 - 1e-4)

        pos_inds = tf.cast(tf.equal(true_hm, 1.), tf.float32)
        neg_inds = 1 - pos_inds
        neg_weights = tf.pow(1 - true_hm, 4)

        # neg entropy loss =  −log(sigmoid(x)) ∗ (1−sigmoid(x))^2 − log(1−sigmoid(x)) ∗ sigmoid(x)^2
        loss = tf.add(- tf.log(pred_hm) * (1 - pred_hm) * (1 - pred_hm) * pos_inds,
                      - tf.log(1 - pred_hm) * pred_hm * pred_hm * neg_weights * neg_inds)
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pos_inds)

        return loss

    def softpuls_focal_loss(self, true_hm: tf.Tensor, pred_hm: tf.Tensor, num_pos: tf.Tensor) -> tf.Tensor:
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory

        Parameters
        ----------
        true_hm : tf.Tensor
            shape : [batch, out_h , out_w, calss_num]
        pred_hm : tf.Tensor
            shape : [batch, out_h , out_w, calss_num]
        num_pos : tf.Tensor
            valid mask sum
            shape : [1, ]

        Returns
        -------
        tf.Tensor
            heatmap loss
            shape : [1,]
        """
        z = true_hm
        x = pred_hm
        x_s = tf.sigmoid(pred_hm)

        pos_inds = tf.cast(tf.equal(z, 1.), tf.float32)
        neg_inds = 1 - pos_inds
        neg_weights = tf.pow(1 - z, 4)

        # neg entropy loss =  −log(sigmoid(x)) ∗ (1−sigmoid(x))^2 − log(1−sigmoid(x)) ∗ sigmoid(x)^2
        loss = tf.add(tf.nn.softplus(-x) * tf.pow(1 - x_s, 2) * pos_inds,
                      (x + tf.nn.softplus(-x)) * tf.pow(x_s, 2) * neg_weights * neg_inds)
        loss = tf.reduce_sum(loss) / num_pos

        return loss

    def regl1_loss(self, gt: tf.Tensor, pred: tf.Tensor, mask: tf.Tensor, num_pos: tf.Tensor) -> tf.Tensor:
        """[summary]

        Parameters
        ----------
        gt : tf.Tensor
            shape : [batch, out_h, out_w, 2]
        pred : tf.Tensor
            shape : [batch, out_h, out_w, 2]
        mask : tf.Tensor
            shape : [batch, out_h, out_w, 1]
        num_pos : tf.Tensor
            valid mask sum
            shape : [1, ]

        Returns
        -------
        tf.Tensor
            total_loss 
            shape : [1,]
        """
        return (tf.reduce_sum(tf.abs((gt - pred) * mask)) / num_pos)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ split the label """
        hm_true, wh_true, off_true, mask = tf.split(y_true, [self.h.class_num, 2, 2, 1], -1)
        hm_pred, wh_pred, off_pred = tf.split(y_pred, [self.h.class_num, 2, 2], -1)
        mask_sum = tf.reduce_sum(tf.cast(mask, tf.float32))

        heatmap_loss = self.softpuls_focal_loss(hm_true, hm_pred, mask_sum)
        wh_loss = self.regl1_loss(wh_true, wh_pred, mask, mask_sum)
        off_loss = self.regl1_loss(off_true, off_pred, mask, mask_sum)
        # total_loss [1, ]
        total_loss = self.hm_weight * heatmap_loss + self.wh_weight * wh_loss + self.offset_weight * off_loss

        return total_loss / self.h.batch_size


def _nms(pred_hm, kernel=3):
    max_hm = tf.nn.max_pool2d(pred_hm, 3, 1, 'SAME')
    keep = tf.cast(tf.equal(max_hm, pred_hm), tf.float32)
    new_hm = pred_hm * keep
    return new_hm


def _parser(pred_hm: tf.Tensor, pred_wh: tf.Tensor, pred_offset: tf.Tensor, h, batch, K=100):
    new_hm = tf.reshape(tf.transpose(pred_hm, [0, 3, 1, 2]), (batch, h.class_num, -1))
    topks, topkids = tf.math.top_k(new_hm, k=K)
    topkids.shape  # [1,20,100]

    yids = topkids // 96  # [batch,class_num, K]
    xids = topkids % 96  # [batch,class_num, K]

    ctopks, ctopkids = tf.math.top_k(tf.reshape(topks, (batch, -1)), k=K)

    clas = ctopkids // K

    new_hm = new_hm.numpy()
    topks, topkids = topks.numpy(), topkids.numpy()
    yids, xids = yids.numpy(), xids.numpy()
    ctopks, ctopkids, clas = ctopks.numpy(), ctopkids.numpy(), clas.numpy()

    yids = yids.ravel()[ctopkids]
    xids = xids.ravel()[ctopkids]

    score = np.array([pred_hm[i, yids[i], xids[i], clas[i]] for i in range(batch)])  # [batch, K]

    wh = np.array([pred_wh[i, yids[i], xids[i]] for i in range(batch)])  # [batch, K, 2]

    offset = np.array([pred_offset[i, yids[i], xids[i]] for i in range(batch)])  # [batch, K, 2]

    bb, yy, xx = np.meshgrid(np.arange(batch), np.arange(h.out_hw[0]), np.arange(h.out_hw[1]), indexing='ij')

    center = np.stack((yy, xx), -1)  # [batch, K, 2]
    center = np.array([center[i, yids[i], xids[i]] for i in range(batch)])  # [batch, K, 2]
    yx = center + offset

    xy = (yx / h.out_hw)[..., [1, 0]]
    wh = wh / h.out_hw

    pred_anns = np.concatenate([clas[..., np.newaxis], xy, wh], -1)
    return pred_anns  # [batch , K ,5]


def ctdet_infer(img_path: Path, infer_model: tf.keras.Model,
                result_path: Path, h: CtdetHelper, K: int = 3):
    """
        CenterNet Detection inference function

    Parameters
    ----------
    img_path : Path

        img path or img folder path

    infer_model : tf.keras.Model

        inference model object

    result_path : Path

        by default None

    h : CtdetHelper

    """
    print(INFO, f'Load Images from {str(img_path)}')
    if img_path.is_dir():
        img_path_list = []
        for suffix in ['bmp', 'jpg', 'jpeg', 'png']:
            img_path_list += list(img_path.glob(f'*.{suffix}'))
        raw_img = np.array([h.read_img(str(p)).numpy() for p in img_path_list])
    elif img_path.is_file():
        raw_img = tf.expand_dims(h.read_img(str(img_path)), 0).numpy()
    else:
        ValueError(f'{ERROR} img_path `{str(img_path)}` is invalid')
    raw_img_hw = np.array([src.shape[0:2] for src in raw_img])

    if result_path != None:
        print(INFO, f'Load Nncase Results from {str(result_path)}')
        if result_path.is_dir():
            ncc_result = np.array([np.fromfile(
                str(result_path / (img_path_list[i].stem + '.bin')),
                dtype='float32') for i in range(len(img_path_list))])  # type:np.ndarray
        elif result_path.is_file():
            ncc_result = np.expand_dims(np.fromfile(str(result_path), dtype='float32'), 0)  # type:np.ndarray
        else:
            ValueError(f'{ERROR} result_path `{str(result_path)}` is invalid')
    else:
        ncc_result = None
    print(INFO, f'Infer Results')
    resize_img = np.stack([h.resize_img(src)[0] for src in raw_img])  # type:np.ndarray

    img = h.normlize_img(resize_img)
    """ get output """
    pred_hm, pred_wh, pred_offset = infer_model.predict(img)  # type:np.ndarray,np.ndarray,np.ndarray

    pred_hm = tf.nn.sigmoid(pred_hm)
    pred_hm = _nms(pred_hm)
    pred_anns = _parser(pred_hm.numpy(), pred_wh, pred_offset, h, img.shape[0], K)
    for i in range(img.shape[0]):
        h.draw_image(resize_img[i], pred_anns[i])
    """ show ctdet result """
    if ncc_result is None:
        pass
    else:
        pass
