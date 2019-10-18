import tensorflow.python as tf
import tensorflow.python.keras.backend as K
from tensorflow import map_fn
import numpy as np
import os
import skimage
import cv2
from math import cos, sin
from imgaug import augmenters as iaa
import imgaug as ia
from tensorflow import py_function
import pickle
from termcolor import colored

INFO = colored('[ INFO  ]', 'blue')
ERROR = colored('[ ERROR ]', 'red')
NOTE = colored('[ NOTE ]', 'green')


def restore_from_pkl(sess: tf.Session, varlist: list, pklfile: str):
    with open(pklfile, 'rb') as f:
        tensordict = pickle.load(f)
    l = len(tensordict.keys())
    cnt = 0
    assgin_list = []
    for var in varlist:
        for k in tensordict.keys():
            if var.name == k:
                assgin_list.append(tf.assign(var, tensordict[k]))
                cnt += 1
    assert l == cnt
    for i in range(len(assgin_list)):
        sess.run(assgin_list[i])


def restore_ckpt(sess: tf.Session, depth_multiplier: float, var_list: list, ckptdir: str):
    if ckptdir == '' or ckptdir == None:
        pass
    elif 'pkl' in ckptdir:
        restore_from_pkl(sess, tf.global_variables(), ckptdir)
    else:
        ckpt = tf.train.get_checkpoint_state(ckptdir)
        loader = tf.train.Saver(var_list=var_list)
        loader.restore(sess, ckpt.model_checkpoint_path)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))


class Helper(object):
    def __init__(self, image_ann: str, class_num: int, anchors: str, in_hw: tuple, out_hw: tuple, validation_split=0.1):
        self.in_hw = np.array(in_hw)
        assert self.in_hw.ndim == 2
        self.out_hw = np.array(out_hw)
        assert self.out_hw.ndim == 2
        self.validation_split = validation_split  # type:float
        if image_ann == None:
            self.train_list = None
            self.test_list = None
        else:
            img_ann_list = np.load(image_ann, allow_pickle=True)
            num = int(len(img_ann_list) * self.validation_split)
            self.train_list = img_ann_list[num:]  # type:np.ndarray
            self.test_list = img_ann_list[:num]  # type:np.ndarray
            self.train_total_data = len(self.train_list)  # type:int
            self.test_total_data = len(self.test_list)  # type:int
        self.grid_wh = (1 / self.out_hw)[:, [1, 0]]  # hw 转 wh 需要交换两列
        if class_num:
            self.class_num = class_num  # type:int
        if anchors:
            self.anchors = np.load(anchors)  # type:np.ndarray
            self.anchor_number = len(self.anchors[0])
            self.output_number = len(self.anchors)
            self.xy_offset = Helper._coordinate_offset(self.anchors, self.out_hw)  # type:np.ndarray
            self.wh_scale = Helper._anchor_scale(self.anchors, self.grid_wh)  # type:np.ndarray

        self.output_shapes = [tf.TensorShape([None] + list(self.out_hw[i]) +
                                             [len(self.anchors[i]), self.class_num + 5])
                              for i in range(len(self.anchors))]

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])
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
        return (xy * self.out_hw[layer][::-1]) - self.xy_offset[layer]

    # def _wh_to_grid(self, wh: np.ndarray) -> np.ndarray:
    #     """ convert true label wh to grid scale

    #     Parameters
    #     ----------
    #     wh : np.ndarray
    #         label wh shape = [out h, out w,anchor num,2]

    #     Returns
    #     -------
    #     np.ndarray
    #         label wh grid scale, shape = [out h, out w,anchor num,2]
    #     """
    #     # return box[3:5] / self.grid_wh
    #     pass

    def _xy_grid_index(self, box_xy: np.ndarray, layer: int) -> [np.ndarray, np.ndarray]:
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
        return np.floor(box_xy * self.out_hw[layer][::-1]).astype('int')

    @staticmethod
    def _fake_iou(a: np.ndarray, b: np.ndarray) -> float:
        """set a,b center to same,then calc the iou value

        Parameters
        ----------
        a : np.ndarray
            array value = [w,h]
        b : np.ndarray
            array value = [w,h]

        Returns
        -------
        float
            iou value
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

    def _get_anchor_index(self, wh: np.ndarray) -> np.ndarray:
        """get the max iou anchor index

        Parameters
        ----------
        wh : np.ndarray
            value = [w,h]

        Returns
        -------
        np.ndarray
            max iou anchor index
            value  = [layer index , anchor index]
        """
        iou = Helper._fake_iou(wh, self.anchors)
        return np.unravel_index(np.argmax(iou), iou.shape)

    def box_to_label(self, true_box: np.ndarray) -> tuple:
        """convert the annotaion to yolo v3 label~

        Parameters
        ----------
        true_box : np.ndarray
            annotation shape :[n,5] value :[n*[p,x,y,w,h]]

        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1], len(self.anchors[i]),
                            5 + self.class_num), dtype='float32') for i in range(self.output_number)]
        for box in true_box:
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            l, n = self._get_anchor_index(box[3:5])  # [layer index, anchor index]
            idx, idy = self._xy_grid_index(box[1:3], l)  # [x index , y index]
            labels[l][idy, idx, n, 0:4] = np.clip(box[1:5], 1e-8, 1.)
            labels[l][idy, idx, n, 4] = 1.
            labels[l][idy, idx, n, 5 + int(box[0])] = 1.

        return labels

    @staticmethod
    def _coordinate_offset(anchors: np.ndarray, out_hw: np.ndarray) -> np.array:
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
        grid = []
        for l in range(len(anchors)):
            grid_y = np.tile(np.reshape(np.arange(0, stop=out_hw[l][0]), [-1, 1, 1, 1]), [1, out_hw[l][1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=out_hw[l][1]), [1, -1, 1, 1]), [out_hw[l][0], 1, 1, 1])
            grid.append(np.concatenate([grid_x, grid_y], axis=-1))
        return np.array(grid)

    @staticmethod
    def _anchor_scale(anchors: np.ndarray, grid_wh: np.ndarray) -> np.array:
        """construct the anchor scale array , used in convert label to annotation

        Parameters
        ----------
        anchors : np.ndarray
            anchors shape = [n,] = [ n x [m,2]]
        out_hw : np.ndarray
            output height width shape = [n,2]

        Returns
        -------
        np.array
            scale shape = [n,] = [n x [m,2]]
        """
        return np.array([anchors[i] * grid_wh[i] for i in range(len(anchors))])

    def _xy_to_all(self, labels: tuple):
        """convert xy scale from grid to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 0:2] = labels[i][..., 0:2] * self.grid_wh[i] + self.xy_offset[i]

    def _wh_to_all(self, labels: tuple):
        """convert wh scale to all image

        Parameters
        ----------
        labels : tuple
        """
        for i in range(len(labels)):
            labels[i][..., 2:4] = np.exp(labels[i][..., 2: 4]) * self.anchors[i]

    def label_to_box(self, labels: tuple, thersh=.7) -> np.ndarray:
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
        new_boxs = np.c_[np.argmax(new_boxs[:, 5:], axis=-1), new_boxs[:, :4]]
        return new_boxs

    def data_augmenter(self, img: np.ndarray, true_box: np.ndarray) -> tuple:
        """ augmenter for image

        Parameters
        ----------
        img : np.ndarray
            img src
        true_box : np.ndarray
            box

        Returns
        -------
        tuple
            [image src,box] after data augmenter
        """
        seq_det = self.iaaseq.to_deterministic()
        p = true_box[:, 0:1]
        xywh_box = true_box[:, 1:]

        bbs = ia.BoundingBoxesOnImage.from_xyxy_array(self.center_to_corner(xywh_box), shape=img.shape)

        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        xyxy_box = bbs_aug.to_xyxy_array()
        new_box = self.corner_to_center(xyxy_box)
        new_box = np.hstack((p[0:new_box.shape[0], :], new_box))
        return image_aug, new_box

    def _read_img(self, img_path: str) -> np.ndarray:
        """ read image

        Parameters
        ----------
        img_path : str


        Returns
        -------
        np.ndarray
            image src
        """
        img = skimage.io.imread(img_path)
        if len(img.shape) != 3:
            img = skimage.color.gray2rgb(img)
        return img[..., :3]

    def _process_img(self, img: np.ndarray, true_box: np.ndarray, is_training: bool, is_resize: bool) -> tuple:
        """ process image and true box , if is training then use data augmenter

        Parameters
        ----------
        img : np.ndarray
            image srs
        true_box : np.ndarray
            box
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
            img_wh = np.array([img.shape[1], img.shape[0]])
            in_wh = self.in_hw[0][::-1]

            """ calculate the affine transform factor """
            scale = in_wh / img_wh  # NOTE affine tranform sacle is [w,h]
            scale[:] = np.min(scale)
            # NOTE translation is [w offset,h offset]
            translation = ((in_wh - img_wh * scale) / 2).astype(int)

            """ calculate the box transform matrix """
            if isinstance(true_box, np.ndarray):
                true_box[:, 1:3] = (true_box[:, 1:3] * img_wh * scale + translation) / in_wh
                true_box[:, 3:5] = (true_box[:, 3:5] * img_wh * scale) / in_wh
            elif isinstance(true_box, tf.Tensor):
                # NOTE use concat replace item assign
                true_box = tf.concat((true_box[:, 0:1],
                                      (true_box[:, 1:3] * img_wh * scale + translation) / in_wh,
                                      (true_box[:, 3:5] * img_wh * scale) / in_wh), axis=1)

            """ apply Affine Transform """
            aff = skimage.transform.AffineTransform(scale=scale, translation=translation)
            img = skimage.transform.warp(img, aff.inverse, output_shape=self.in_hw[0], preserve_range=True).astype('uint8')

        if is_training:
            img, true_box = self.data_augmenter(img, true_box)

        # normlize image
        img = img / np.max(img)
        return img, true_box

    def generator(self, is_training=True, is_resize=True, is_make_lable=True, train_list=True):
        for image_path, true_box in train_list:
            img = self._read_img(image_path)
            img, true_box = self._process_img(img, true_box, is_training, is_resize)
            if is_make_lable:
                yield img, self.box_to_label(true_box)
            else:
                yield img, true_box

    def _create_dataset(self, image_ann_list: np.ndarray, batch_size: int, rand_seed: int, is_training: bool, is_resize: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_training))

        def _dataset_parser(img_path: str, true_box: np.ndarray):
            img = self._read_img(img_path.numpy().decode())
            img, true_box = self._process_img(img, true_box, is_training, is_resize)
            labels = self.box_to_label(true_box)
            return (img.astype('float32'), *labels)

        @tf.function
        def _parser_wrapper(img_path: str, true_box: np.ndarray):
            img, *labels = py_function(_dataset_parser, [img_path, true_box], [tf.float32] * (len(self.anchors) + 1))
            # NOTE use wrapper function and dynamic list construct (x,(y_1,y_2,...))
            return img, tuple(labels)

        def gen():
            while True:
                for img_path, true_box, _ in image_ann_list:
                    # NOTE use copy avoid change the annotaion value !
                    yield img_path, np.copy(true_box)

        dataset = (tf.data.Dataset.from_generator(gen, (tf.framework_ops.dtypes.string, tf.float32), ([], [None, 5])).
                   shuffle(batch_size * 500 if is_training == True else batch_size * 50, rand_seed).repeat().
                   map(_parser_wrapper, tf.data.experimental.AUTOTUNE).
                   batch(batch_size, True).prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

    def set_dataset(self, batch_size, rand_seed, is_training=True, is_resize=True):
        self.train_dataset = self._create_dataset(self.train_list, batch_size, rand_seed, is_training, is_resize)
        self.test_dataset = self._create_dataset(self.test_list, batch_size, rand_seed, False, is_resize)
        self.batch_size = batch_size
        self.train_epoch_step = self.train_total_data // self.batch_size
        self.test_epoch_step = self.test_total_data // self.batch_size

    def get_iter(self, is_training=True):
        if is_training:
            return self.train_dataset.make_one_shot_iterator().get_next()
        else:
            return self.test_dataset.make_one_shot_iterator().get_next()

    def draw_box(self, img: np.ndarray, true_box: np.ndarray, is_show=True, scores=None):
        """ draw img and show bbox , set true_box = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        true_box : np.ndarray

           shape : [p,x,y,w,h]

        is_show : bool

            show image
        """
        if isinstance(true_box, np.ndarray):
            p = true_box[:, 0]
            xyxybox = self.center_to_corner(true_box[:, 1:])
            for i, a in enumerate(xyxybox):
                classes = int(p[i])
                r_top = tuple(a[0:2].astype(int))
                l_bottom = tuple(a[2:].astype(int))
                r_bottom = (r_top[0], l_bottom[1])
                org = (np.maximum(np.minimum(r_bottom[0], img.shape[1] - 12), 0),
                       np.maximum(np.minimum(r_bottom[1], img.shape[0] - 12), 0))
                cv2.rectangle(img, r_top, l_bottom, self.colormap[classes])
                if isinstance(scores, np.ndarray):
                    cv2.putText(img, f'{classes} {scores[i]:.2f}', org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, self.colormap[classes], thickness=1)
                else:
                    cv2.putText(img, f'{classes}', org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, self.colormap[classes], thickness=1)

        if is_show:
            skimage.io.imshow(img)
            skimage.io.show()

    def center_to_corner(self, true_box, to_all_scale=True):
        if to_all_scale:
            x1 = (true_box[:, 0:1] - true_box[:, 2:3] / 2) * self.in_hw[0, 1]
            y1 = (true_box[:, 1:2] - true_box[:, 3:4] / 2) * self.in_hw[0, 0]
            x2 = (true_box[:, 0:1] + true_box[:, 2:3] / 2) * self.in_hw[0, 1]
            y2 = (true_box[:, 1:2] + true_box[:, 3:4] / 2) * self.in_hw[0, 0]
        else:
            x1 = (true_box[:, 0:1] - true_box[:, 2:3] / 2)
            y1 = (true_box[:, 1:2] - true_box[:, 3:4] / 2)
            x2 = (true_box[:, 0:1] + true_box[:, 2:3] / 2)
            y2 = (true_box[:, 1:2] + true_box[:, 3:4] / 2)

        xyxy_box = np.hstack([x1, y1, x2, y2])
        return xyxy_box

    def corner_to_center(self, xyxy_box, from_all_scale=True):
        if from_all_scale:
            x = ((xyxy_box[:, 2:3] + xyxy_box[:, 0:1]) / 2) / self.in_hw[0, 1]
            y = ((xyxy_box[:, 3:4] + xyxy_box[:, 1:2]) / 2) / self.in_hw[0, 0]
            w = (xyxy_box[:, 2:3] - xyxy_box[:, 0:1]) / self.in_hw[0, 1]
            h = (xyxy_box[:, 3:4] - xyxy_box[:, 1:2]) / self.in_hw[0, 0]
        else:
            x = ((xyxy_box[:, 2:3] + xyxy_box[:, 0:1]) / 2)
            y = ((xyxy_box[:, 3:4] + xyxy_box[:, 1:2]) / 2)
            w = (xyxy_box[:, 2:3] - xyxy_box[:, 0:1])
            h = (xyxy_box[:, 3:4] - xyxy_box[:, 1:2])

        true_box = np.hstack([x, y, w, h])
        return true_box


def tf_xywh_to_all(grid_pred_xy: tf.Tensor, grid_pred_wh: tf.Tensor, layer: int, h: Helper) -> [tf.Tensor, tf.Tensor]:
    """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

    Parameters
    ----------
    grid_pred_xy : tf.Tensor

    grid_pred_wh : tf.Tensor

    layer : int
        the output layer
    h : Helper


    Returns
    -------
    tuple

        after process, [all_pred_xy, all_pred_wh] 
    """
    with tf.name_scope('xywh_to_all_%d' % layer):
        all_pred_xy = (tf.sigmoid(grid_pred_xy[..., :]) + h.xy_offset[layer]) / h.out_hw[layer][::-1]
        all_pred_wh = tf.exp(grid_pred_wh[..., :]) * h.anchors[layer]
    return all_pred_xy, all_pred_wh


def tf_xywh_to_grid(all_true_xy: tf.Tensor, all_true_wh: tf.Tensor, layer: int, h: Helper) -> [tf.Tensor, tf.Tensor]:
    """convert true label xy wh to grid scale

    Parameters
    ----------
    all_true_xy : tf.Tensor

    all_true_wh : tf.Tensor

    layer : int
        layer index
    h : Helper


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        grid_true_xy, grid_true_wh shape = [out h ,out w,anchor num , 2 ]
    """
    with tf.name_scope('xywh_to_grid_%d' % layer):
        grid_true_xy = (all_true_xy * h.out_hw[layer][::-1]) - h.xy_offset[layer]
        grid_true_wh = tf.log(all_true_wh / h.anchors[layer])
    return grid_true_xy, grid_true_wh


def tf_reshape_box(true_xy_A: tf.Tensor, true_wh_A: tf.Tensor, p_xy_A: tf.Tensor, p_wh_A: tf.Tensor, layer: int, helper: Helper) -> tuple:
    """ reshape the xywh to [?,h,w,anchor_nums,true_box_nums,2]
        NOTE  must use obj mask in atrue xywh !
    Parameters
    ----------
    true_xy_A : tf.Tensor
        shape will be [true_box_nums,2]

    true_wh_A : tf.Tensor
        shape will be [true_box_nums,2]

    p_xy_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]

    p_wh_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]

    layer : int

    helper : Helper


    Returns
    -------
    tuple
        true_cent, true_box_wh, pred_cent, pred_box_wh
    """
    with tf.name_scope('reshape_box_%d' % layer):
        true_cent = true_xy_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        true_box_wh = true_wh_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]

        true_cent = tf.tile(true_cent, [helper.batch_size, helper.out_hw[layer][0], helper.out_hw[layer][1], helper.anchor_number, 1, 1])
        true_box_wh = tf.tile(true_box_wh, [helper.batch_size, helper.out_hw[layer][0], helper.out_hw[layer][1], helper.anchor_number, 1, 1])

        pred_cent = p_xy_A[..., tf.newaxis, :]
        pred_box_wh = p_wh_A[..., tf.newaxis, :]
        pred_cent = tf.tile(pred_cent, [1, 1, 1, 1, tf.shape(true_xy_A)[0], 1])
        pred_box_wh = tf.tile(pred_box_wh, [1, 1, 1, 1, tf.shape(true_wh_A)[0], 1])

    return true_cent, true_box_wh, pred_cent, pred_box_wh


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


def calc_ignore_mask(t_xy_A: tf.Tensor, t_wh_A: tf.Tensor, p_xy: tf.Tensor, p_wh: tf.Tensor, obj_mask: tf.Tensor, iou_thresh: float, layer: int, helper: Helper) -> tf.Tensor:
    """clac the ignore mask

    Parameters
    ----------
    t_xy_A : tf.Tensor
        raw ture xy,shape = [batch size,h,w,anchors,2]
    t_wh_A : tf.Tensor
        raw true wh,shape = [batch size,h,w,anchors,2]
    p_xy : tf.Tensor
        raw pred xy,shape = [batch size,h,w,anchors,2]
    p_wh : tf.Tensor
        raw pred wh,shape = [batch size,h,w,anchors,2]
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
        pred_xy, pred_wh = tf_xywh_to_all(p_xy, p_wh, layer, helper)

        # def lmba(bc):
        #     vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
        #     vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
        #     iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
        #     best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
        #     return tf.cast(best_iou < iou_thresh, tf.float32)
        # return map_fn(lmba, tf.range(helper.batch_size), dtype=tf.float32)
        ignore_mask = []
        for bc in range(helper.batch_size):
            vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
            vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
            iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
            best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
            ignore_mask.append(tf.cast(best_iou < iou_thresh, tf.float32))
    return tf.stack(ignore_mask)


def create_loss_fn(h: Helper, obj_thresh: float, iou_thresh: float, obj_weight: float,
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
    shapes = [[-1] + list(h.out_hw[i]) + [len(h.anchors[i]), h.class_num + 5]for i in range(len(h.anchors))]

    # @tf.function
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
        """ split the label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        pred_cls = y_pred[..., 5:]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        true_confidence = y_true[..., 4:5]
        true_cls = y_true[..., 5:]

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh
        obj_mask_bool = y_true[..., 4] > obj_thresh

        """ calc the ignore mask  """

        ignore_mask = calc_ignore_mask(all_true_xy, all_true_wh, grid_pred_xy,
                                       grid_pred_wh, obj_mask_bool,
                                       iou_thresh, layer, h)

        grid_true_xy, grid_true_wh = tf_xywh_to_grid(all_true_xy, all_true_wh, layer, h)
        # NOTE When wh=0 , tf.log(0) = -inf, so use K.switch to avoid it
        grid_true_wh = K.switch(obj_mask_bool, grid_true_wh, tf.zeros_like(grid_true_wh))

        """ define loss """
        coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

        xy_loss = tf.reduce_sum(
            obj_mask * coord_weight * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=grid_true_xy, logits=grid_pred_xy)) / h.batch_size

        wh_loss = tf.reduce_sum(
            obj_mask * coord_weight * wh_weight * tf.square(tf.subtract(
                x=grid_true_wh, y=grid_pred_wh))) / h.batch_size

        obj_loss = obj_weight * tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence)) / h.batch_size

        noobj_loss = noobj_weight * tf.reduce_sum(
            (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence)) / h.batch_size

        cls_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_cls, logits=pred_cls)) / h.batch_size

        total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss

        return total_loss

    return loss_fn
