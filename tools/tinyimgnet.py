import tensorflow as tf
import numpy as np
from tools.base import BaseHelper
from matplotlib.pyplot import imshow, show
k = tf.keras
kl = tf.keras.layers


class TinyImgnetHelper(BaseHelper):
    def __init__(self, image_ann: str, class_num: int, in_hw: list):
        """ TinyImgnetHelper

        Parameters
        ----------
        image_ann : str

            `**.npy` file path

        class_num : int

            class num

        in_hw : list

            input height weight

        """
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None

        self.class_num: int = class_num
        self.in_hw = in_hw
        self.meta: dict = np.load(image_ann, allow_pickle=True)[()]

        self.train_list = self.meta['train_list']
        self.val_list = self.meta['val_list']
        self.test_list = self.meta['test_list']

        self.train_total_data = self.meta['train_num']
        self.val_total_data = self.meta['val_num']
        self.test_total_data = self.meta['test_num']

    @staticmethod
    def parser_example(stream: bytes):
        example = tf.io.parse_single_example(stream, {
            'img_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })  # type:dict
        img = tf.image.decode_image(example['img_raw'], channels=3)  # type:tf.Tensor
        label = example['label']  # type:tf.Tensor
        img.set_shape((64, 64, 3))
        label.set_shape(())
        return img, label

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:

        def _wapper(img, label):
            img = tf.image.resize(img, self.in_hw)

            if is_augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)

            if is_normlize:
                img = self.normlize_img(img)
            return img, label

        if is_training:
            return (tf.data.Dataset.list_files(image_ann_list, True).
                    interleave(tf.data.TFRecordDataset, self.class_num, 1, -1).
                    shuffle(batch_size * 500).
                    repeat().
                    map(TinyImgnetHelper.parser_example, -1).
                    batch(batch_size, True).
                    map(_wapper, -1).
                    prefetch(-1))
        else:
            return (tf.data.Dataset.list_files(image_ann_list, True).
                    interleave(tf.data.TFRecordDataset, self.class_num, 1, -1).
                    map(TinyImgnetHelper.parser_example, -1).
                    batch(batch_size, True).
                    map(_wapper, -1).
                    prefetch(-1))


class Sparse_Classify_Loss(k.losses.Loss):
    def __init__(self, scale=30, reduction='auto', name=None):
        """ sparse softmax loss with scale

        Parameters
        ----------
        scale : int, optional

            loss scale, by default 30

        reduction : [type], optional

            by default `auto`

        name : str, optional

            by default None

        """
        super().__init__(reduction=reduction, name=name)
        self.scale = scale

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return k.backend.sparse_categorical_crossentropy(y_true, self.scale * y_pred, True)
