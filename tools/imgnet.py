import tensorflow as tf
import numpy as np
from tools.base import BaseHelper
from matplotlib.pyplot import imshow, show
from tensorflow.python.keras.losses import LossFunctionWrapper
k = tf.keras
kl = tf.keras.layers


class ImgnetHelper(BaseHelper):
    def __init__(self, image_ann: str, class_num: int, in_hw: list, mixup: bool = False):
        """ ImgnetHelper

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
        self.mixup = mixup
        self.meta: dict = np.load(image_ann, allow_pickle=True)[()]

        self.train_list = self.meta['train_list']
        self.val_list = self.meta['val_list']

        self.train_total_data = self.meta['train_num']
        self.val_total_data = self.meta['val_num']

    def read_img(self, img_path: tf.Tensor) -> tf.Tensor:
        return tf.image.decode_jpeg(tf.io.read_file(img_path), 3)

    def resize_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        img = tf.image.resize_image_with_pad(img, self.in_hw[0], self.in_hw[1])
        return img, ann

    def augment_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.5)
        return img, ann

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       is_augment: bool, is_normlize: bool,
                       is_training: bool) -> tf.data.Dataset:

        def parser(img_path: tf.Tensor, ann: tf.Tensor):
            img = self.read_img(img_path)
            img, ann = self.resize_img(img, ann)
            if is_augment:
                img, ann = self.augment_img(img, ann)
            if is_normlize:
                img = self.normlize_img(img)
            label = tf.one_hot(ann, self.class_num)
            return img, label

        if is_training:
            name_ds = tf.data.Dataset.from_tensor_slices(image_ann_list[0])
            label_ds = tf.data.Dataset.from_tensor_slices(image_ann_list[1])
            ds = (tf.data.Dataset.zip((name_ds, label_ds)).
                  shuffle(1000 * batch_size).
                  repeat().
                  map(parser, -1).
                  batch(batch_size, True).
                  prefetch(-1))

        else:
            raise NotImplementedError("Now can't build test dataset")

        return ds


class Classify_Loss(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction='auto',
                 name='categorical_crossentropy'):
        super().__init__(
            k.losses.categorical_crossentropy, name=name, reduction=reduction,
            from_logits=from_logits, label_smoothing=label_smoothing)
