import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.losses as kls
import tensorflow.python.keras.constraints as kc
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper
from tensorflow.python.keras.utils.losses_utils import ReductionV2
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
from tools.base import INFO, ERROR, NOTE, BaseHelper
import imgaug.augmenters as iaa
from typing import List


class FcaeRecHelper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: tuple, embedding_size: int):
        super().__init__()

        self.in_hw = np.array(in_hw)
        self.embedding_size = embedding_size

        img_paths, identity, mask = np.load(image_ann, allow_pickle=True)  # type:[np.ndarray,np.ndarray,np.ndarray]
        self.img_paths = img_paths
        self.idx_range = np.arange(len(self.img_paths))  # type:np.ndarray

        train_idx = np.where(mask == 0)[0]
        test_idx = np.hstack(np.where(mask == 1) + np.where(mask == 2))

        self.train_list = np.array([np.array([img_paths[idx], identity[idx]]) for idx in train_idx])  # type:np.ndarray
        self.test_list = np.array([np.array([img_paths[idx], identity[idx]]) for idx in test_idx])  # type:np.ndarray

        self.train_total_data = len(self.train_list)  # type:int
        self.test_total_data = len(self.test_list)  # type:int

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.Augmenter

    def _build_datapipe(self, image_ann_list: np.ndarray, batch_size: int, rand_seed: int, is_augment: bool, is_normlize: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))
        img_shape = list(self.in_hw) + [3]

        def _get_idx(i: int):
            a_path, identitys = image_ann_list[i]
            p_path = self.img_paths[np.random.choice(identitys)]
            n_path = self.img_paths[np.random.choice(np.delete(self.idx_range, identitys))]
            return a_path, p_path, n_path

        def _parser(i: tf.Tensor):
            a_path, p_path, n_path = tf.numpy_function(_get_idx, [i], [tf.string, tf.string, tf.string], 'get_idx')
            # load image
            a_img = self.read_img(a_path)
            p_img = self.read_img(p_path)
            n_img = self.read_img(n_path)

            # normlize image
            if is_normlize is True:
                a_img = self.normlize_img(a_img)  # type:tf.Tensor
                p_img = self.normlize_img(p_img)  # type:tf.Tensor
                n_img = self.normlize_img(n_img)  # type:tf.Tensor
            else:
                a_img = tf.cast(a_img, tf.float32)
                p_img = tf.cast(p_img, tf.float32)
                n_img = tf.cast(n_img, tf.float32)

            a_img.set_shape(img_shape)
            p_img.set_shape(img_shape)
            n_img.set_shape(img_shape)

            return (a_img, p_img, n_img), (1.)

        img_ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .shuffle(batch_size * 500 if is_augment == True else batch_size * 50, rand_seed).repeat()
                  .map(_parser, -1)
                  .batch(batch_size, True).prefetch(-1))

        return img_ds

    def set_dataset(self, batch_size, rand_seed, is_augment=True, is_normlize=True):
        self.train_dataset = self._build_datapipe(self.train_list, batch_size, rand_seed, is_augment, is_normlize)
        self.test_dataset = self._build_datapipe(self.test_list, batch_size, rand_seed, False, is_normlize)

        self.batch_size = batch_size
        self.train_epoch_step = self.train_total_data // self.batch_size
        self.test_epoch_step = self.test_total_data // self.batch_size


class Triplet_Loss(kls.Loss):
    def __init__(self, h: FcaeRecHelper, alpha: float, reduction=ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.h = h
        self.alpha = alpha
        self.p_dist_var = tf.get_variable('landmark_error', shape=(self.h.batch_size), dtype=tf.float32,
                                          initializer=tf.zeros_initializer(), trainable=False)  # type:tf.Variable

    def call(self, y_true, y_pred: tf.Tensor):
        a, p, n = tf.split(y_pred, 3, axis=-1)
        p_dist = tf.reduce_sum(tf.square(a - p), axis=-1)  # [batch_size]
        n_dist = tf.reduce_sum(tf.square(a - n), axis=-1)  # [batch_size]
        total_loss = tf.reduce_sum(tf.maximum(p_dist - n_dist + self.alpha, 0), axis=0) / self.h.batch_size
        return total_loss + 0 * self.p_dist_var.assign(p_dist)


class FaceAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels."""

    def __init__(self, p_dist: ResourceVariable, threshold: float, name='acc', dtype=None):
        super(FaceAccuracy, self).__init__(
            lambda y_true, y_pred, p_dist, threshold: tf.reduce_sum(tf.cast(p_dist < threshold, tf.float32)),
            name, dtype=dtype, p_dist=p_dist.read_value(), threshold=threshold)
