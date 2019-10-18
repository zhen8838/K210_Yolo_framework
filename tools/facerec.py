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
    def __init__(self, image_ann: str, in_hw: tuple, embedding_size: int, use_softmax: bool = True):
        """ face recogintion helper

        Parameters
        ----------
        BaseHelper : [type]



        image_ann : str

            image annotation file path

        in_hw : tuple

            in image height width

        embedding_size : int

            embedding size

        use_softmax : bool, optional

            Note Use Softmax loss set to `True`, Use Triplet loss set to `False`, by default True

        """
        super().__init__(image_ann, 0.1)
        self.in_hw = np.array(in_hw)
        self.embedding_size = embedding_size
        self.use_softmax = use_softmax
        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.Augmenter

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       rand_seed: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))
        img_shape = list(self.in_hw) + [3]
        if self.use_softmax:
            def _parser(i: tf.Tensor):
                img_path, identity = tf.numpy_function(lambda ii: image_ann_list[ii][[0, 2]], [i],
                                                       [tf.string, tf.int32], 'get_idx')
                # load image
                raw_img = self.read_img(img_path)

                # normlize image
                if is_normlize is True:
                    img = self.normlize_img(raw_img)  # type:tf.Tensor
                else:
                    img = tf.cast(raw_img, tf.float32)

                img.set_shape(img_shape)
                # Note y_true shape will be [? ,1]
                return img, [identity]
        else:
            def _get_idx(i: int):
                a_path, identitys, _ = image_ann_list[i]
                p_path = self.img_paths[np.random.choice(identitys)]
                n_path = self.img_paths[np.random.choice(np.delete(self.idx_range, identitys))]
                return a_path, p_path, n_path

            def _parser(i: tf.Tensor):
                a_path, p_path, n_path = tf.numpy_function(
                    _get_idx, [i],
                    [tf.string, tf.string, tf.string], 'get_idx')
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

        if is_training:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .shuffle(batch_size * 500 if is_augment == True else batch_size * 50, rand_seed).repeat()
                  .map(_parser, -1)
                  .batch(batch_size, True).prefetch(-1))
        else:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .map(_parser, -1)
                  .batch(batch_size, True).prefetch(-1))
        return ds


class Triplet_Loss(kls.Loss):
    def __init__(self, h: FcaeRecHelper, alpha: float,
                 reduction=ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.h = h
        self.alpha = alpha
        self.dist_var = tf.get_variable('distance_diff', shape=(self.h.batch_size, 1), dtype=tf.float32,
                                        initializer=tf.zeros_initializer(), trainable=False)  # type:tf.Variable

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        a, p, n = tf.split(y_pred, 3, axis=-1)
        p_dist = tf.reduce_sum(tf.square(a - p), axis=-1, keep_dims=True)  # [batch_size,1]
        n_dist = tf.reduce_sum(tf.square(a - n), axis=-1, keep_dims=True)  # [batch_size,1]
        dist_diff = p_dist - n_dist
        total_loss = tf.reduce_sum(tf.nn.relu(dist_diff + self.alpha)) / self.h.batch_size
        return total_loss + 0 * self.dist_var.assign(dist_diff)


class Sparse_Amsoftmax_Loss(kls.Loss):
    def __init__(self, batch_size: int, scale: int = 30, margin: int = 0.35,
                 reduction=ReductionV2.AUTO, name=None):
        """ sparse addivate margin softmax

        Parameters
        ----------

        scale : int, optional

            by default 30

        margin : int, optional

            by default 0.35

        """
        super().__init__(reduction=reduction, name=name)
        self.scale = scale
        self.margin = margin
        self.batch_idxs = tf.expand_dims(tf.arange(0, batch_size), 1)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # y_true shape = [None, 1] dtype=int32
        idxs = tf.concatenate([self.batch_idxs, y_true], 1)
        y_true_pred = tf.gather_nd(y_pred, idxs)
        y_true_pred = tf.expand_dims(y_true_pred, 1)
        y_true_pred_margin = y_true_pred - self.margin
        _Z = tf.concatenate([y_pred, y_true_pred_margin], 1)
        _Z = _Z * scale
        logZ = tf.reduce_logsumexp(_Z, 1, keep_dims=True)
        logZ = logZ + tf.log(1 - tf.exp(self.scale * y_true_pred - logZ))
        return - y_true_pred_margin * self.scale + logZ


class Sparse_Asoftmax_Loss(kls.Loss):
    def __init__(self, batch_size: int, scale: int = 30, margin: int = 0.35,
                 reduction=ReductionV2.AUTO, name=None):
        """ sparse addivate softmax

        Parameters
        ----------

        scale : int, optional

            by default 30

        margin : int, optional

            by default 0.35

        """
        super().__init__(reduction=reduction, name=name)
        self.scale = scale
        self.batch_idxs = tf.expand_dims(tf.arange(0, batch_size), 1)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true shape = [None, 1] dtype=int32
        idxs = tf.concatenate([self.batch_idxs, y_true], 1)
        y_true_pred = tf.gather_nd(y_pred, idxs)  # find the y_pred
        y_true_pred = tf.expand_dims(y_true_pred, 1)
        y_true_pred_margin = 1 - 8 * tf.square(y_true_pred) + 8 * tf.square(tf.square(y_true_pred))
        # min(y_true_pred, y_true_pred_margin)
        y_true_pred_margin = y_true_pred_margin - tf.nn.relu(y_true_pred_margin - y_true_pred)
        _Z = tf.concatenate([y_pred, y_true_pred_margin], 1)
        _Z = _Z * scale  # use scale expand value range
        logZ = tf.logsumexp(_Z, 1, keep_dims=True)  # 用logsumexp，保证梯度不消失
        logZ = logZ + tf.log(1 - tf.exp(scale * y_true_pred - logZ))  # Z - exp(scale * y_true_pred)
        return - y_true_pred_margin * scale + logZ


class TripletAccuracy(MeanMetricWrapper):
    """ Triplet loss Calculates how often predictions matches labels. """

    def __init__(self, dist: ResourceVariable, threshold: float, name='acc', dtype=None):
        super(TripletAccuracy, self).__init__(
            lambda y_true, y_pred, dist, threshold: tf.cast(dist < -threshold, tf.float32),
            name, dtype=dtype, dist=dist.read_value(), threshold=threshold)
