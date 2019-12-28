import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
import tensorflow.python.keras.losses as kls
import tensorflow.python.keras.constraints as kc
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper
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
        ])  # type: iaa.meta.Augmenter

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))
        img_shape = list(self.in_hw) + [3]
        if self.use_softmax:
            def _parser(i: tf.Tensor):
                img_path, identity = tf.numpy_function(
                    lambda ii: (image_ann_list[ii][0],
                                image_ann_list[ii][2].astype(np.int32)),
                    [i], [tf.string, tf.int32], 'get_idx')
                # load image
                raw_img = self.read_img(img_path)

                # normlize image
                if is_normlize is True:
                    img = self.normlize_img(raw_img)  # type:tf.Tensor
                else:
                    img = tf.cast(raw_img, tf.float32)

                img.set_shape(img_shape)
                identity.set_shape([])
                # Note y_true shape will be [batch,1]
                return (img), ([identity])
        else:
            idx_range = np.arange(len(image_ann_list))

            def _get_idx(i: int):
                a_path, identitys, _ = image_ann_list[i]
                p_path = image_ann_list[np.random.choice(identitys)][0]
                n_path = image_ann_list[np.random.choice(np.delete(idx_range, identitys))][0]
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
                  .shuffle(batch_size * 500).repeat()
                  .map(_parser, -1)
                  .batch(batch_size, True).prefetch(-1))
        else:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .map(_parser, -1)
                  .batch(batch_size, True).prefetch(-1))
        return ds


class TripletLoss(kls.Loss):
    def __init__(self, batch_size: int, alpha: float,
                 reduction='auto', name=None):
        super().__init__(reduction=reduction, name=name)
        self.batch_size = batch_size
        self.alpha = alpha
        self.dist_var = tf.get_variable('distance_diff', shape=(self.batch_size, 1), dtype=tf.float32,
                                        initializer=tf.zeros_initializer(), trainable=False)  # type:tf.Variable

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        a, p, n = tf.split(y_pred, 3, axis=-1)
        p_dist = tf.reduce_sum(tf.square(a - p), axis=-1, keep_dims=True)  # [batch_size,1]
        n_dist = tf.reduce_sum(tf.square(a - n), axis=-1, keep_dims=True)  # [batch_size,1]
        dist_diff = p_dist - n_dist
        total_loss = tf.reduce_sum(tf.nn.relu(dist_diff + self.alpha)) / self.batch_size
        return total_loss + 0 * self.dist_var.assign(dist_diff)


class Sparse_SoftmaxLoss(kls.Loss):
    def __init__(self, scale=30, reduction='auto', name=None):
        """ sparse softmax loss with scale

        Parameters
        ----------
        scale : int, optional

            loss scale, by default 30

        reduction : [type], optional

            by default 'auto'

        name : str, optional

            by default None

        """
        super().__init__(reduction=reduction, name=name)
        self.scale = scale

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return k.backend.sparse_categorical_crossentropy(y_true, self.scale * y_pred, True)


class Sparse_AmsoftmaxLoss(kls.Loss):
    def __init__(self, batch_size: int, scale: int = 30, margin: int = 0.35,
                 reduction='auto', name=None):
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
        self.batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ loss calc

        Parameters
        ----------
        y_true : tf.Tensor

            shape = [batch,1] type = tf.int32

        y_pred : tf.Tensor

            shape = [batch,class num] type = tf.float32

        Returns
        -------

        tf.Tensor

            loss
        """
        idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
        y_true_pred = tf.gather_nd(y_pred, idxs)
        y_true_pred = tf.expand_dims(y_true_pred, 1)
        y_true_pred_margin = y_true_pred - self.margin
        _Z = tf.concat([y_pred, y_true_pred_margin], 1)
        _Z = _Z * self.scale
        logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
        logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ))
        return - y_true_pred_margin * self.scale + logZ


class Sparse_AsoftmaxLoss(kls.Loss):
    def __init__(self, batch_size: int, scale: int = 30, margin: int = 0.35,
                 reduction='auto', name=None):
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
        self.batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32), 1)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
        y_true_pred = tf.gather_nd(y_pred, idxs)  # find the y_pred
        y_true_pred = tf.expand_dims(y_true_pred, 1)
        y_true_pred_margin = 1 - 8 * tf.square(y_true_pred) + 8 * tf.square(tf.square(y_true_pred))
        # min(y_true_pred, y_true_pred_margin)
        y_true_pred_margin = y_true_pred_margin - tf.nn.relu(y_true_pred_margin - y_true_pred)
        _Z = tf.concat([y_pred, y_true_pred_margin], 1)
        _Z = _Z * self.scale  # use scale expand value range
        logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
        logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ))  # Z - exp(scale * y_true_pred)
        return - y_true_pred_margin * self.scale + logZ


class TripletAccuracy(MeanMetricWrapper):
    """ Triplet loss Calculates how often predictions matches labels. """

    def __init__(self, dist: ResourceVariable, threshold: float, name='acc', dtype=None):
        super(TripletAccuracy, self).__init__(
            lambda y_true, y_pred, dist, threshold: tf.cast(dist < -threshold, tf.float32),
            name, dtype=dtype, dist=dist.read_value(), threshold=threshold)
