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


class Train_ann(object):
    def __init__(self, train_ann_list: dict):
        super().__init__()
        self.img_ann: List[Tuple[AnyStr, int]] = train_ann_list['img_ann']
        self.idmap: List[List[int]] = train_ann_list['idmap']


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
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None

        if image_ann == None:
            self.train_list: np.ndarray = None
            self.val_list: np.ndarray = None
            self.test_list: np.ndarray = None
        else:
            img_ann_list: dict = np.load(image_ann, allow_pickle=True)[()]
            self.train_list: Train_ann = Train_ann(img_ann_list['train_data'])
            self.val_list: str = img_ann_list['val_data']
            self.test_list: str = img_ann_list['val_data']

            self.train_total_data: int = img_ann_list['train_num']
            self.val_total_data: int = img_ann_list['val_num']
            self.test_total_data: int = img_ann_list['val_num']
        del img_ann_list
        self.in_hw = np.array(in_hw)
        self.embedding_size = embedding_size
        self.use_softmax = use_softmax

    def augment_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        img = tf.image.random_flip_left_right(img)

        l = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 4, tf.int32)[0]
        img = tf.cond(l[0] == 1, lambda: img,
                      lambda: tf.image.random_hue(img, 0.15))
        img = tf.cond(l[1] == 1, lambda: img,
                      lambda: tf.image.random_saturation(img, 0.6, 1.6))
        img = tf.cond(l[2] == 1, lambda: img,
                      lambda: tf.image.random_brightness(img, 0.1))
        img = tf.cond(l[3] == 1, lambda: img,
                      lambda: tf.image.random_contrast(img, 0.7, 1.3))
        return img, ann

    def build_train_datapipe(self, image_ann_list: Train_ann, batch_size: int,
                             is_augment: bool, is_normlize: bool,
                             is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))

        img_shape = list(self.in_hw) + [3]
        if self.use_softmax:
            def parser(i: tf.Tensor):
                img_path, label = tf.numpy_function(
                    lambda idx: image_ann_list.img_ann[idx], [i],
                    [tf.string, tf.int64], 'get_idx')
                # load image
                raw_img = self.read_img(img_path)
                if is_augment:
                    raw_img, _ = self.augment_img(raw_img, None)
                # normlize image
                if is_normlize is True:
                    img = self.normlize_img(raw_img)  # type:tf.Tensor
                else:
                    img = tf.cast(raw_img, tf.float32)

                img.set_shape(img_shape)
                label.set_shape([])
                # Note y_true shape will be [batch,1]
                return (img, img), ([label])

        else:
            id_list = np.arange(len(image_ann_list.idmap))

            def parser(i: tf.Tensor):
                img_a_path, label = tf.numpy_function(
                    lambda idx: image_ann_list.img_ann[idx], [i],
                    [tf.string, tf.int64], 'get_idx')
                img_p_path = tf.numpy_function(
                    lambda idx: image_ann_list.img_ann[np.random.choice(image_ann_list.idmap[idx])][0], [label],
                    [tf.string], 'get_p_idx')
                img_n_path = tf.numpy_function(
                    lambda idx: image_ann_list.img_ann[np.random.choice(np.delete(id_list, label))][0], [label],
                    [tf.string], 'get_p_idx')

                # load image
                raw_img_a = self.read_img(img_a_path)
                raw_img_p = self.read_img(img_p_path)
                raw_img_n = self.read_img(img_n_path)
                if is_augment:
                    raw_img_a, _ = self.augment_img(raw_img_a, None)
                    raw_img_p, _ = self.augment_img(raw_img_p, None)
                    raw_img_n, _ = self.augment_img(raw_img_n, None)
                # normlize image
                if is_normlize is True:
                    img_a: tf.Tensor = self.normlize_img(raw_img_a)
                    img_p: tf.Tensor = self.normlize_img(raw_img_p)
                    img_n: tf.Tensor = self.normlize_img(raw_img_n)
                else:
                    img_a = tf.cast(raw_img_a, tf.float32)
                    img_p = tf.cast(raw_img_p, tf.float32)
                    img_n = tf.cast(raw_img_n, tf.float32)

                img_a.set_shape(img_shape)
                img_p.set_shape(img_shape)
                img_n.set_shape(img_shape)
                label.set_shape([])
                # Note y_true shape will be [batch,1]
                return (img_a, img_p, img_n), ([label])

        ds = (tf.data.Dataset.from_tensor_slices(
            tf.range(self.train_total_data))
            .shuffle(batch_size * 500).repeat()
            .map(parser, -1)
            .batch(batch_size, True).prefetch(-1))

        return ds

    def build_val_datapipe(self, image_ann_list: str, batch_size: int,
                           is_augment: bool, is_normlize: bool,
                           is_training: bool) -> tf.data.Dataset:

        img_shape = list(self.in_hw) + [3]
        if self.use_softmax:
            def parser(stream: bytes):
                example = tf.io.parse_single_example(stream, {
                    'img_a': tf.io.FixedLenFeature([], tf.string),
                    'img_b': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                })  # type:dict
                img_a: tf.Tensor = tf.image.decode_image(example['img_a'], channels=3)
                img_b: tf.Tensor = tf.image.decode_image(example['img_b'], channels=3)
                label: tf.Tensor = example['label']
                img_a.set_shape(img_shape)
                img_b.set_shape(img_shape)
                label.set_shape(())
                # Note y_true shape will be [batch,1]
                return (img_a, img_b), ([label])
        else:
            # NOTE when use tripet loss , need feed 3 images
            def parser(stream: bytes):
                example = tf.io.parse_single_example(stream, {
                    'img_a': tf.io.FixedLenFeature([], tf.string),
                    'img_b': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                })  # type:dict
                img_a: tf.Tensor = tf.image.decode_image(example['img_a'], channels=3)
                img_b: tf.Tensor = tf.image.decode_image(example['img_b'], channels=3)
                label: tf.Tensor = example['label']
                img_a.set_shape(img_shape)
                img_b.set_shape(img_shape)
                label.set_shape(())
                # Note y_true shape will be [batch,1]
                return (img_a, img_b, img_a), ([label])
        if is_training:
            ds = (tf.data.TFRecordDataset(image_ann_list)
                  .shuffle(batch_size * 500).repeat()
                  .map(parser, -1)
                  .batch(batch_size, True).prefetch(-1))
        else:
            ds = (tf.data.TFRecordDataset(image_ann_list)
                  .shuffle(batch_size * 500)
                  .map(parser, -1)
                  .batch(batch_size, True).prefetch(-1))

        return ds

    def set_dataset(self, batch_size: int, is_augment: bool = True,
                    is_normlize: bool = True, is_training: bool = True):
        self.batch_size = batch_size
        if is_training:
            self.train_dataset = self.build_train_datapipe(self.train_list, batch_size,
                                                           is_augment, is_normlize, is_training)
            self.val_dataset = self.build_val_datapipe(self.val_list, batch_size,
                                                       False, is_normlize, is_training)

            self.train_epoch_step = self.train_total_data // self.batch_size
            self.val_epoch_step = self.val_total_data // self.batch_size
        else:
            self.test_dataset = self.build_val_datapipe(self.test_list, batch_size,
                                                        False, is_normlize, is_training)
            self.test_epoch_step = self.test_total_data // self.batch_size


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
