""" dcase2018 task5 helper functions """
from tools.base import BaseHelper
import tensorflow_probability as tfp
import tensorflow as tf
from tools.training_engine import BaseTrainingLoop
import numpy as np


def parser_example(stream: bytes) -> [tf.Tensor, tf.Tensor]:
    example = tf.io.parse_single_example(stream, {
        'mel_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)})
    return example['mel_raw'], example['label']


def decode_mel(s: tf.Tensor) -> tf.Tensor:
    return tf.io.parse_tensor(s, tf.float32)


def freqmask(img, max_w=26):
    coord = tf.random.uniform([], 0, tf.shape(img, tf.int64)[0], tf.int64)
    width = tf.random.uniform([], 8, max_w, tf.int64)
    cut = tf.stack([coord - width, coord + width])
    cut = tf.clip_by_value(cut, 0, tf.shape(img, tf.int64)[0])
    new_img = tf.concat([img[:cut[0]],
                         tf.zeros_like(img[cut[0]:cut[1]]),
                         img[cut[1]:]], 0)
    return new_img


def power_to_db(img, ref=1.0, amin=1e-10, top_db=80.0):
    magnitude = img
    ref_value = tf.abs(ref)
    log_spec = 10.0 * (tf.math.log(tf.maximum(amin, magnitude)) / tf.math.log(10.))
    log_spec -= 10.0 * (tf.math.log(tf.maximum(amin, ref_value)) / tf.math.log(10.))
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    return log_spec


def normlize(img) -> tf.Tensor:
    img = power_to_db(img)
    img = (img - tf.reduce_mean(img)) / (tf.math.reduce_std(img) + 1e-7)
    return img


def mixup(imga, anna, imgb, annb) -> [tf.Tensor, tf.Tensor]:
    rate = tfp.distributions.Beta(1., 1.).sample([])
    img = imga * rate + imgb * (1 - rate)
    ann = tf.cast(anna, tf.float32) * rate + tf.cast(annb, tf.float32) * (1 - rate)
    return img, ann


class DCASETask5Helper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: list, nclasses: int):
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None
        if image_ann == None:
            self.train_list: str = None
            self.val_list: str = None
            self.test_list: str = None
            self.unlabel_list: str = None
        else:
            img_ann_list = np.load(image_ann, allow_pickle=True)[()]
            # NOTE can use dict set trian and test dataset
            self.unlabel_list: str = img_ann_list['train_labeled_data']
            self.train_list: str = img_ann_list['train_unlabeled_data']
            self.val_list: str = img_ann_list['val_data']
            self.test_list: str = None
            self.train_total_data: int = img_ann_list['train_labeled_num']
            self.unlabel_total_data: int = img_ann_list['train_unlabeled_num']
            self.val_total_data: int = img_ann_list['val_num']
            self.test_total_data: int = None
        self.in_hw: tf.Tensor = tf.constant(in_hw, tf.int64)
        self.nclasses = nclasses

    def build_train_datapipe(self, batch_size: int, is_augment: bool) -> tf.data.Dataset:
        def _pipe(stream: bytes):
            mel_raw, label = parser_example(stream)
            mel = decode_mel(mel_raw)
            if is_augment:
                mel = freqmask(mel)
            mel = tf.expand_dims(mel, -1)
            mel.set_shape((None, None, 1))
            return mel, label

        def _record_ds(x): return (tf.data.TFRecordDataset(x, num_parallel_reads=2).
                                   shuffle(batch_size * 30).
                                   repeat())

        ds = (tf.data.Dataset.from_tensor_slices([self.train_list, self.unlabel_list]).
              interleave(_record_ds, num_parallel_calls=-1).
              map(_pipe, -1).
              batch(batch_size, drop_remainder=True).
              prefetch(None))

        return ds

    def build_val_datapipe(self, batch_size: int) -> tf.data.Dataset:
        def _pipe(stream: bytes):
            mel_raw, label = parser_example(stream)
            mel = decode_mel(mel_raw)
            mel = tf.expand_dims(mel, -1)
            mel.set_shape((None, None, 1))
            return mel, label

        ds: tf.data.Dataset = (tf.data.TFRecordDataset(self.val_list, num_parallel_reads=2).
                               map(_pipe, num_parallel_calls=-1).
                               batch(batch_size, drop_remainder=True).
                               prefetch(None))
        return ds

    def set_dataset(self, batch_size, is_augment):
        self.batch_size = batch_size
        self.train_dataset = self.build_train_datapipe(batch_size, is_augment)
        self.val_dataset = self.build_val_datapipe(batch_size)
        self.train_epoch_step = self.train_total_data // batch_size
        self.val_epoch_step = self.val_total_data // self.batch_size


class Task5SupervisedLoop(BaseTrainingLoop):

    def set_metrics_dict(self):
        d = {
            'train': {
                'loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
                'acc': tf.keras.metrics.SparseCategoricalAccuracy('train_acc', dtype=tf.float32)},
            'val': {
                'loss': tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
                'acc': tf.keras.metrics.SparseCategoricalAccuracy('val_acc', dtype=tf.float32)}}
        return d

    @tf.function
    def train_step(self, iterator, num_steps_to_run, metrics):
        def step_fn(inputs):
            """Per-Replica training step function."""
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = self.train_model(images, training=True)
                # Loss calculations.
                #
                # Part 1: Prediction loss.
                loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
                loss_xe = tf.reduce_mean(loss_xe)
                # Part 2: Model weights regularization
                loss_wd = tf.reduce_sum(self.train_model.losses)
                loss = loss_xe + loss_wd
            grads = tape.gradient(loss, self.train_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_variables))
            metrics.loss.update_state(loss)
            metrics.acc.update_state(labels, tf.nn.softmax(logits))
        for _ in tf.range(num_steps_to_run):
            step_fn(next(iterator))

    @tf.function
    def val_step(self, dataset, metrics):
        def step_fn(inputs):
            """Per-Replica training step function."""
            images, labels = inputs
            logits = self.train_model(images, training=False)
            loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
            loss_xe = tf.reduce_mean(loss_xe)
            loss_wd = tf.reduce_sum(self.train_model.losses)
            loss = loss_xe + loss_wd
            metrics.loss.update_state(loss)
            metrics.acc.update_state(labels, tf.nn.softmax(logits))
        for inputs in dataset:
            step_fn(inputs)


class DCASETask5SSLHelper(DCASETask5Helper):
    def __init__(self, image_ann, in_hw, nclasses):
        super().__init__(image_ann, in_hw, nclasses)

    def build_train_datapipe(self, batch_size: int, is_augment: bool) -> tf.data.Dataset:
        def _pipe(stream: bytes):
            mel_raw, label = parser_example(stream)
            mel = decode_mel(mel_raw)
            mel = freqmask(mel)
            mel = normlize(mel)
            mel = tf.expand_dims(mel, -1)
            mel.set_shape((None, None, 1))
            return mel, label

        label_ds = (tf.data.TFRecordDataset(self.train_list, num_parallel_reads=2).
                    shuffle(batch_size * 30).
                    repeat().
                    map(_pipe, -1).
                    batch(batch_size, drop_remainder=True))
        unlabel_ds: tf.data.Dataset = (tf.data.TFRecordDataset(self.unlabel_list, num_parallel_reads=2).
                                       shuffle(batch_size * 30).
                                       repeat().
                                       map(_pipe, -1).
                                       batch(batch_size, drop_remainder=True))

        def dictlize(ds_label, ds_unlabel):
            label_img, label = ds_label
            unlabel_img, unlabel = ds_unlabel
            return {'label_img': label_img,
                    'unlabel_img': unlabel_img,
                    'label': tf.one_hot(tf.squeeze(label, -1), self.nclasses, axis=-1)}

        ds = (tf.data.Dataset.zip((label_ds, unlabel_ds)).
              map(dictlize, -1).prefetch(None))

        self.train_epoch_step = self.train_total_data // batch_size
        return ds

    def build_val_datapipe(self, batch_size: int, is_augment: bool) -> tf.data.Dataset:
        def _pipe(stream: bytes):
            mel_raw, label = parser_example(stream)
            mel = decode_mel(mel_raw)
            mel = normlize(mel)
            mel = tf.expand_dims(mel, -1)
            mel.set_shape((None, None, 1))
            return mel, label

        def dictlize(label_img, label):
            return {'label_img': label_img,
                    'unlabel_img': tf.zeros_like(label_img),
                    'label': tf.one_hot(tf.squeeze(label, -1), self.nclasses, axis=-1)}

        ds: tf.data.Dataset = (tf.data.TFRecordDataset(self.val_list, num_parallel_reads=2).
                               shuffle(batch_size * 30).
                               repeat().
                               map(_pipe, num_parallel_calls=-1).
                               batch(batch_size, drop_remainder=True).
                               map(dictlize))
        self.val_epoch_step = self.val_total_data // batch_size
        return ds
