""" dcase2018 task5 helper functions """
from tools.base import BaseHelper
import tensorflow as tf
from transforms.audio.ops import freq_mask
from transforms.audio.rand_augment import RandAugment
from transforms.audio.ct_augment import CTAugment
from tools.training_engine import BaseTrainingLoop, EmaHelper
import numpy as np


def parser_example(stream: bytes) -> [tf.Tensor, tf.Tensor]:
  example = tf.io.parse_single_example(
      stream, {
          'mel_raw': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.int64)
      })
  return example['mel_raw'], example['label']


class DCASETask5Helper(BaseHelper):

  def __init__(self, data_ann: str, in_hw: list, nclasses: int):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None
    if data_ann == None:
      self.train_list: str = None
      self.val_list: str = None
      self.test_list: str = None
      self.unlabel_list: str = None
    else:
      data_ann_list = np.load(data_ann, allow_pickle=True)[()]
      # NOTE can use dict set trian and test dataset
      self.train_list: str = data_ann_list['train_labeled_data']
      self.unlabel_list: str = data_ann_list['train_unlabeled_data']
      self.val_list: str = data_ann_list['val_data']
      self.test_list: str = None
      self.train_total_data: int = data_ann_list['train_labeled_num']
      self.unlabel_total_data: int = data_ann_list['train_unlabeled_num']
      self.val_total_data: int = data_ann_list['val_num']
      self.test_total_data: int = None
    self.in_hw: list = list(in_hw)
    self.nclasses = nclasses

  def build_train_datapipe(self, batch_size: int,
                           is_augment: bool) -> tf.data.Dataset:

    def _pipe(stream: bytes):
      mel_raw, label = parser_example(stream)
      mel = tf.io.parse_tensor(mel_raw, tf.float32)
      mel.set_shape(self.in_hw)
      if is_augment:
        mel = freq_mask(mel)
      mel = tf.expand_dims(mel, -1)
      return {'data': mel, 'label': label}

    ds = tf.data.TFRecordDataset([self.train_list], num_parallel_reads=2).shuffle(
        batch_size * 300).repeat().map(_pipe, -1).batch(
            batch_size, drop_remainder=True).prefetch(-1)

    return ds

  def build_val_datapipe(self, batch_size: int) -> tf.data.Dataset:

    def _pipe(stream: bytes):
      mel_raw, label = parser_example(stream)
      mel = tf.io.parse_tensor(mel_raw, tf.float32)
      mel.set_shape(self.in_hw)
      mel = tf.expand_dims(mel, -1)
      return {'data': mel, 'label': label}

    ds: tf.data.Dataset = (
        tf.data.TFRecordDataset(self.val_list, num_parallel_reads=2).map(
            _pipe,
            num_parallel_calls=-1).batch(batch_size,
                                         drop_remainder=True).prefetch(None))
    return ds

  def set_dataset(self, batch_size, is_augment):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment)
    self.val_dataset = self.build_val_datapipe(batch_size)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class Task5SupervisedLoop(BaseTrainingLoop):

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss':
                tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'acc':
                tf.keras.metrics.SparseCategoricalAccuracy(
                    'train_acc', dtype=tf.float32)
        },
        'val': {
            'loss':
                tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'acc':
                tf.keras.metrics.SparseCategoricalAccuracy(
                    'val_acc', dtype=tf.float32)
        }
    }
    return d

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      datas, labels = inputs['data'], inputs['label']
      with tf.GradientTape() as tape:
        logits = self.train_model(datas, training=True)
        # Loss calculations.
        #
        # Part 1: Prediction loss.
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss_xe = tf.reduce_mean(loss_xe)
        # Part 2: Model weights regularization
        loss_wd = tf.reduce_sum(self.train_model.losses)
        loss = loss_xe + loss_wd
      scaled_loss = self.optimizer_minimize(loss, tape, self.optimizer,
                                            self.train_model)
      if self.hparams.ema.enable:
        self.ema.update()
      metrics.loss.update_state(scaled_loss)
      metrics.acc.update_state(labels, tf.nn.softmax(logits))

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      datas, labels = inputs['data'], inputs['label']
      logits = self.val_model(datas, training=False)
      loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
      loss_xe = tf.reduce_mean(loss_xe)
      loss_wd = tf.reduce_sum(self.val_model.losses)
      loss = loss_xe + loss_wd
      metrics.loss.update_state(loss)
      metrics.acc.update_state(labels, tf.nn.softmax(logits))

    for inputs in dataset:
      self.strategy.experimental_run_v2(step_fn, args=(inputs,))


class FixMatchSSLHelper(object):

  @staticmethod
  def create_augmenter(name: str, kwarg: dict):
    if not name or (name == 'none') or (name == 'noop'):
      return (lambda x: x)
    elif name == 'randaugment':
      base_augmenter = RandAugment(**kwarg)
      return (None, lambda data: {
          'data': data
      }, lambda x: base_augmenter(x, aug_key='aug_data'))
    elif name == 'ctaugment':
      base_augmenter = CTAugment(**kwarg)
      return (base_augmenter,
              lambda x: base_augmenter(x, probe=True, aug_key=None),
              lambda x: base_augmenter(x, probe=False, aug_key='aug_data'))
    else:
      raise ValueError('Invalid augmentation type {0}'.format(name))

  @staticmethod
  def _combine_sup_unsup_datasets(sup_data: dict, unsup_data: dict) -> dict:
    """Combines supervised and usupervised samples into single dictionary.
    
    Args:
        sup_data (dict): dictionary with examples from supervised dataset.
        unsup_data (dict):  dictionary with examples from unsupervised dataset.
    
    Returns:
        dict: combined suvervised and unsupervised examples.
    """
    # Copy all values from supervised data as is
    output_dict = dict(sup_data)

    # take only 'data' and 'aug_data' from unsupervised dataset and
    # rename then into 'unsup_data' and 'unsup_aug_data'
    if 'data' in unsup_data:
      output_dict['unsup_data'] = unsup_data.pop('data')
    if 'aug_data' in unsup_data:
      output_dict['unsup_aug_data'] = unsup_data.pop('aug_data')
    if 'label' in unsup_data:
      output_dict['unsup_label'] = unsup_data.pop('label')

    return output_dict


class DCASETask5FixMatchSSLHelper(DCASETask5Helper, FixMatchSSLHelper):

  def __init__(self, data_ann, in_hw, nclasses, unlabel_dataset_ratio: int,
               augment_kwargs: dict):
    super().__init__(data_ann, in_hw, nclasses)
    self.unlabel_dataset_ratio = unlabel_dataset_ratio
    tmp = self.create_augmenter(**augment_kwargs)
    self.augmenter: CTAugment = tmp[0]
    self.sup_aug_fn = tmp[1]
    self.unsup_aug_fn = tmp[2]

  def build_train_datapipe(self, batch_size: int,
                           is_augment: bool) -> tf.data.Dataset:

    def label_pipe(stream: bytes):
      mel_raw, label = parser_example(stream)
      mel = tf.io.parse_tensor(mel_raw, tf.float32)
      mel.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.sup_aug_fn(mel)
      else:
        data_dict = {'data': mel}

      for k in data_dict.keys():
        if 'data' in k:
          data_dict[k] = tf.expand_dims(data_dict[k], -1)

      data_dict['label'] = label
      return data_dict

    def unlabel_pipe(stream: bytes):
      mel_raw, label = parser_example(stream)
      mel: tf.Tensor = tf.io.parse_tensor(mel_raw, tf.float32)
      mel.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.unsup_aug_fn(mel)
      else:
        data_dict = {'data': mel}

      for k in data_dict.keys():
        if 'data' in k:
          data_dict[k] = tf.expand_dims(data_dict[k], -1)

      data_dict['label'] = label
      return data_dict

    label_ds = tf.data.TFRecordDataset(
        self.train_list,
        num_parallel_reads=2).shuffle(batch_size * 100).repeat().map(
            label_pipe, -1).batch(
                batch_size, drop_remainder=True)
    unlabel_ds: tf.data.Dataset = tf.data.TFRecordDataset(
        self.unlabel_list,
        num_parallel_reads=2).shuffle(batch_size * 100).repeat().map(
            unlabel_pipe, -1).batch(
                batch_size * self.unlabel_dataset_ratio, drop_remainder=True)

    ds = tf.data.Dataset.zip((label_ds, unlabel_ds)).map(
        self._combine_sup_unsup_datasets).prefetch(tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds


class AugmenterStateSync(tf.keras.callbacks.Callback):

  def __init__(self,
               augmenter: CTAugment,
               update_augmenter_state: bool,
               verbose: bool = False):
    super().__init__()
    self.augmenter = augmenter
    self.update_augmenter_state = update_augmenter_state
    self.verbose = verbose

  def on_epoch_begin(self, epoch, logs=None):
    self.augmenter.sync_state()
    augmenter_state = self.augmenter.get_state()
    augmenter_state = [
        f'{k}: {v.numpy()}\n' for (k, v) in augmenter_state.items()
    ]
    if self.verbose:
      print('Augmenter state:\n', '\n'.join(augmenter_state))


class Task5FixMatchSslLoop(BaseTrainingLoop):

  def set_augmenter(self, augmenter):
    if self.hparams.update_augmenter_state:
      self.augmenter: CTAugment = augmenter

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss':
                tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'acc':
                tf.keras.metrics.SparseCategoricalAccuracy(
                    'train_acc', dtype=tf.float32)
        },
        'val': {
            'loss':
                tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'acc':
                tf.keras.metrics.SparseCategoricalAccuracy(
                    'val_acc', dtype=tf.float32)
        }
    }
    return d

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      sup_data = inputs['data']
      sup_label = inputs['label']
      unsup_data = inputs['unsup_data']
      unsup_aug_data = inputs['unsup_aug_data']
      with tf.GradientTape() as tape:
        # batch = tf.shape(sup_label)[0]
        # logits = self.train_model(
        #     tf.concat([sup_data, unsup_data, unsup_aug_data], 0), training=True)
        # logit_sup = logits[:batch]
        # logit_unsup, logit_aug_unsup = tf.split(logits[batch:], 2)
        logit_sup = self.train_model(sup_data, training=True)
        logit_unsup = self.train_model(unsup_data, training=True)
        logit_aug_unsup = self.train_model(unsup_aug_data, training=True)

        # Supervised loss
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sup_label, logits=logit_sup)
        loss_xe = tf.reduce_mean(loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logit_unsup))
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(pseudo_labels, axis=1), logits=logit_aug_unsup)
        pseudo_mask = (
            tf.reduce_max(pseudo_labels, axis=1) >=
            self.hparams.fixmatch.confidence)
        pseudo_mask = tf.cast(pseudo_mask, tf.float32)
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

        # Model weights regularization
        loss_wd = tf.reduce_sum(self.train_model.losses)

        loss = loss_xe + self.hparams.fixmatch.wu * loss_xeu + loss_wd

      scaled_loss = self.optimizer_minimize(loss, tape, self.optimizer,
                                            self.train_model)

      if self.hparams.ema.enable:
        self.ema.update()

      if self.hparams.update_augmenter_state:
        if self.hparams.ema.enable and self.hparams.update_augmenter_state:
          probe_logits = self.ema.model(inputs['probe_data'], training=False)
        else:
          probe_logits = self.train_model(inputs['probe_data'], training=False)
        probe_logits = tf.cast(probe_logits, tf.float32)
        self.augmenter.update(inputs, tf.nn.softmax(probe_logits))

      metrics.loss.update_state(scaled_loss)
      metrics.acc.update_state(sup_label, logit_sup)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      datas, labels = inputs['data'], inputs['label']
      logits = self.val_model(datas, training=False)
      loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
      loss_xe = tf.reduce_mean(loss_xe)
      loss_wd = tf.reduce_sum(self.val_model.losses)
      loss = loss_xe + loss_wd
      metrics.loss.update_state(loss)
      metrics.acc.update_state(labels, logits)

    for inputs in dataset:
      self.strategy.experimental_run_v2(step_fn, args=(inputs,))
