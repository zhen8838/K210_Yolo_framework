import tensorflow as tf
import numpy as np
from tools.base import BaseHelper, INFO
from typing import Tuple
from tools.dcasetask5 import FixMatchSSLHelper
from tools.training_engine import BaseTrainingLoop, EmaHelper
from transforms.image.rand_augment import RandAugment
from transforms.image.ct_augment import CTAugment
import transforms.image.ops as image_ops


class KerasDatasetHelper(FixMatchSSLHelper, BaseHelper):

  def __init__(self, dataset: str, label_ratio: float, unlabel_dataset_ratio: int,
               augment_kwargs: dict):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None
    dataset_dict = {
        'mnist': tf.keras.datasets.mnist,
        'cifar10': tf.keras.datasets.cifar10,
        'cifar100': tf.keras.datasets.cifar100,
        'fashion_mnist': tf.keras.datasets.fashion_mnist
    }
    if dataset == None:
      self.train_list: str = None
      self.val_list: str = None
      self.test_list: str = None
      self.unlabel_list: str = None
    else:
      assert dataset in dataset_dict.keys(), 'dataset is invalid!'
      (x_train, y_train), (x_test, y_test) = dataset_dict[dataset].load_data()
      # NOTE can use dict set trian and test dataset
      y_train = y_train.ravel().astype('int32')
      y_test = y_test.ravel().astype('int32')
      label_set = set(y_train)
      label_idxs = []
      unlabel_idxs = []
      for l in label_set:
        idxes = np.where(y_train == l)[0]
        label_idxs.append(idxes[:int(len(idxes) * label_ratio)])
        unlabel_idxs.append(idxes[int(len(idxes) * label_ratio):])
      label_idxs = np.concatenate(label_idxs, 0)
      unlabel_idxs = np.concatenate(unlabel_idxs, 0)

      self.train_list: Tuple[np.ndarray, np.ndarray] = (x_train[label_idxs],
                                                        y_train[label_idxs])
      self.unlabel_list: Tuple[np.ndarray, np.ndarray] = (x_train[unlabel_idxs],
                                                          y_train[unlabel_idxs])
      self.val_list: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
      self.test_list: Tuple[np.ndarray, np.ndarray] = None
      self.train_total_data: int = len(label_idxs)
      self.unlabel_total_data: int = len(unlabel_idxs)
      self.val_total_data: int = len(x_test)
      self.test_total_data: int = None

    self.in_hw: list = list(x_train.shape[1:])
    self.nclasses = len(label_set)
    self.unlabel_dataset_ratio = unlabel_dataset_ratio
    tmp = self.create_augmenter(**augment_kwargs)
    self.augmenter: CTAugment = tmp[0]
    self.sup_aug_fn: callable = tmp[1]
    self.unsup_aug_fn: callable = tmp[2]

  @staticmethod
  def weak_aug_fn(data):
    """Augmentation which does random left-right flip and random shift of the image."""
    w = 4
    data = tf.image.random_flip_left_right(data)
    data_pad = tf.pad(data, [[w, w], [w, w], [0, 0]], mode='REFLECT')
    data = tf.image.random_crop(data_pad, tf.shape(data))
    return data

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
    print(INFO, f'Use {name}')

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:

    def label_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.sup_aug_fn(img)
      else:
        data_dict = {'data': img}
      # normalize image
      if is_normalize:
        data_dict = dict(
            map(lambda kv: (kv[0], image_ops.normalize(kv[1])),
                data_dict.items()))

      data_dict['label'] = label
      return data_dict

    def unlabel_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.unsup_aug_fn(img)
        data_dict['data'] = self.weak_aug_fn(data_dict['data'])
      else:
        data_dict = {'data': img}
      # normalize image
      if is_normalize:
        data_dict = dict(
            map(lambda kv: (kv[0], image_ops.normalize(kv[1])),
                data_dict.items()))

      data_dict['label'] = label
      return data_dict

    label_ds = tf.data.Dataset.from_tensor_slices(self.train_list).shuffle(
        batch_size * 300).repeat().map(label_pipe, -1).batch(
            batch_size, drop_remainder=True)
    unlabel_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        self.unlabel_list).shuffle(batch_size * 300).repeat().map(
            unlabel_pipe, -1).batch(
                batch_size * self.unlabel_dataset_ratio, drop_remainder=True)

    ds = tf.data.Dataset.zip((label_ds, unlabel_ds)).map(
        self._combine_sup_unsup_datasets).prefetch(tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    def _pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      data_dict = {'data': img}
      # normalize image
      if is_normalize:
        data_dict = dict(
            map(lambda kv: (kv[0], image_ops.normalize(kv[1])),
                data_dict.items()))

      data_dict['label'] = label
      return data_dict

    ds: tf.data.Dataset = (
        tf.data.Dataset.from_tensor_slices(self.val_list).map(
            _pipe, num_parallel_calls=-1).batch(batch_size).prefetch(None))
    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = False):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class UDASslLoop(BaseTrainingLoop):
  """ UDA
  
  hparams:
    nclasses: 10
    tsa: # 训练信号退火
      mode: no # Can choice from [linear,exp,log,no]
      scale: 5 # scale
      max_epoch: 256 # max update epoch
    uda:
      temperature: 1 # sharpening by temperature
      confidence: 0.95 # Ignore predictions < confidence
      wu: 1 # loss unlabeled weights
      we: 0 # loss entropy 

  """

  def set_augmenter(self, augmenter):
    if self.hparams.update_augmenter_state:
      self.augmenter: CTAugment = augmenter

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss':
                tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'acc':
                tf.keras.metrics.CategoricalAccuracy(
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

  def tsa_threshold(self) -> tf.Tensor:
    """ 训练信号退火阈值计算函数 """
    # step ratio will be maxed at tas_pos*train_epoch_step updates
    step_ratio = tf.cast(self.optimizer.iterations, tf.float32) / tf.cast(
        self.train_epoch_step * self.hparams.tsa.max_epoch, tf.float32)

    if self.hparams.tsa.mode == 'linear':
      coeff = step_ratio
    elif self.hparams.tsa.mode == 'exp':  # [exp(-5), exp(0)] = [1e-2, 1]
      coeff = tf.exp((step_ratio-1) * self.hparams.tsa.scale)
    elif self.hparams.tsa.mode == 'log':  # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
      coeff = 1 - tf.exp((-step_ratio) * self.hparams.tsa.scale)
    elif self.hparams.tsa.mode == 'no':
      coeff = tf.cast(1.0, tf.float32)
    elif self.hparams.tsa.mode != 'no':
      raise NotImplementedError(self.hparams.tsa.mode)
    coeff = tf.math.minimum(coeff, 1.0)  # bound the coefficient
    p_min = 1. / self.hparams.nclasses
    return coeff * (1-p_min) + p_min

  def tsa_loss_mask(self, labels, logits):
    """ 滤置信度高于训练信号退火阈值的对应样本损失 """
    thresh = self.tsa_threshold()
    p_class = tf.nn.softmax(logits, axis=-1)
    p_correct = tf.reduce_sum(labels * p_class, axis=-1)
    loss_mask = tf.cast(p_correct <= thresh,
                        tf.float32)  # Ignore confident predictions.
    return tf.stop_gradient(loss_mask)

  @staticmethod
  def softmax_temperature_controlling(logits: tf.Tensor, T: float) -> tf.Tensor:
    # this is essentially the same as sharpening in mixmatch
    logits = logits / T
    return tf.stop_gradient(logits)

  @staticmethod
  def confidence_based_masking(logits: tf.Tensor, p_class=None, thresh=0.9):
    if logits is not None:
      p_class = tf.nn.softmax(logits, axis=-1)
    p_class_max = tf.reduce_max(p_class, axis=-1)
    # Ignore unconfident predictions.
    loss_mask = tf.cast(p_class_max >= thresh, tf.float32)
    return tf.stop_gradient(loss_mask)

  @staticmethod
  def kl_divergence_from_logits(p_logits: tf.Tensor,
                                q_logits: tf.Tensor) -> tf.Tensor:
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)
    kl = tf.reduce_sum(p * (log_p-log_q), -1)
    return kl

  @staticmethod
  def entropy_from_logits(logits: tf.Tensor) -> tf.Tensor:
    log_prob = tf.nn.log_softmax(logits, axis=-1)
    prob = tf.exp(log_prob)
    ent = tf.reduce_sum(-prob * log_prob, axis=-1)
    return ent

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      sup_data = inputs['data']
      sup_label = inputs['label']
      unsup_data = inputs['unsup_data']
      unsup_aug_data = inputs['unsup_aug_data']
      with tf.GradientTape() as tape:
        logit_sup = self.train_model(sup_data, training=True)
        logit_unsup = self.train_model(unsup_data, training=True)
        logit_aug_unsup = self.train_model(unsup_aug_data, training=True)
        sup_label = tf.one_hot(sup_label, self.hparams.nclasses)
        
        # 生成伪标签
        logits_weak_tgt = self.softmax_temperature_controlling(
            logit_unsup, T=self.hparams.uda.temperature)
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logit_unsup))
        pseudo_mask = self.confidence_based_masking(
            logits=None,
            p_class=pseudo_labels,
            thresh=self.hparams.uda.confidence)

        # 锐化后的分布与未锐化的分布计算一致性损失
        kld = self.kl_divergence_from_logits(logits_weak_tgt, logit_aug_unsup)
        # 计算logits_weak的熵
        entropy = self.entropy_from_logits(logit_unsup)
        # 对一致性损失进行mask
        loss_xeu = tf.reduce_mean(kld * pseudo_mask)
        loss_ent = tf.reduce_mean(entropy)

        # 对于监督学习部分使用tsa进行mask
        loss_mask = self.tsa_loss_mask(labels=sup_label, logits=logit_sup)
        loss_xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=sup_label, logits=logit_sup)
        loss_xe = tf.reduce_sum(loss_xe * loss_mask) / tf.math.maximum(
            tf.reduce_sum(loss_mask), 1.0)

        # Model weights regularization
        loss_wd = tf.reduce_sum(self.train_model.losses)
        loss = loss_xe + loss_xeu * self.hparams.uda.wu + loss_ent * self.hparams.uda.we + loss_wd

      grads = tape.gradient(loss, self.train_model.trainable_variables)
      self.optimizer.apply_gradients(
          zip(grads, self.train_model.trainable_variables))

      if self.hparams.ema.enable:
        EmaHelper.update_ema_vars(self.val_model.variables,
                                  self.train_model.variables,
                                  self.hparams.ema.decay)

      if self.hparams.update_augmenter_state:
        if self.hparams.ema.enable and self.hparams.update_augmenter_state:
          probe_logits = self.val_model(inputs['probe_data'], training=False)
        else:
          probe_logits = self.train_model(inputs['probe_data'], training=False)
        probe_logits = tf.cast(probe_logits, tf.float32)
        self.augmenter.update(inputs, tf.nn.softmax(probe_logits))

      metrics.loss.update_state(loss)
      metrics.acc.update_state(sup_label, logit_sup)

    for _ in tf.range(num_steps_to_run):
      step_fn(next(iterator))

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
      step_fn(inputs)
