import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tools.base import BaseHelper, INFO
from typing import Tuple, List
from tools.dcasetask5 import FixMatchSSLHelper
from tools.training_engine import BaseTrainingLoop, EmaHelper
from transforms.image.rand_augment import RandAugment
from transforms.image.ct_augment import CTAugment
import transforms.image.ops as image_ops
from functools import partial


class KerasDatasetSemiHelper(FixMatchSSLHelper, BaseHelper):
  """ KerasDatasetSemiHelper

  Args:
      dataset: "cifar10"
      label_ratio: 0.05 # Unlabeled sample ratio.
      unlabel_dataset_ratio: 7 # Unlabeled batch size ratio.
      augment_kwargs:
        name: ctaugment
        kwarg:
          num_layers: 3
          confidence_threshold: 0.8
          decay: 0.99
          epsilon: 0.001
          prob_to_apply: null
          num_levels: 10
      augment_mode: augment_anchor
  """

  def __init__(self,
               dataset: str,
               label_ratio: float,
               unlabel_dataset_ratio: int,
               augment_kwargs: dict,
               augment_mode='augment_anchor',
               mixed_precision_dtype='float32'):
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

      # Make data More randomal
      np.random.shuffle(unlabel_idxs)
      np.random.shuffle(label_idxs)

      self.train_list: Tuple[np.ndarray, np.ndarray] = (x_train[label_idxs],
                                                        y_train[label_idxs])
      self.unlabel_list: Tuple[np.ndarray, np.ndarray] = (x_train[unlabel_idxs],
                                                          y_train[unlabel_idxs])
      self.val_list: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
      self.test_list: Tuple[np.ndarray, np.ndarray] = None
      self.train_total_data: int = len(unlabel_idxs)
      self.unlabel_total_data: int = len(unlabel_idxs)
      self.val_total_data: int = len(x_test)
      self.test_total_data: int = None

    self.in_hw: list = list(x_train.shape[1:])
    self.nclasses = len(label_set)
    self.unlabel_dataset_ratio = unlabel_dataset_ratio
    self.mixed_precision_dtype = mixed_precision_dtype
    assert augment_mode in ['augment_anchor', 'k_augment']
    self.augment_mode = augment_mode
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

      def fn(data, aug_key='data'):
        output_dict = {}
        if aug_key is not None:
          output_dict[aug_key] = data
        if aug_key != 'data':
          output_dict['data'] = data
        return output_dict

      return (
          None,
          partial(fn, aug_key='data'),
          partial(fn, aug_key='aug_data'),
      )

    elif name == 'randaugment':

      base_augmenter = RandAugment(**kwarg)
      return (
          None,
          lambda data: {
              'data': data
          },
          partial(base_augmenter, aug_key='aug_data'),
      )
    elif name == 'ctaugment':
      base_augmenter = CTAugment(**kwarg)
      return (
          base_augmenter,
          partial(base_augmenter, probe=True, aug_key=None),
          partial(base_augmenter, probe=False, aug_key='aug_data'),
      )
    else:
      raise ValueError('Invalid augmentation type {0}'.format(name))
    print(INFO, f'Use {name}')

  def normlize(self, data_dict: dict):
    for key, v in data_dict.items():
      if key.endswith('data'):
        v = tf.cast(v, self.mixed_precision_dtype)
        v = image_ops.normalize(v, tf.constant(127.5, self.mixed_precision_dtype),
                                tf.constant(127.5, self.mixed_precision_dtype))
        data_dict[key] = v

  def build_augment_anchor_train_datapipe(self,
                                          batch_size: int,
                                          is_augment: bool,
                                          is_normalize: bool = True
                                          ) -> tf.data.Dataset:
    """ 构建用于augment anchor损失的数据管道
    NOTE: 使用这个方式时，unlabel_dataset_ratio即无标签数据的batch倍数

    Args:
        batch_size (int): 
        is_augment (bool): 
        is_normalize (bool, optional): . Defaults to True.

    Returns:
        tf.data.Dataset: ds
    """

    def label_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.sup_aug_fn(img)
      else:
        data_dict = {'data': img}
      # normalize image
      if is_normalize:
        self.normlize(data_dict)

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
        self.normlize(data_dict)

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

  def build_k_augment_train_datapipe(self,
                                     batch_size: int,
                                     is_augment: bool,
                                     is_normalize: bool = True
                                     ) -> tf.data.Dataset:
    """ 构建用于k个增强的数据管道
    NOTE: 使用这个方式时，unlabel_dataset_ratio即k个增强

    Args:
        batch_size (int): 
        is_augment (bool): 
        is_normalize (bool, optional): . Defaults to True.

    Returns:
        tf.data.Dataset: ds
    """

    def label_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:
        data_dict = self.sup_aug_fn(img)
      else:
        data_dict = {'data': img}
      # normalize image
      if is_normalize:
        self.normlize(data_dict)

      data_dict['label'] = label
      return data_dict

    def unlabel_pipe(img: tf.Tensor, label: tf.Tensor):
      img.set_shape(self.in_hw)
      if is_augment:

        k_aug_img = tf.stack([
            self.unsup_aug_fn(img)['aug_data']
            for _ in range(self.unlabel_dataset_ratio)
        ])
        # k_aug_img shape = [self.unlabel_dataset_ratio,h,w,c]
        weak_aug_img = self.weak_aug_fn(img)
        # weak_aug_img shape = [h,w,c]

        data_dict = {'data': weak_aug_img, 'aug_data': k_aug_img}
      else:
        data_dict = {'data': img}

      # normalize image
      if is_normalize:
        self.normlize(data_dict)

      data_dict['label'] = label
      return data_dict

    label_ds = tf.data.Dataset.from_tensor_slices(self.train_list).shuffle(
        batch_size * 300).repeat().map(label_pipe, -1).batch(
            batch_size, drop_remainder=True)
    unlabel_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        self.unlabel_list).shuffle(batch_size * 300).repeat().map(
            unlabel_pipe, -1).batch(
                batch_size, drop_remainder=True)

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
        self.normlize(data_dict)

      data_dict['label'] = label
      return data_dict

    ds: tf.data.Dataset = (
        tf.data.Dataset.from_tensor_slices(self.val_list).map(
            _pipe, num_parallel_calls=-1).batch(batch_size).prefetch(None))
    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    if self.augment_mode == 'k_augment':
      self.train_dataset = self.build_k_augment_train_datapipe(
          batch_size, is_augment, is_normalize)
    elif self.augment_mode == 'augment_anchor':
      self.train_dataset = self.build_augment_anchor_train_datapipe(
          batch_size, is_augment, is_normalize)

    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class FixMatchMixUpSslLoop(BaseTrainingLoop):
  """ FixMatch中用类似ICT中的方式给伪标签实施MixUp

  hparams:
    nclasses: 10
    fixmatchmixup:
      confidence: 0.95 # 伪标签样本损失置信度过滤阈值
      wu: 1.0 # 无标签样本伪标签损失权重
      wmu: 1.0 # 无标签样本mixup后伪标签损失权重
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

  @staticmethod
  def array_shuflle(n: int, beta: float) -> [tf.Tensor, tf.Tensor]:
    """ get shuffle array

    Args:
        n (int): lens
        beta (float): beta

    Returns:
        mix (tf.Tensor): shape [len]
        index (tf.Tensor): shape [len]
    """
    mix = tfp.distributions.Beta(beta, beta).sample([n])
    mix = tf.expand_dims(tf.maximum(mix, 1 - mix), -1)
    index = tf.random.shuffle(tf.range(n))
    return mix, index

  @staticmethod
  def apply_mixup_one(a: tf.Tensor, b: tf.Tensor, mix: tf.Tensor,
                      index: tf.Tensor) -> tf.Tensor:
    bs = tf.gather(b, index)
    # reshape mix for broadcast
    mix = tf.reshape(mix, [-1] + [1] * (len(a.shape.as_list()) - 1))
    mixed = a * mix + bs * (1 - mix)
    return mixed

  @staticmethod
  def apply_mixup(x0: tf.Tensor, x1: tf.Tensor, l0: tf.Tensor, l1: tf.Tensor,
                  mix: tf.Tensor, index: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    xmix = FixMatchMixUpSslLoop.apply_mixup_one(x0, x1, mix, index)
    lmix = FixMatchMixUpSslLoop.apply_mixup_one(l0, l1, mix, index)
    return xmix, lmix

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      sup_data = inputs['data']
      sup_label = inputs['label']
      unsup_data = inputs['unsup_data']
      unsup_aug_data = inputs['unsup_aug_data']
      unsup_batch = tf.shape(unsup_data)[0]
      with tf.GradientTape() as tape:
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
            self.hparams.fixmatchmixup.confidence)
        pseudo_mask = tf.cast(pseudo_mask, tf.float32)
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

        # Pseudo-label MixUped cross entropy for unlabeled data
        # NOTE 当使用augment anchor时，unsup_data对应着unsup_aug_data
        mix, index = self.array_shuflle(unsup_batch, 0.5)
        mix_pseudo_labels = tf.argmax(pseudo_labels, axis=1)
        mix_pseudo_labels = tf.one_hot(mix_pseudo_labels, self.hparams.nclasses)
        mix_pseudo_labels = self.apply_mixup_one(mix_pseudo_labels,
                                                 mix_pseudo_labels, mix, index)
        mix_unsup_aug_data = self.apply_mixup_one(unsup_aug_data, unsup_aug_data,
                                                  mix, index)
        mix_logit_aug_unsup = self.train_model(unsup_aug_data, training=True)
        loss_xeu_mix = tf.nn.softmax_cross_entropy_with_logits(
            labels=mix_pseudo_labels, logits=mix_logit_aug_unsup)
        loss_xeu_mix = tf.reduce_mean(loss_xeu_mix)

        # Model weights regularization
        loss_wd = tf.reduce_sum(self.train_model.losses)

        loss = (
            loss_xe + self.hparams.fixmatchmixup.wu * loss_xeu +
            self.hparams.fixmatchmixup.wmu * loss_xeu_mix + loss_wd)

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
      self.run_step_fn(step_fn, args=(next(iterator),))

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
      self.run_step_fn(step_fn, args=(inputs,))


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
      coeff = tf.exp((step_ratio - 1) * self.hparams.tsa.scale)
    elif self.hparams.tsa.mode == 'log':  # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
      coeff = 1 - tf.exp((-step_ratio) * self.hparams.tsa.scale)
    elif self.hparams.tsa.mode == 'no':
      coeff = tf.cast(1.0, tf.float32)
    elif self.hparams.tsa.mode != 'no':
      raise NotImplementedError(self.hparams.tsa.mode)
    coeff = tf.math.minimum(coeff, 1.0)  # bound the coefficient
    p_min = 1. / self.hparams.nclasses
    return coeff * (1 - p_min) + p_min

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
    kl = tf.reduce_sum(p * (log_p - log_q), -1)
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
      self.run_step_fn(step_fn, args=(next(iterator),))

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
      self.run_step_fn(step_fn, args=(inputs,))


class PMovingAverage(object):

  def __init__(self, name, nclass, buf_size):
    # MEAN aggregation is used by DistributionStrategy to aggregate
    # variable updates across shards
    self.ma = tf.Variable(
        tf.ones([buf_size, nclass]) / nclass,
        trainable=False,
        name=name,
        aggregation=tf.VariableAggregation.MEAN)

  def __call__(self):
    v = tf.reduce_mean(self.ma, axis=0)
    return v / tf.reduce_sum(v)

  def update(self, entry):
    entry = tf.reduce_mean(entry, axis=0)
    return tf.assign(self.ma, tf.concat([self.ma[1:], [entry]], axis=0))


class PData(object):

  def __init__(self, name, nclass):
    self.p_data = tf.Variable(
        self.renorm(tf.ones([nclass])),
        trainable=False,
        name=name,
        aggregation=tf.VariableAggregation.MEAN)
    self.has_update = True

  @staticmethod
  def renorm(v):
    return v / tf.reduce_sum(v, axis=-1, keepdims=True)

  def __call__(self):
    return self.p_data / tf.reduce_sum(self.p_data)

  def update(self, entry, decay=0.999):
    entry = tf.reduce_mean(entry, axis=0)
    return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))


class MixMode(object):
  # A class for mixing data for various combination of labeled and unlabeled.
  # x = labeled example
  # y = unlabeled example
  # For example "xx.yxy" means: mix x with x, mix y with both x and y.
  MODES = 'xx.yy xxy.yxy xx.yxy xx.yx xx. .yy xxy. .yxy .'.split()

  def __init__(self, mode):
    assert mode in self.MODES
    self.mode = mode

  @staticmethod
  def augment_pair(x0, l0, x1, l1, beta):
    """ beta must >= 0 """
    mix = tfp.distributions.Beta(beta, beta).sample([tf.shape(x0)[0]])
    mix = tf.reshape(tf.maximum(mix, 1 - mix), [tf.shape(x0)[0], 1, 1, 1])
    index = tf.random.shuffle(tf.range(tf.shape(x0)[0]))
    xs = tf.gather(x1, index)
    ls = tf.gather(l1, index)
    xmix = x0 * mix + xs * (1 - mix)
    lmix = l0 * mix[:, :, 0, 0] + ls * (1 - mix[:, :, 0, 0])
    return xmix, lmix

  @staticmethod
  def augment(x, l, beta):
    return MixMode.augment_pair(x, l, x, l, beta)

  def __call__(self, xl: list, ll: list, betal: list):
    assert len(xl) == len(ll) >= 2
    assert len(betal) == 2
    if self.mode == '.':
      return xl, ll
    elif self.mode == 'xx.':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      return [mx0] + xl[1:], [ml0] + ll[1:]
    elif self.mode == '.yy':
      mx1, ml1 = self.augment(
          tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
      return (xl[:1] + tf.split(mx1,
                                len(xl) - 1), ll[:1] + tf.split(ml1,
                                                                len(ll) - 1))
    elif self.mode == 'xx.yy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(
          tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
      return ([mx0] + tf.split(mx1,
                               len(xl) - 1), [ml0] + tf.split(ml1,
                                                              len(ll) - 1))
    elif self.mode == 'xxy.':
      mx, ml = self.augment(
          tf.concat(xl, 0), tf.concat(ll, 0),
          sum(betal) / len(betal))
      return (tf.split(mx, len(xl))[:1] + xl[1:],
              tf.split(ml, len(ll))[:1] + ll[1:])
    elif self.mode == '.yxy':
      mx, ml = self.augment(
          tf.concat(xl, 0), tf.concat(ll, 0),
          sum(betal) / len(betal))
      return (xl[:1] + tf.split(mx, len(xl))[1:],
              ll[:1] + tf.split(ml, len(ll))[1:])
    elif self.mode == 'xxy.yxy':
      mx, ml = self.augment(
          tf.concat(xl, 0), tf.concat(ll, 0),
          sum(betal) / len(betal))
      return tf.split(mx, len(xl)), tf.split(ml, len(ll))
    elif self.mode == 'xx.yxy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), betal[1])
      mx1, ml1 = [tf.split(m, len(xl))[1:] for m in (mx1, ml1)]
      return [mx0] + mx1, [ml0] + ml1
    elif self.mode == 'xx.yx':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = zip(*[
          self.augment_pair(xl[i], ll[i], xl[0], ll[0], betal[1])
          for i in range(1, len(xl))
      ])
      return [mx0] + list(mx1), [ml0] + list(ml1)
    raise NotImplementedError(self.mode)


class MixMatchSslLoop(BaseTrainingLoop):
  """ MixMatch

  hparams:
    nclasses: 10
    mixmatch:
      beta: 0.5 # Mixup beta distribution.
      w_match: 100 # Weight for distribution matching loss.
      nu: 2 # augment num
      mixmode: "xxy.yxy" # the mix mode
      dbuf: 128 # label distribution moving average buffer size
      T: 0.5 # sharping tempeture
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
    # 当前估计标签分布的移动平均值
    self.p_model = PMovingAverage('p_model', self.hparams.nclasses,
                                  self.hparams.mixmatch.dbuf)
    # 校正分布（仅用于绘图）
    self.p_target = PMovingAverage('p_target', self.hparams.nclasses,
                                   self.hparams.mixmatch.dbuf)
    # 无标签数据的分布推理
    self.p_data = PData('p_data', self.hparams.nclasses)
    # 设置mixup的模式，默认是标记数据会与(标记数据，无标记数据)混合，无标记数据会与(标记数据，无标记数据)混合
    self.mix_augment = MixMode(self.hparams.mixmatch.mixmode)
    return d

  def guess_label(self, y: tf.Tensor, classifier: tf.keras.Model):
    logits_y = [classifier(yi, training=True) for yi in y]
    logits_y = tf.concat(logits_y, 0)
    # Compute predicted probability distribution py.
    # p_model_y shape = [K,batch,calss_num]
    p_model_y = tf.reshape(
        tf.nn.softmax(logits_y), [len(y), -1, self.hparams.nclasses])
    # 求均值
    p_model_y = tf.reduce_mean(p_model_y, axis=0)
    # 锐化
    p_target = tf.pow(p_model_y, 1. / self.hparams.mixmatch.T)
    p_target /= tf.reduce_sum(p_target, axis=1, keepdims=True)
    return p_target, p_model_y

  @staticmethod
  def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
      groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
      offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

  @staticmethod
  def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = MixMatchSslLoop.interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
      xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      xt_in: tf.Tensor = inputs['data']
      l_in: tf.Tensor = inputs['label']
      y_in: tf.Tensor = inputs['unsup_aug_data']
      xt_in = tf.cast(xt_in, tf.float32)
      y_in = tf.cast(y_in, tf.float32)
      batch, *hwc = xt_in.shape.as_list()
      with tf.GradientTape() as tape:
        # 计算猜测标签
        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess_p_target, guess_p_model = self.guess_label(
            tf.split(y, self.hparams.mixmatch.nu), self.train_model)
        ly = tf.stop_gradient(guess_p_target)  # 取消梯度
        lx = tf.one_hot(l_in, self.hparams.nclasses)
        xy, labels_xy = self.mix_augment(
            [xt_in] + tf.split(y, self.hparams.mixmatch.nu),
            [lx] + [ly] * self.hparams.mixmatch.nu,
            [self.hparams.mixmatch.beta, self.hparams.mixmatch.beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        batches = self.interleave([x] + y, batch)
        logits = [self.train_model(batchi, training=True) for batchi in batches]

        logits = self.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        # 交叉熵
        loss_xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)

        # 一致正则熵
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)

        loss_wd = tf.reduce_mean(self.train_model.losses)

        loss = loss_xe + loss_l2u * self.hparams.mixmatch.w_match + loss_wd

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
      metrics.acc.update_state(lx, tf.nn.softmax(logits_x))

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

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
      self.run_step_fn(step_fn, args=(inputs,))


class InfoMaxLoop(BaseTrainingLoop):
  """ InfoMax self-supervised learning

  Args:
      update_augmenter_state: *UPDATE_AUGMENTER_STATE
      infomax:
        wkl: 0.01 # weight for kl prior loss
        wginfo: 0.5 # weight for global information loss
        wlinfo: 1.5 # weight for local information loss
      ema:
        enable: false
        decay: 0.999
  """

  def set_augmenter(self, augmenter):
    if self.hparams.update_augmenter_state:
      self.augmenter: CTAugment = augmenter

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'kl_loss': tf.keras.metrics.Mean('kl_loss', dtype=tf.float32),
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'l_loss': tf.keras.metrics.Mean('l_loss', dtype=tf.float32),
        },
        'val': {}
    }
    self.encoder: tf.keras.Model = tf.keras.backend.function(
        self.val_model.input,
        [self.val_model.get_layer(name='normal_mean').output])
    return d

  @staticmethod
  def prior_kl_loss(mean, log_sigma):
    loss = -0.5 * tf.reduce_mean(1 + log_sigma - tf.square(mean) -
                                 tf.exp(log_sigma))
    return loss

  @staticmethod
  def info_loss(
      true_scores: tf.Tensor,
      false_scores: tf.Tensor,
      false_label: tf.Tensor,
  ) -> tf.Tensor:
    """ information loss

    Args:
        true_scores (tf.Tensor): shape [batch , ?] 
        false_scores (tf.Tensor): shape [batch , ?] 
        false_label (tf.Tensor): shape [batch , ?] NOTE: not equal is 1, else 0

    Returns:
        tf.Tensor: loss, shape = scalar
    """
    # yapf: disable
    loss = -tf.reduce_mean(
        tf.math.log(true_scores + 1e-6) +
        tf.math.log(1 - false_scores + 1e-6) * false_label +
        tf.math.log(false_scores + 1e-6) * (1 - false_label))
    # yapf: enable
    return loss

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      unsup_data = inputs['unsup_data']
      with tf.GradientTape() as tape:
        (
            unsup_logits,
            unsup_z_mean,
            unsup_z_log_sigma,
            unsup_zz_true_scores,
            unsup_zz_false_scores,
            unsup_zz_label,
            unsup_zf_true_scores,
            unsup_zf_false_scores,
            unsup_zf_label,
        ) = self.train_model(
            unsup_data, training=True)

        # prior kl loss
        unsup_kl_loss = self.prior_kl_loss(unsup_z_mean, unsup_z_log_sigma)

        # information loss
        unsup_zz_loss = self.info_loss(unsup_zz_true_scores,
                                       unsup_zz_false_scores, unsup_zz_label)
        unsup_zf_loss = self.info_loss(unsup_zf_true_scores,
                                       unsup_zf_false_scores, unsup_zf_label)

        # Model weights regularization
        wd_loss = tf.reduce_mean(self.train_model.losses)

        # yapf: disable
        loss = (self.hparams.infomax.wkl * unsup_kl_loss +
                self.hparams.infomax.wginfo * unsup_zz_loss +
                self.hparams.infomax.wlinfo * unsup_zf_loss +
                wd_loss)
        # yapf: enable

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
      metrics.kl_loss.update_state(unsup_kl_loss)
      metrics.g_loss.update_state(unsup_zz_loss)
      metrics.l_loss.update_state(unsup_zf_loss)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  @staticmethod
  def sample_knn(zs: np.ndarray, img_list: np.ndarray):
    img_h, img_w = img_list.shape[1:-1]
    n = 10
    topn = 10
    figure1 = np.zeros((img_h * n, img_w * topn, 3))
    figure2 = np.zeros((img_h * n, img_w * topn, 3))
    zs_ = zs / (zs**2).sum(1, keepdims=True)**0.5
    for i in range(n):
      one = np.random.choice(len(img_list))
      idxs = ((zs**2).sum(1) + (zs[one]**2).sum() -
              2 * np.dot(zs, zs[one])).argsort()[:topn]
      for j, k in enumerate(idxs):
        digit = img_list[k]
        figure1[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = digit
      idxs = np.dot(zs_, zs_[one]).argsort()[-n:][::-1]
      for j, k in enumerate(idxs):
        digit = img_list[k]
        figure2[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = digit
    figure1 = figure1.astype('uint8')
    figure2 = figure2.astype('uint8')
    return figure1, figure2

  def val_step(self, dataset, metrics):
    with tf.keras.backend.learning_phase_scope(0):
      zs_list = []
      data_list = []
      for inputs in dataset:
        datas = inputs['data']
        zs = self.encoder(datas)[0]
        zs_list.append(zs)
        data_list.append(datas.numpy())
      zs_list: np.ndarray = np.concatenate(zs_list, 0)
      data_list = np.concatenate(data_list, 0)
      data_list: np.ndarray = image_ops.renormalize(data_list, 127.5, 127.5)
      figure1, figure2 = self.sample_knn(zs_list, data_list)
      self.summary.save_images({'knn': np.stack([figure1, figure2], 0)})


class InfoMaxSslV1Loop(InfoMaxLoop):
  """ InfoMax semi-supervised learning V1

  Args:
      update_augmenter_state: *UPDATE_AUGMENTER_STATE
      infomax:
        ws: 1. # weight for supervised cross entropy loss
        wkl: 0.01 # weight for kl prior loss
        wginfo: 0.5 # weight for global information loss
        wlinfo: 1.5 # weight for local information loss
      ema:
        enable: false
        decay: 0.999
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

  @staticmethod
  def interior_partion_loss(model: tf.keras.Model,
                            data: tf.Tensor) -> List[tf.Tensor]:
    """ calc infomax model kl loss and information loss

    Args:
        model (tf.keras.Model): train model
        data (tf.Tensor): data

    Returns:
        List[tf.Tensor]: logits, kl_loss, zz_loss, zf_loss
    """
    (
        logits,
        z_mean,
        z_log_sigma,
        zz_true_scores,
        zz_false_scores,
        zz_label,
        zf_true_scores,
        zf_false_scores,
        zf_label,
    ) = model(
        data, training=True)
    # prior kl loss
    kl_loss = InfoMaxLoop.prior_kl_loss(z_mean, z_log_sigma)

    # information loss
    zz_loss = InfoMaxLoop.info_loss(zz_true_scores, zz_false_scores, zz_label)
    zf_loss = InfoMaxLoop.info_loss(zf_true_scores, zf_false_scores, zf_label)
    return logits, kl_loss, zz_loss, zf_loss

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: dict):
      """Per-Replica training step function."""
      sup_data = inputs['data']
      sup_label = inputs['label']
      unsup_data = inputs['unsup_data']
      # unsup_aug_data = inputs['unsup_aug_data']
      with tf.GradientTape() as tape:

        sup_logits, sup_kl_loss, sup_zz_loss, sup_zf_loss = self.interior_partion_loss(
            self.train_model, sup_data)
        # Supervised loss
        sup_xe_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sup_label, logits=sup_logits))

        unsup_logits, unsup_kl_loss, unsup_zz_loss, unsup_zf_loss = self.interior_partion_loss(
            self.train_model, unsup_data)

        # Model weights regularization
        wd_loss = tf.reduce_mean(self.train_model.losses)
        # yapf: disable
        loss = (self.hparams.infomax.ws * sup_xe_loss +
                self.hparams.infomax.wkl * (sup_kl_loss + unsup_kl_loss) +
                self.hparams.infomax.wginfo * (sup_zz_loss + unsup_zz_loss) +
                self.hparams.infomax.wlinfo * (sup_zf_loss + unsup_zf_loss) +
                wd_loss)
        # yapf: enable

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
      metrics.acc.update_state(sup_label, sup_logits)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

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
      self.run_step_fn(step_fn, args=(inputs,))


class InfoMaxSslV2Loop(InfoMaxSslV1Loop):
  """ InfoMax semi-supervised learning V2
    使用fixmatch结合infomax

  Args:
      update_augmenter_state: *UPDATE_AUGMENTER_STATE
      infomax:
        confidence: 0.85 # weight for pseudo label cross entropy loss mask
        ws: 1. # weight for supervised cross entropy loss
        wu: 1. # weight for pseudo label cross entropy loss
        wkl: 0.01 # weight for kl prior loss
        wginfo: 0.5 # weight for global information loss
        wlinfo: 1.5 # weight for local information loss
      ema:
        enable: false
        decay: 0.999
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss':
                tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'xe_loss':
                tf.keras.metrics.Mean('xe_loss', dtype=tf.float32),
            'xeu_loss':
                tf.keras.metrics.Mean('xeu_loss', dtype=tf.float32),
            'kl_loss':
                tf.keras.metrics.Mean('kl_loss', dtype=tf.float32),
            'zz_loss':
                tf.keras.metrics.Mean('zz_loss', dtype=tf.float32),
            'zf_loss':
                tf.keras.metrics.Mean('zf_loss', dtype=tf.float32),
            'wd_loss':
                tf.keras.metrics.Mean('wd_loss', dtype=tf.float32),
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
      data = inputs['data']
      label = inputs['label']
      udata = inputs['unsup_data']
      uaugdata = inputs['unsup_aug_data']
      with tf.GradientTape() as tape:

        logits, kl_loss, zz_loss, zf_loss = self.interior_partion_loss(
            self.train_model, data)

        ulogits, ukl_loss, uzz_loss, uzf_loss = self.interior_partion_loss(
            self.train_model, udata)

        uauglogits, uaugkl_loss, uaugzz_loss, uaugzf_loss = self.interior_partion_loss(
            self.train_model, uaugdata)

        # Supervised loss
        xe_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=logits))

        # Un-Supervised loss
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(ulogits))
        xeu_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(pseudo_labels, axis=1), logits=uauglogits)
        pseudo_mask = (
            tf.reduce_max(pseudo_labels, axis=1) >=
            self.hparams.infomax.confidence)
        pseudo_mask = tf.cast(pseudo_mask, tf.float32)
        xeu_loss = tf.reduce_mean(xeu_loss * pseudo_mask)

        # Model weights regularization
        wd_loss = tf.reduce_mean(self.train_model.losses)

        xe_loss = self.hparams.infomax.ws * xe_loss
        xeu_loss = self.hparams.infomax.wu * xeu_loss
        kl_loss = self.hparams.infomax.wkl * (kl_loss + ukl_loss + uaugkl_loss)
        zz_loss = self.hparams.infomax.wginfo * (zz_loss + uzz_loss + uaugzz_loss)
        zf_loss = self.hparams.infomax.wlinfo * (zf_loss + uzf_loss + uaugzf_loss)

        loss = xe_loss + xeu_loss + kl_loss + zz_loss + zf_loss + wd_loss

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
      metrics.xe_loss.update_state(xe_loss)
      metrics.xeu_loss.update_state(xeu_loss)
      metrics.kl_loss.update_state(kl_loss)
      metrics.zz_loss.update_state(zz_loss)
      metrics.zf_loss.update_state(zf_loss)
      metrics.wd_loss.update_state(wd_loss)
      metrics.acc.update_state(label, logits)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))
