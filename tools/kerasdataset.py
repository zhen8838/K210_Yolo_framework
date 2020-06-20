import tensorflow as tf
from typing import Tuple, List
from tools.training_engine import BaseTrainingLoop, BaseHelperV2, EmaHelper
import transforms.image.ops as image_ops
from tools.custom import CircleLoss, SparseCircleLoss


class KerasDatasetHelper(BaseHelperV2):
  """ 
    hparams:
      class_num: 10
  """

  def set_datasetlist(self):
    dataset_dict = {
        'mnist': tf.keras.datasets.mnist,
        'cifar10': tf.keras.datasets.cifar10,
        'cifar100': tf.keras.datasets.cifar100,
        'fashion_mnist': tf.keras.datasets.fashion_mnist
    }

    assert self.dataset_root in dataset_dict.keys(), 'dataset is invalid!'
    (x_train, y_train), (x_test, y_test) = dataset_dict[self.dataset_root].load_data()
    # y_train = tf.keras.utils.to_categorical(y_train, self.hparams.class_num)
    # y_test = tf.keras.utils.to_categorical(y_test, self.hparams.class_num)
    # NOTE can use dict set trian and test dataset
    self.train_list: Tuple[np.ndarray, np.ndarray] = (x_train, y_train)
    self.val_list: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
    self.test_list: Tuple[np.ndarray, np.ndarray] = self.val_list
    self.train_total_data: int = len(x_train)
    self.val_total_data: int = len(x_test)
    self.test_total_data: int = self.val_total_data

  def build_train_datapipe(self, batch_size, is_augment, is_normalize=True, is_training=True):
    def map_fn(img, y):
      if is_normalize:
        img = image_ops.normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      else:
        img = tf.cast(img, tf.float32)
      return img, y

    if is_training:
      ds = (tf.data.Dataset.from_tensor_slices(self.train_list).
            shuffle(batch_size * 100).
            repeat().
            map(map_fn, self.hparams.num_parallel_calls).
            batch(batch_size, True).
            prefetch(tf.data.experimental.AUTOTUNE))
    else:
      ds = (tf.data.Dataset.from_tensor_slices(self.test_list).
            shuffle(batch_size * 100).
            map(map_fn, -1).
            batch(batch_size, True))

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size, is_normalize=True):
    return self.build_train_datapipe(batch_size, is_augment=False,
                                     is_normalize=is_normalize,
                                     is_training=False)


class KerasDatasetLoop(BaseTrainingLoop):
  """ 
  Args:
    loss_type: CircleLoss,...
    loss_args: 
      a: xxx
      b: xxx
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss': tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc', dtype=tf.float32),
        },
        'val': {
            'loss': tf.keras.metrics.Mean('vloss', dtype=tf.float32),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('vacc', dtype=tf.float32),
        }
    }
    return d

  def local_variables_init(self):
    obj = eval(self.hparams.loss_type)
    self.loss_fn: CircleLoss = obj(**self.hparams.loss_args.dicts())

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs: List[tf.Tensor]):
      """Per-Replica training step function."""
      x, y_true = inputs

      with tf.GradientTape() as tape:
        y_pred = self.train_model(x, training=True)
        loss = self.loss_fn.call(y_true, y_pred)
        loss_wd = tf.reduce_sum(self.train_model.losses)
        loss = tf.reduce_sum(loss) + loss_wd

      scaled_loss = self.optimizer_minimize(loss, tape,
                                            self.optimizer,
                                            self.train_model)

      if self.hparams.ema.enable:
        self.ema.update()
      # loss metric
      metrics.loss.update_state(scaled_loss)
      metrics.acc.update_state(y_true, y_pred)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):
    if self.hparams.ema.enable:
      val_model = self.ema.model
    else:
      val_model = self.val_model

    def step_fn(inputs: dict):
      x, y_true = inputs
      y_pred = val_model(x, training=False)
      loss = self.loss_fn.call(y_true, y_pred)
      loss_wd = tf.reduce_sum(val_model.losses)
      loss = tf.reduce_sum(loss) + loss_wd

      metrics.loss.update_state(loss)
      metrics.acc.update_state(y_true, y_pred)

    for inputs in dataset:
      self.run_step_fn(step_fn, args=(inputs,))
