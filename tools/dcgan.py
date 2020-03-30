import tensorflow as tf
from tools.base import BaseHelper
import transforms.image.ops as image_ops
from tools.training_engine import EasyDict, GanBaseTrainingLoop


class KerasDatasetGanHelper(BaseHelper):

  def __init__(self, dataset: str, mixed_precision_dtype: str, hparams: dict):
    """ 
      hparams:
        noise_dims: 100 # when use dcgan, input noise shape
    """
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
      if len(x_train.shape) == 3:
        x_train = x_train[..., None]
        x_test = x_test[..., None]
      # NOTE can use dict set trian and test dataset
      self.train_list: Tuple[np.ndarray, np.ndarray] = (x_train, y_train)
      self.val_list: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
      self.test_list: Tuple[np.ndarray, np.ndarray] = None
      self.train_total_data: int = len(x_train)
      self.val_total_data: int = len(x_test)
      self.test_total_data: int = None

    self.in_hw: list = list(x_train.shape[1:])
    self.nclasses = len(set(y_train))
    self.mixed_precision_dtype = mixed_precision_dtype
    self.hparams = EasyDict(hparams)

  def normlize(self, img):
    img = tf.cast(img, self.mixed_precision_dtype)
    img = image_ops.normalize(img, tf.constant(127.5, self.mixed_precision_dtype),
                              tf.constant(127.5, self.mixed_precision_dtype))
    return img

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(img: tf.Tensor):
      img.set_shape(self.in_hw)

      if is_normalize:
        img = self.normlize(img)

      data_dict = {
          'data': img,
          'noise': tf.random.normal([self.hparams.noise_dim])
      }

      return data_dict

    ds = tf.data.Dataset.from_tensor_slices(self.train_list[0]).shuffle(
        batch_size * 300).repeat().map(pipe, tf.data.experimental.AUTOTUNE).batch(
            batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(i):
      data_dict = {'noise': tf.random.normal([1, self.hparams.noise_dim])}
      return data_dict

    ds: tf.data.Dataset = (
        tf.data.Dataset.range(batch_size).map(
            pipe, num_parallel_calls=-1).batch(batch_size))
    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class DCGanLoop(GanBaseTrainingLoop):

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'd_loss': tf.keras.metrics.Mean('d_loss', dtype=tf.float32),
        },
        'val': {
            'loss': tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        }
    }
    return d

  @staticmethod
  def discriminator_loss(real_output, fake_output):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

  @staticmethod
  def generator_loss(fake_output):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        tf.ones_like(fake_output), fake_output)

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      datas, noises = inputs['data'], inputs['noise']
      with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = self.g_model(noises, training=True)

        real_output = self.d_model(datas, training=True)
        fake_output = self.d_model(generated_images, training=True)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        if self.strategy:
          scaled_gen_loss = gen_loss / self.strategy.num_replicas_in_sync
          scaled_disc_loss = disc_loss / self.strategy.num_replicas_in_sync
        else:
          scaled_gen_loss = gen_loss
          scaled_disc_loss = disc_loss
      g_grad = g_tape.gradient(scaled_gen_loss, self.g_model.trainable_variables)
      d_grad = d_tape.gradient(scaled_disc_loss, self.d_model.trainable_variables)

      self.g_optimizer.apply_gradients(
          zip(g_grad, self.g_model.trainable_variables))
      self.d_optimizer.apply_gradients(
          zip(d_grad, self.d_model.trainable_variables))

      # if self.hparams.ema.enable:
      #   EmaHelper.update_ema_vars(self.val_model.variables,
      #                             self.train_model.variables,
      #                             self.hparams.ema.decay)
      metrics.g_loss.update_state(tf.reduce_mean(scaled_gen_loss))
      metrics.d_loss.update_state(tf.reduce_mean(scaled_disc_loss))

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  # @tf.function
  # def val_step(self, dataset, metrics):

  #   def step_fn(inputs):
  #     """Per-Replica training step function."""
  #     datas, labels = inputs['data'], inputs['label']
  #     logits = self.val_model(datas, training=False)
  #     loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  #     loss_xe = tf.reduce_mean(loss_xe)
  #     loss_wd = tf.reduce_sum(self.val_model.losses)
  #     loss = loss_xe + loss_wd
  #     metrics.loss.update_state(loss)
  #     metrics.acc.update_state(labels, tf.nn.softmax(logits))

  #   for inputs in dataset:
  #     if self.strategy:
  #       self.strategy.experimental_run_v2(step_fn, args=(inputs,))
  #     else:
  #       step_fn(inputs,)