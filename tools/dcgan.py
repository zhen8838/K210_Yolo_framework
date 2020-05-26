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
  """ GanBaseTrainingLoop
  
  Args:
      hparams:
        noise_dim: *NOISE_DIM # generator model input noise dim
        val_nimg: 16 # when validation generate image numbers
  
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'd_loss': tf.keras.metrics.Mean('d_loss', dtype=tf.float32),
        },
        'val': {}
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

      scaled_g_loss = self.optimizer_minimize(gen_loss, g_tape, self.g_optimizer,
                                              self.g_model)
      scaled_d_loss = self.optimizer_minimize(disc_loss, d_tape, self.d_optimizer,
                                              self.d_model)

      if self.hparams.ema.enable:
        self.ema.update()
      metrics.g_loss.update_state(tf.reduce_mean(scaled_g_loss))
      metrics.d_loss.update_state(tf.reduce_mean(scaled_d_loss))

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  def local_variables_init(self):
    self.val_seed: tf.Tensor = tf.random.normal(
        [self.hparams.val_nimg, self.hparams.noise_dim])

  @staticmethod
  @tf.function
  def val_generate_images(g_model, test_input):
    img = g_model(test_input, training=False)
    img = image_ops.renormalize(img, 127.5, 127.5)
    imgw = tf.split(img, 4)
    imgh = tf.split(tf.concat(imgw, 2), 4)
    nimg = tf.concat(imgh, 1)
    return nimg

  def val_step(self, dataset, metrics):
    img = self.val_generate_images(self.g_model, self.val_seed)
    self.summary.save_images({'img': img})
