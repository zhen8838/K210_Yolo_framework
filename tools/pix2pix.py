import tensorflow as tf
from tools.base import BaseHelper
import transforms.image.ops as image_ops
from tools.training_engine import EasyDict, GanBaseTrainingLoop
from typing import List


class CMPFacadeHelper(BaseHelper):

  def __init__(self, dataset_root: str, in_hw: list, mixed_precision_dtype: str,
               hparams: dict):
    """ 
      hparams:
        null
    """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    if dataset_root == None:
      self.train_list: str = None
      self.val_list: str = None
      self.test_list: str = None
      self.unlabel_list: str = None
    else:

      # NOTE can use dict set trian and test dataset
      self.train_list: List[str] = tf.io.gfile.glob('/'.join(
          [dataset_root, 'train', '*.jpg']))

      self.val_list: List[str] = tf.io.gfile.glob('/'.join(
          [dataset_root, 'test', '*.jpg']))

      self.test_list: List[str] = tf.io.gfile.glob('/'.join(
          [dataset_root, 'test', '*.jpg']))

      self.train_total_data: int = len(self.train_list)
      self.val_total_data: int = len(self.val_list)
      self.test_total_data: int = len(self.test_list)

    self.in_hw: list = in_hw
    self.mixed_precision_dtype = mixed_precision_dtype
    self.hparams = EasyDict(hparams)

  def normalize(self, input_image, real_image):
    input_image = tf.cast(input_image, self.mixed_precision_dtype)
    input_image = image_ops.normalize(
        input_image, tf.constant(127.5, self.mixed_precision_dtype),
        tf.constant(127.5, self.mixed_precision_dtype))

    real_image = tf.cast(real_image, self.mixed_precision_dtype)
    real_image = image_ops.normalize(
        real_image, tf.constant(127.5, self.mixed_precision_dtype),
        tf.constant(127.5, self.mixed_precision_dtype))

    return input_image, real_image

  def read_img(self, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    return input_image, real_image

  @staticmethod
  def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

  @staticmethod
  def random_crop(input_image, real_image, height, width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, 3])
    return cropped_image[0], cropped_image[1]

  @staticmethod
  def random_jitter(input_image, real_image, height, width):
    # resizing to 286 x 286 x 3
    input_image, real_image = CMPFacadeHelper.resize(input_image, real_image, 286,
                                                     286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = CMPFacadeHelper.random_crop(
        input_image, real_image, height, width)

    if tf.random.uniform(()) > 0.5:
      # random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(img_path: tf.Tensor):
      input_image, real_image = self.read_img(img_path)
      if is_augment:
        input_image, real_image = self.random_jitter(input_image, real_image,
                                                     self.in_hw[0], self.in_hw[1])
      if is_normalize:
        input_image, real_image = self.normalize(input_image, real_image)

      return {'input_data': input_image, 'real_data': real_image}

    ds = tf.data.Dataset.from_tensor_slices(self.train_list).map(
        pipe, -1).shuffle(400).repeat().batch(batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(img_path: tf.Tensor):
      input_image, real_image = self.read_img(img_path)
      input_image, real_image = self.resize(input_image, real_image,
                                            self.in_hw[0], self.in_hw[1])
      if is_normalize:
        input_image, real_image = self.normalize(input_image, real_image)

      return {'input_data': input_image, 'real_data': real_image}

    ds = tf.data.Dataset.from_tensor_slices(
        self.val_list[0:1]).repeat().map(pipe).batch(1)
    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class Pix2PixLoop(GanBaseTrainingLoop):
  """ Pix2PixLoop
  
  Args:
      hparams:
        wl1: 100 # generator loss = gan_loss + wl1 * l1_loss, in paper wl1 = 100. 
  
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'gan_loss': tf.keras.metrics.Mean('gan_loss', dtype=tf.float32),
            'l1_loss': tf.keras.metrics.Mean('l1_loss', dtype=tf.float32),
            'd_loss': tf.keras.metrics.Mean('d_loss', dtype=tf.float32),
        },
        'val': {}
    }
    return d

  @staticmethod
  def generator_loss(disc_generated_output, gen_output, target, LAMBDA):
    """
    * 是由标签为1的二分类交叉熵组成
    * 还包括了生成图像与标签图像的mae误差损失
    * 计算生成器总损失的公式为gan_loss + LAMBDA * l1_loss，其中LAMBDA =100。该值由论文作者决定。
    
    
    """
    gan_loss = tf.keras.backend.binary_crossentropy(
        tf.ones_like(disc_generated_output),
        disc_generated_output,
        from_logits=True)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA*l1_loss)

    return total_gen_loss, gan_loss, l1_loss

  @staticmethod
  def discriminator_loss(disc_real_output, disc_generated_output):
    """
    * 鉴频器损耗功能需要2个输入；真实图像，生成图像
    * real_loss是真实图像和一系列图像的S形交叉熵损失（因为这些是真实图像）
    * generate_loss是生成图像和零数组的S型交叉熵损失（因为这些是伪图像）
    * 然后total_loss是real_loss和generate_loss的总和
    
    Returns:
        total_disc_loss
    """
    real_loss = tf.keras.backend.binary_crossentropy(
        tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

    generated_loss = tf.keras.backend.binary_crossentropy(
        tf.zeros_like(disc_generated_output),
        disc_generated_output,
        from_logits=True)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      input_image, target = inputs['input_data'], inputs['real_data']
      with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        gen_output = self.g_model(input_image, training=True)

        disc_real_output = self.d_model([input_image, target], training=True)
        disc_generated_output = self.d_model([input_image, gen_output],
                                             training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
            disc_generated_output, gen_output, target, self.hparams.wl1)
        disc_loss = self.discriminator_loss(disc_real_output,
                                            disc_generated_output)

        if self.strategy:
          scaled_gen_loss = gen_total_loss / self.strategy.num_replicas_in_sync
          scaled_disc_loss = disc_loss / self.strategy.num_replicas_in_sync
        else:
          scaled_gen_loss = gen_total_loss
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
      metrics.g_loss.update_state(gen_total_loss)
      metrics.gan_loss.update_state(gen_gan_loss)
      metrics.l1_loss.update_state(gen_l1_loss)
      metrics.d_loss.update_state(disc_loss)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @staticmethod
  @tf.function
  def val_generate_images(g_model, test_input, target):
    prediction = g_model(test_input, training=True)
    a = image_ops.renormalize(test_input, 127.5, 127.5)
    b = image_ops.renormalize(target, 127.5, 127.5)
    c = image_ops.renormalize(prediction, 127.5, 127.5)
    nimg = tf.cast(tf.concat([a, b, c], 2), tf.uint8)
    return nimg

  def local_variables_init(self):
    self.val_iterator = iter(self.val_dataset)

  def val_step(self, dataset, metrics):
    inputs = next(self.val_iterator)
    example_input, example_target = inputs['input_data'], inputs['real_data']
    img = self.strategy.experimental_run_v2(
        self.val_generate_images, (self.g_model, example_input, example_target))
    self.summary.save_images({'img': img})
