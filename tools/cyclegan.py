import tensorflow_datasets as tfds
import tensorflow as tf
import sys
sys.path.insert(0, '/home/zqh/Documents/K210_Yolo_framework')
from tools.base import BaseHelper
from tools.training_engine import EasyDict
import transforms.image.ops as image_ops


class CycleGanHelper(BaseHelper):

  def __init__(self, dataset: str, in_hw: list, mixed_precision_dtype: str,
               hparams: dict):
    """ 
      hparams:
        null
    """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    if dataset == None:
      self.train_list: str = None
      self.val_list: str = None
      self.test_list: str = None
      self.unlabel_list: str = None
    else:
      assert dataset in ['cycle_gan/horse2zebra']
      datasets, metadata = tfds.load(
          dataset,
          data_dir='~/workspace/tensorflow_datasets',
          with_info=True,
          as_supervised=True)
      train_horses, train_zebras = datasets['trainA'], datasets['trainB']
      test_horses, test_zebras = datasets['testA'], datasets['testB']

      # NOTE can use dict set trian and test dataset
      self.train_list: List[tf.data.Dataset] = [train_horses, train_zebras]

      self.val_list: List[tf.data.Dataset] = [test_horses, test_zebras]

      self.test_list: List[tf.data.Dataset] = self.val_list

      self.train_total_data: int = metadata.splits[
          'trainA'].num_examples + metadata.splits['trainB'].num_examples
      self.val_total_data: int = metadata.splits[
          'testA'].num_examples + metadata.splits['testB'].num_examples
      self.test_total_data: int = self.val_total_data

    self.in_hw: list = in_hw
    self.mixed_precision_dtype = mixed_precision_dtype
    self.hparams = EasyDict(hparams)

  @staticmethod
  def random_crop(image, h, w):
    cropped_image = tf.image.random_crop(image, size=[h, w, 3])
    return cropped_image

  def normalize(self, image):
    image = tf.cast(image, self.mixed_precision_dtype)
    image = image_ops.normalize(image,
                                tf.constant(127.5, self.mixed_precision_dtype),
                                tf.constant(127.5, self.mixed_precision_dtype))
    return image

  @staticmethod
  def random_jitter(image, h, w):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = CycleGanHelper.random_crop(image, h, w)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(image, label):

      if is_augment:
        image = self.random_jitter(image, self.in_hw[0], self.in_hw[1])
      if is_normalize:
        image = self.normalize(image)

      return image

    ds_x = self.train_list[0].map(pipe,
                                  -1).shuffle(300).repeat().batch(batch_size)
    ds_y = self.train_list[1].map(pipe,
                                  -1).shuffle(300).repeat().batch(batch_size)

    ds = tf.data.Dataset.zip((ds_x, ds_y))
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(image, label):
      if is_normalize:
        image = self.normalize(image)
      return image

    ds_x = self.train_list[0].map(pipe).repeat().batch(1)
    ds_y = self.train_list[1].map(pipe).repeat().batch(1)
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


if __name__ == "__main__":
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  from models import gannet
  h = CycleGanHelper('cycle_gan/horse2zebra', [256, 256], 'float32', dict())
  batch_size = 8
  h.set_dataset(batch_size, True, True)

  generator_g = gannet.Generator(norm_type='instancenorm')
  generator_f = gannet.Generator(norm_type='instancenorm')

  discriminator_x = gannet.Discriminator(norm_type='instancenorm', target=False)
  discriminator_y = gannet.Discriminator(norm_type='instancenorm', target=False)

  @tf.function
  def generate_images(model, test_input):
    prediction = model(test_input)
    test_input = image_ops.renormalize(test_input, 127.5, 127.5)
    prediction = image_ops.renormalize(prediction, 127.5, 127.5)
    nimg = tf.cast(tf.concat([test_input, prediction], 0), tf.uint8)
    return nimg

  def discriminator_loss(real, generated):
    real_loss = tf.keras.backend.binary_crossentropy(
        tf.ones_like(real), real, from_logits=True)

    generated_loss = tf.keras.backend.binary_crossentropy(
        tf.zeros_like(generated), generated, from_logits=True)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

  def generator_loss(generated):
    return tf.keras.backend.binary_crossentropy(
        tf.ones_like(generated), generated, from_logits=True)

  def calc_cycle_loss(real_image, cycled_image, LAMBDA=10):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

  def identity_loss(real_image, same_image, LAMBDA=10):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

  generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  @tf.function
  def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.

      fake_y = generator_g(real_x, training=True)
      cycled_x = generator_f(fake_y, training=True)

      fake_x = generator_f(real_y, training=True)
      cycled_y = generator_g(fake_x, training=True)

      # same_x and same_y are used for identity loss.
      same_x = generator_f(real_x, training=True)
      same_y = generator_g(real_y, training=True)

      disc_real_x = discriminator_x(real_x, training=True)
      disc_real_y = discriminator_y(real_y, training=True)

      disc_fake_x = discriminator_x(fake_x, training=True)
      disc_fake_y = discriminator_y(fake_y, training=True)

      # calculate the loss
      gen_g_loss = generator_loss(disc_fake_y)
      gen_f_loss = generator_loss(disc_fake_x)

      total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(
          real_y, cycled_y)

      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(
          real_y, same_y)
      total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(
          real_x, same_x)

      disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
      disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return tf.reduce_mean(total_gen_g_loss), tf.reduce_mean(
        disc_x_loss), tf.reduce_mean(total_gen_f_loss), tf.reduce_mean(
            disc_y_loss)

  train_iter = iter(h.train_dataset)
  val_iter = iter(h.val_dataset)
  epoch_step = h.train_total_data // batch_size
  file_writer = tf.summary.create_file_writer('/tmp/cyclegan')
  with file_writer.as_default():
    for epoch in range(50):
      print("epoch: ", epoch)
      for _ in range(epoch_step):
        image_x, image_y = next(train_iter)
        gx_loss, dx_loss, gx_loss, dy_loss = train_step(image_x, image_y)
        step = epoch * epoch_step + _
        tf.summary.scalar('g_loss', gx_loss, step)
        tf.summary.scalar('l_loss', dx_loss, step)
        tf.summary.scalar('kl_loss', gx_loss, step)
        tf.summary.scalar('wd_loss', dy_loss, step)

      sample_x, sample_y = next(val_iter)
      darw_img = generate_images(generator_g, sample_x)
      tf.summary.image("val/image", darw_img, step=step)
