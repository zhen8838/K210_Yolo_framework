import tensorflow as tf
import numpy as np
from tools.base import INFO, ERROR, NOTE
from tools.training_engine import BaseHelperV2, EasyDict, GanBaseTrainingLoop
import transforms.image.ops as image_ops
import os
from tensorflow.python.keras.losses import huber_loss
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt


class AnimeGanHelper(BaseHelperV2):
  """
    from paper: `AnimeGAN: a novel lightweight GAN for photo animation`
    
    dataset download from `https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1`
  
  Args:
      hparams:
        style: Hayao # use which anime style for training
  
  """

  def set_datasetlist(self):
    assert tf.io.gfile.exists(self.dataset_root), 'dataset_root not exists !'
    real_root = os.path.join(self.dataset_root, 'train_photo')
    val_root = os.path.join(self.dataset_root, 'test/test_photo')
    anime_root = os.path.join(self.dataset_root, self.hparams.style, 'style')
    animesooth_root = os.path.join(self.dataset_root, self.hparams.style,
                                   'smooth')

    self.train_list: List[list[str]] = [[
        os.path.join(root, fname) for fname in tf.io.gfile.listdir(root)
    ] for root in [real_root, anime_root, animesooth_root]]

    self.val_list: List[str] = [[
        os.path.join(root, fname) for fname in tf.io.gfile.listdir(root)
    ] for root in [val_root]]
    self.test_list: List[str] = self.val_list
    self.train_total_data: int = max([len(l) for l in self.train_list])
    self.val_total_data: int = max([len(l) for l in self.val_list])
    self.test_total_data: int = self.val_total_data

  def normalize(self, image):
    image = tf.cast(image, self.mixed_precision_dtype)
    image = image_ops.normalize(image,
                                tf.constant(127.5, self.mixed_precision_dtype),
                                tf.constant(127.5, self.mixed_precision_dtype))
    return image

  @staticmethod
  def random_crop(image, h, w):
    cropped_image = tf.image.random_crop(image, size=[h, w, 3])
    return cropped_image

  @staticmethod
  def random_jitter(image, h, w):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to h x w x 3
    image = AnimeGanHelper.random_crop(image, h, w)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image

  @staticmethod
  def imread(image_path: str) -> tf.Tensor:
    image = tf.image.decode_image(
        tf.io.read_file(image_path), channels=3, expand_animations=False)
    return image

  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:
    """ need `real image, anime image, anime image gray, anime image smooth gray` """

    def pipe(real_path, anime_path, anime_smooth_path):
      real = self.imread(real_path)
      anime = self.imread(anime_path)
      anime_smooth = self.imread(anime_smooth_path)
      data_dict = {}
      data_dict['real_data'] = real
      data_dict['anime_data'] = anime
      data_dict['anime_gray_data'] = tf.tile(
          tf.image.rgb_to_grayscale(anime), [1, 1, 3])
      # data_dict['anime_smooth_data'] = tf.tile(
      #     tf.image.rgb_to_grayscale(anime_smooth), [1, 1, 3])
      data_dict['anime_smooth_data'] = anime_smooth
      if is_augment:
        for key, v in data_dict.items():
          if key.startswith('real'):
            # data_dict[key] = self.random_jitter(v, self.in_hw[0], self.in_hw[1])
            data_dict[key] = tf.image.random_flip_left_right(v)

      if is_normalize:
        for key, v in data_dict.items():
          if key.endswith('data'):
            data_dict[key] = self.normalize(v)
            data_dict[key].set_shape(self.in_hw + [3])
      return data_dict

    ds_real = tf.data.Dataset.from_tensor_slices(
        self.train_list[0]).shuffle(300).repeat()
    ds_anime = tf.data.Dataset.from_tensor_slices(
        self.train_list[1]).shuffle(300).repeat()
    ds_anime_smooth = tf.data.Dataset.from_tensor_slices(
        self.train_list[2]).shuffle(300).repeat()

    ds = tf.data.Dataset.zip(
        (ds_real, ds_anime, ds_anime_smooth)).map(pipe, -1).batch(
            batch_size, drop_remainder=True).prefetch(-1)
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:

    def pipe(real_path):
      real = self.imread(real_path)
      data_dict = {}
      data_dict['real_data'] = tf.image.resize(real, self.in_hw)
      if is_normalize:
        for key, v in data_dict.items():
          if key.endswith('data'):
            data_dict[key] = self.normalize(v)
            data_dict[key].set_shape(self.in_hw + [3])
      return data_dict

    ds = tf.data.Dataset.from_tensor_slices(
        self.test_list[0][:4]).map(pipe).batch(4)

    return ds

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.train_epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size

  @staticmethod
  def _reduce_hw(img_hw: List[int], min_hw: List[int]) -> List[int]:
    im_h, im_w = img_hw
    if im_h <= min_hw[0]:
      im_h = min_hw[0]
    else:
      x = im_h % 32
      im_h = im_h - x

    if im_w < min_hw[1]:
      im_w = min_hw[1]
    else:
      y = im_w % 32
      im_w = im_w - y
    return [im_h, im_w]

  def parser_outputs(self, outputs: np.ndarray,
                     orig_imgs: np.ndarray) -> np.ndarray:

    outputs = image_ops.renormalize(outputs, 127.5, 127.5).astype('uint8')
    return np.concatenate((orig_imgs, outputs), -2)

  def infer(self, img_path: Path, infer_model: tf.keras.Model, result_path: Path,
            batch_size: int, save_dir: str):
    print(INFO, f'Load Images from {str(img_path)}')
    if img_path.is_dir():
      img_paths = []
      for suffix in ['bmp', 'jpg', 'jpeg', 'png']:
        img_paths += list(map(str, img_path.glob(f'*.{suffix}')))
    elif img_path.is_file():
      img_paths = [str(img_path)]
    else:
      ValueError(f'{ERROR} img_path `{str(img_path)}` is invalid')

    if result_path is not None:
      print(INFO, f'Load NNcase Results from {str(result_path)}')
      if result_path.is_dir():
        ncc_results: np.ndarray = np.array([
            np.fromfile(
                str(result_path / (Path(img_paths[i]).stem + '.bin')),
                dtype='float32') for i in range(len(img_paths))
        ])
      elif result_path.is_file():
        ncc_results = np.expand_dims(
            np.fromfile(str(result_path), dtype='float32'), 0)  # type:np.ndarray
      else:
        ValueError(f'{ERROR} result_path `{str(result_path)}` is invalid')
    else:
      ncc_results = None

    print(INFO, f'Infer Results')
    orig_imgs = []
    det_imgs = []
    for img_path in img_paths:
      img = self.imread(img_path)
      img_hw = self._reduce_hw(img.shape[:2], self.in_hw)
      img = tf.image.resize(img, img_hw)
      orig_imgs.append(img.numpy().astype('uint8'))
      det_img = self.normalize(img)
      det_imgs.append(det_img)

    det_imgs = tf.stack(det_imgs)
    orig_imgs = np.array(orig_imgs)

    outputs = infer_model.predict(det_imgs, batch_size=batch_size)
    """ parser batch out """
    results = self.parser_outputs(outputs, orig_imgs)
    """ show result """
    if result_path is None:
      """ draw gpu result """
      for i, res_img in enumerate(results):
        if save_dir:
          tf.io.write_file((Path(save_dir) / f'{i}.jpg').as_posix(),
                           tf.image.encode_jpeg(res_img))
        else:
          plt.imshow(res_img)
          plt.tight_layout()
          plt.axis('off')
          plt.xticks([])
          plt.yticks([])
          plt.show()
    else:
      """ draw gpu result and nncase result """
      pass


class AnimeGanInitLoop(GanBaseTrainingLoop):
  """ AnimeGanInitLoop for generator weight init
  
  Args:
      hparams:
        wc: 1.5 # l1 loss with pre-trained model weight 
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
        },
        'val': {}
    }
    return d

  @staticmethod
  def l1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

  @staticmethod
  def con_loss(pre_train_model, real_data, fake_data):
    real_fmap = pre_train_model(real_data, training=False)
    fake_fmap = pre_train_model(fake_data, training=False)
    con_loss = AnimeGanInitLoop.l1_loss(real_fmap, fake_fmap)
    return con_loss

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      real_data = inputs['real_data']
      with tf.GradientTape() as tape:
        gen_output = self.g_model(real_data, training=True)
        con_loss = self.con_loss(self.p_model, real_data, gen_output)
        loss = self.hparams.wc * con_loss

      scaled_loss = self.optimizer_minimize(loss, tape, self.g_optimizer,
                                            self.g_model)

      if self.hparams.ema.enable:
        self.ema.update()
      metrics.g_loss.update_state(scaled_loss)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  def local_variables_init(self):
    inputs = tf.keras.Input([256, 256, 3])
    model = tf.keras.applications.MobileNetV2(
        include_top=False,
        alpha=1.3,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None,
        classes=1000)
    self.p_model: tf.keras.Model = tf.keras.Model(
        inputs,
        model.get_layer('block_6_expand').output)
    # model: tf.keras.Model = tf.keras.applications.VGG19(
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=inputs,
    #     pooling=None,
    #     classes=1000)
    # self.p_model = tf.keras.Model(
    #     inputs,
    #     tf.keras.layers.Activation('linear', dtype=tf.float32)(
    #         model.get_layer('block4_conv4').output))
    self.p_model.trainable = False


class AnimeGanLoop(AnimeGanInitLoop):
  """ AnimeGanLoop for generator weight init
  
  Args:
      hparams:
        wc: 1.5 # l1 loss with pre-trained model weight 
        ws: 3.0 # sty loss weight 
        wcl: 10.0 # color loss weight 
        wg: 300.0 # generator loss weight
        wd: 300.0 # discriminator loss weight 
        wreal: 1.0 # discriminator real anime loss weight
        wgray: 1.0 # discriminator gray anime loss weight
        wfake: 1.0 # discriminator generate loss weight
        wblur: 0.1 # discriminator smooth anime loss weight
        ltype: lsgan # gan loss type in [gan, lsgan, wgan-gp, wgan-lp, dragan, hinge]
        ld: 10.0 # gradient penalty lambda
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'd_loss': tf.keras.metrics.Mean('d_loss', dtype=tf.float32),
        },
        'val': {},
        'debug': {
            'g_col_loss':
                tf.keras.metrics.Mean('g_col_loss', dtype=tf.float32),
            'g_con_loss':
                tf.keras.metrics.Mean('g_con_loss', dtype=tf.float32),
            'g_sty_loss':
                tf.keras.metrics.Mean('g_sty_loss', dtype=tf.float32),
            'g_col_loss':
                tf.keras.metrics.Mean('g_col_loss', dtype=tf.float32),
            'g_gen_loss':
                tf.keras.metrics.Mean('g_gen_loss', dtype=tf.float32),
            'd_real_loss':
                tf.keras.metrics.Mean('d_real_loss', dtype=tf.float32),
            'd_fake_loss':
                tf.keras.metrics.Mean('d_fake_loss', dtype=tf.float32),
            'd_real_blur_loss':
                tf.keras.metrics.Mean('d_real_blur_loss', dtype=tf.float32),
            'd_gray_loss':
                tf.keras.metrics.Mean('d_gray_loss', dtype=tf.float32),
        }
    }
    return d

  @staticmethod
  def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast(
        (tf.size(x) // b), tf.float32)

  @staticmethod
  def style_loss(style, fake):
    return AnimeGanLoop.l1_loss(AnimeGanLoop.gram(style), AnimeGanLoop.gram(fake))

  @staticmethod
  def con_sty_loss(pre_train_model, real, anime, fake):
    real_feature_map = pre_train_model(real, training=False)

    fake_feature_map = pre_train_model(fake, training=False)

    anime_feature_map = pre_train_model(anime, training=False)

    c_loss = AnimeGanLoop.l1_loss(real_feature_map, fake_feature_map)
    s_loss = AnimeGanLoop.style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss

  @staticmethod
  def rgb2yuv(rgb):
    rgb = image_ops.renormalize(rgb, 127.5, 127.5)
    return tf.image.rgb_to_yuv(rgb)

  @staticmethod
  def color_loss(con, fake):
    con = AnimeGanLoop.rgb2yuv(con)
    fake = AnimeGanLoop.rgb2yuv(fake)

    return AnimeGanLoop.l1_loss(con[..., 0], fake[..., 0]) + huber_loss(
        con[..., 1], fake[..., 1]) + huber_loss(con[..., 2], fake[..., 2])

  @staticmethod
  def generator_loss(loss_type, fake_logit):

    if loss_type == 'wgan-gp' or loss_type == 'wgan-lp':
      fake_loss = -tf.reduce_mean(fake_logit)

    if loss_type == 'lsgan':
      fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))

    if loss_type == 'gan' or loss_type == 'dragan':
      fake_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.ones_like(fake_logit), logits=fake_logit))

    if loss_type == 'hinge':
      fake_loss = -tf.reduce_mean(fake_logit)

    return fake_loss

  @staticmethod
  def gradient_panalty(loss_type: str, gradtape: tf.GradientTape,
                       discriminator: tf.keras.Model, real: tf.Tensor,
                       fake: tf.Tensor, ld: float):
    # 梯度惩罚
    if 'dragan' in loss_type:
      eps = tf.random.uniform(tf.shape(real), 0., 1.)
      _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
      # magnitude of noise decides the size of local region
      x_std = tf.sqrt(x_var)

      fake = real + 0.5*x_std*eps

    batch = tf.shape(real)[0]
    alpha = tf.random.uniform([batch, 1, 1, 1], 0., 1.)
    interpolated = real + alpha * (fake-real)
    gradtape.watch(interpolated)
    logit = discriminator(interpolated, training=True)
    with gradtape.stop_recording():
      # gradient of D(interpolated) NOTE : should test more
      grad = gradtape.gradient(logit, interpolated)[0]
    grad_norm = tf.norm(tf.keras.layers.Flatten()(grad), axis=1)  # l2 norm

    gp_loss = 0
    # WGAN - LP
    if 'lp' in loss_type:
      gp_loss = ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
    elif 'gp' in loss_type or 'dragan' == loss_type:
      gp_loss = ld * tf.reduce_mean(tf.square(grad_norm - 1.))

    return gp_loss

  @staticmethod
  def discriminator_loss(loss_type, real, gray, fake, real_blur):
    if loss_type == 'wgan-gp' or loss_type == 'wgan-lp':
      real_loss = -tf.reduce_mean(real)
      gray_loss = tf.reduce_mean(gray)
      fake_loss = tf.reduce_mean(fake)
      real_blur_loss = tf.reduce_mean(real_blur)

    if loss_type == 'lsgan':

      real_loss = tf.reduce_mean(tf.square(real - 1.0))
      gray_loss = tf.reduce_mean(tf.square(gray))
      fake_loss = tf.reduce_mean(tf.square(fake))
      real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_type == 'gan' or loss_type == 'dragan':
      real_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.ones_like(real), logits=real))
      gray_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.zeros_like(gray), logits=gray))
      fake_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.zeros_like(fake), logits=fake))
      real_blur_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_type == 'hinge':
      real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
      gray_loss = tf.reduce_mean(tf.nn.relu(1.0 + gray))
      fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))
      real_blur_loss = tf.reduce_mean(tf.nn.relu(1.0 + real_blur))

    return real_loss, gray_loss, fake_loss, real_blur_loss

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      real_data = inputs['real_data']
      anime_data = inputs['anime_data']
      anime_gray_data = inputs['anime_gray_data']
      anime_smooth_data = inputs['anime_smooth_data']
      with tf.GradientTape(persistent=True) as tape:
        # forward
        gen_output = self.g_model(real_data, training=True)
        gen_logit = self.d_model(gen_output, training=True)

        anime_logit = self.d_model(anime_data, training=True)
        smooth_logit = self.d_model(anime_smooth_data, training=True)
        gray_logit = self.d_model(anime_gray_data, training=True)

        # generator loss
        # con_loss, sty_loss = self.con_sty_loss(self.p_model, real_data,
        #                                        anime_gray_data, gen_output)
        con_loss, sty_loss = self.con_sty_loss(self.p_model, real_data,
                                               anime_data, gen_output)
        con_loss = self.hparams.wc * con_loss
        sty_loss = self.hparams.ws * sty_loss
        col_loss = self.hparams.wcl * self.color_loss(real_data, gen_output)
        gen_loss = self.hparams.wg * self.generator_loss(self.hparams.ltype,
                                                         gen_logit)
        g_loss = col_loss + con_loss + sty_loss + col_loss + gen_loss

        # gradient panalty
        if ('gp' in self.hparams.ltype or 'lp' in self.hparams.ltype or
            'dragan' in self.hparams.ltype):
          gp_loss = self.gradient_panalty(self.hparams.ltype, tape, self.d_model,
                                          anime_data, gen_output, self.hparams.ld)
        else:
          gp_loss = 0.0
        real_loss, gray_loss, fake_loss, real_blur_loss = self.discriminator_loss(
            self.hparams.ltype, anime_logit, gray_logit, gen_logit, smooth_logit)
        # discriminator loss
        real_loss = self.hparams.wd * real_loss * self.hparams.wreal
        gray_loss = self.hparams.wd * gray_loss * self.hparams.wgray
        fake_loss = self.hparams.wd * fake_loss * self.hparams.wfake
        real_blur_loss = self.hparams.wd * real_blur_loss * self.hparams.wblur
        d_loss = real_loss + fake_loss + real_blur_loss + gray_loss + gp_loss

      scaled_g_loss = self.optimizer_minimize(g_loss, tape, self.g_optimizer,
                                              self.g_model)

      scaled_d_loss = self.optimizer_minimize(d_loss, tape, self.d_optimizer,
                                              self.d_model)

      if self.hparams.ema.enable:
        self.ema.update()
      metrics.g_loss.update_state(scaled_g_loss)
      self.metrics.debug.g_col_loss.update_state(col_loss)
      self.metrics.debug.g_con_loss.update_state(con_loss)
      self.metrics.debug.g_sty_loss.update_state(sty_loss)
      self.metrics.debug.g_col_loss.update_state(col_loss)
      self.metrics.debug.g_gen_loss.update_state(gen_loss)

      metrics.d_loss.update_state(scaled_d_loss)
      self.metrics.debug.d_real_loss.update_state(real_loss)
      self.metrics.debug.d_fake_loss.update_state(fake_loss)
      self.metrics.debug.d_real_blur_loss.update_state(real_blur_loss)
      self.metrics.debug.d_gray_loss.update_state(gray_loss)

    for _ in tf.range(num_steps_to_run):
      self.strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  def val_step(self, dataset, metrics):
    for inputs in dataset:
      real_data = inputs['real_data']
      gen_output = self.g_model(real_data, training=False)
      gen_output = tf.cast(
          image_ops.renormalize(gen_output, 127.5, 127.5), tf.uint8)
      self.summary.save_images({'gen': gen_output}, 4)
