from tools.training_engine import BaseHelperV2, GanBaseTrainingLoop
from pathlib import Path
import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from transforms.image.ops import normalize
from typing import List
from functools import partial


class PhotoTransferHelper(BaseHelperV2):
  def set_datasetlist(self):
    dataset_root = Path(self.dataset_root)
    self.trainA = dataset_root / 'trainA'
    self.trainB = dataset_root / 'trainB'
    self.testA = dataset_root / 'testA'
    self.testB = dataset_root / 'testB'

  def build_train_datapipe(self, batch_size, is_augment, is_normalize=True):
    def map_fn(img, key_name):
      if is_augment:
        img = tf.image.resize(img, (self.in_hw[0] + 30, self.in_hw[1] + 30))
        img = tf.image.random_crop(img, self.in_hw)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 3)
      else:
        img = tf.image.resize(img, self.in_hw)

      if is_normalize:
        img = normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      else:
        img = tf.cast(img, tf.float32)
      return {key_name: img}

    trainAds = (tf.data.Dataset.list_files(
        self.trainA.as_posix() + '/*.jpg', shuffle=True).
        shuffle(batch_size * 100).
        repeat().
        map(partial(map_fn, key_name='photo'), -1))

    trainBds = (tf.data.Dataset.list_files(
        self.trainB.as_posix() + '/*.jpg', shuffle=True).
        shuffle(batch_size * 100).
        repeat().
        map(partial(map_fn, key_name='cartoon'), -1))

    ds = (tf.data.Dataset.zip((trainAds, trainBds)).
          batch(batch_size, True).
          prefetch(tf.data.experimental.AUTOTUNE))

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size, is_normalize=True):
    def map_fn(img, key_name):
      img = tf.image.resize(img, self.in_hw)

      if is_normalize:
        img = normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      else:
        img = tf.cast(img, tf.float32)
      return {key_name: img}

    testAds = (tf.data.Dataset.list_files(
        self.testA.as_posix() + '/*.jpg', shuffle=True).
        shuffle(100).
        repeat().
        map(partial(map_fn, key_name='photo'), -1))

    testBds = (tf.data.Dataset.list_files(
        self.testB.as_posix() + '/*.jpg', shuffle=True).
        shuffle(100).
        repeat().
        map(partial(map_fn, key_name='cartoon'), -1))

    ds = (tf.data.Dataset.zip((testAds, testBds)).
          batch(1, True).
          prefetch(tf.data.experimental.AUTOTUNE))

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds


class UGATITPhotoTransferLoop(GanBaseTrainingLoop):
  def __init__(self, generator_model: List[k.Model],
               discriminator_model: List[k.Model],
               val_model: List[k.Model],
               generator_optimizer,
               discriminator_optimizer,
               strategy, **kwargs):
    assert isinstance(generator_model, list)
    assert isinstance(discriminator_model, list)
    genA2B, genB2A = generator_model
    disGA, disGB, disLA, disLB = discriminator_model

    self.genA2B: k.Model = genA2B
    self.genB2A: k.Model = genB2A
    self.disGA: k.Model = disGA
    self.disGB: k.Model = disGB
    self.disLA: k.Model = disLA
    self.disLB: k.Model = disLB

    self.train_model = self.g_model = genA2B
    self.d_model = disGA
    self.g_optimizer = self.optimizer = generator_optimizer
    self.d_optimizer = discriminator_optimizer
    self.val_model = val_model
    self.strategy = strategy
    self.models_dict: Mapping[str, k.Model] = {
        'generator_model': self.g_model,
        'discriminator_model': self.d_model,
        'val_model': self.val_model,
    }
    self.__init_kwargs(kwargs)
    self.__init_metrics()

  @staticmethod
  def discriminator_loss(real_logit, fake_logit):
    return tf.losses.mse(real_logit, tf.ones_like(real_logit)) + \
        tf.losses.mse(fake_logit, tf.zeros_like(fake_logit))

  @staticmethod
  def generator_loss(fake_logit):
    return tf.losses.mse(fake_logit, tf.ones_like(fake_logit))

  @staticmethod
  def bce_loss(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):
    def step_fn(inputs: dict):
      real_A, real_B = inputs['photo'], inputs['cartoon']
      with tf.GradientTape() as d_tape:
        # train discriminator
        fake_A2B, _, _ = self.genA2B(real_A)
        fake_B2A, _, _ = self.genB2A(real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        D_ad_loss_GA = self.discriminator_loss(real_GA_logit, fake_GA_logit)

        D_ad_cam_loss_GA = self.discriminator_loss(real_GA_cam_logit, fake_GA_cam_logit)
        D_ad_loss_LA = self.discriminator_loss(real_LA_logit, fake_LA_logit)
        D_ad_cam_loss_LA = self.discriminator_loss(real_LA_cam_logit, fake_LA_cam_logit)
        D_ad_loss_GB = self.discriminator_loss(real_GB_logit, fake_GB_logit)
        D_ad_cam_loss_GB = self.discriminator_loss(real_GB_cam_logit, fake_GB_cam_logit)
        D_ad_loss_LB = self.discriminator_loss(real_LB_logit, fake_LB_logit)
        D_ad_cam_loss_LB = self.discriminator_loss(real_LB_cam_logit, fake_LB_cam_logit)

        D_loss_A = self.hparams.adv_weight * \
            (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.hparams.adv_weight * \
            (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
        Discriminator_loss = D_loss_A + D_loss_B

      self.optimizer_minimize(Discriminator_loss, d_tape, self.d_optimizer,
                              [self.disGA, self.disGB, self.disLA, self.disLB])

      with tf.GradientTape() as g_tape:
        # train generator
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        G_ad_loss_GA = self.generator_loss(fake_GA_logit)
        G_ad_cam_loss_GA = self.generator_loss(fake_GA_cam_logit)
        G_ad_loss_LA = self.generator_loss(fake_LA_logit)
        G_ad_cam_loss_LA = self.generator_loss(fake_LA_cam_logit)
        G_ad_loss_GB = self.generator_loss(fake_GB_logit)
        G_ad_cam_loss_GB = self.generator_loss(fake_GB_cam_logit)
        G_ad_loss_LB = self.generator_loss(fake_LB_logit)
        G_ad_cam_loss_LB = self.generator_loss(fake_LB_cam_logit)

        G_recon_loss_A = tf.abs(fake_A2B2A - real_A)
        G_recon_loss_B = tf.abs(fake_B2A2B - real_B)

        G_identity_loss_A = tf.abs(fake_A2A - real_A)
        G_identity_loss_B = tf.abs(fake_B2B - real_B)
        # todo 这里需要修改
        G_id_loss_A = self.facenet.cosine_distance(real_A, fake_A2B)
        G_id_loss_B = self.facenet.cosine_distance(real_B, fake_B2A)

        G_cam_loss_A = (self.bce_loss(tf.ones_like(fake_B2A_cam_logit), fake_B2A_cam_logit) +
                        self.bce_loss(tf.zeros_like(fake_A2A_cam_logit), fake_A2A_cam_logit))
        G_cam_loss_B = (self.bce_loss(tf.ones_like(fake_A2B_cam_logit), fake_A2B_cam_logit) +
                        self.bce_loss(tf.zeros_like(fake_B2B_cam_logit), fake_B2B_cam_logit))

        G_loss_A = (self.hparams.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA +
                                               G_ad_loss_LA + G_ad_cam_loss_LA) +
                    self.hparams.cycle_weight * G_recon_loss_A +
                    self.hparams.identity_weight * G_identity_loss_A +
                    self.hparams.cam_weight * G_cam_loss_A +
                    self.hparams.faceid_weight * G_id_loss_A)
                    
        G_loss_B = (self.hparams.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB +
                                               G_ad_loss_LB + G_ad_cam_loss_LB) +
                    self.hparams.cycle_weight * G_recon_loss_B +
                    self.hparams.identity_weight * G_identity_loss_B +
                    self.hparams.cam_weight * G_cam_loss_B +
                    self.hparams.faceid_weight * G_id_loss_B)

        Generator_loss = G_loss_A + G_loss_B
      self.optimizer_minimize(Generator_loss, g_tape, self.g_optimizer, [self.genA2B, self.genB2A])

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  def val_step(self, dataset, metrics):
    return super().val_step(dataset, metrics)
