from tools.training_engine import BaseHelperV2, GanBaseTrainingLoop
from pathlib import Path
import tensorflow as tf
k = tf.keras
K = tf.keras.backend
kl = tf.keras.layers
from transforms.image.ops import normalize, renormalize
from typing import List
from functools import partial
import numpy as np
import cv2


class PhotoTransferHelper(BaseHelperV2):
  """ 
  hparams: 
    num_parallel_calls: -1
  """
  def set_datasetlist(self):
    dataset_root = np.load(self.dataset_root, allow_pickle=True)[()]
    self.trainA = dataset_root['trainA']
    self.trainB = dataset_root['trainB']
    self.testA = dataset_root['testA']
    self.testB = dataset_root['testB']
    self.train_total_data = dataset_root['train_num']
    self.test_total_data = dataset_root['test_num']
    self.val_total_data = self.test_total_data

  def build_train_datapipe(self, batch_size, is_augment, is_normalize=True, is_training=True):
    def map_fn(stream):
      features = tf.io.parse_single_example(
          stream,
          {'img_raw': tf.io.FixedLenFeature([], tf.string)})
      img_str = features['img_raw']
      img = tf.image.decode_png(img_str, channels=3)
      if is_augment:
        img = tf.image.resize(img, (self.in_hw[0] + 15, self.in_hw[1] + 15))
        img = tf.image.random_crop(img, (self.in_hw[0], self.in_hw[1], 3))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.3)
      else:
        img = tf.image.resize(img, self.in_hw)

      if is_normalize:
        img = normalize(tf.cast(img, tf.float32), 127.5, 127.5)
      else:
        img = tf.cast(img, tf.float32)
      return img

    def map_two_fn(stream_a, stream_b):
      img_a = map_fn(stream_a)
      img_b = map_fn(stream_b)
      return {'realA': img_a, 'realB': img_b}

    if is_training:
      ds = (tf.data.Dataset.zip((tf.data.TFRecordDataset(self.trainA),
                                 tf.data.TFRecordDataset(self.trainB))).
            shuffle(batch_size * 100).
            repeat().
            map(map_two_fn, self.hparams.num_parallel_calls).
            batch(batch_size, True).
            prefetch(tf.data.experimental.AUTOTUNE))
    else:
      ds = (tf.data.Dataset.zip((tf.data.TFRecordDataset(self.testA),
                                 tf.data.TFRecordDataset(self.testB))).
            shuffle(100).
            repeat().
            map(map_two_fn, -1).
            batch(batch_size, True))

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    return ds

  def build_val_datapipe(self, batch_size, is_normalize=True):
    return self.build_train_datapipe(1, is_augment=False,
                                     is_normalize=is_normalize,
                                     is_training=False)


class PhotoTransferLoop(GanBaseTrainingLoop):
  """ PhotoTransferLoop

  Args:
      hparams:
        adv_weight: 1 # Weight for GAN
        cycle_weight: 50 # Weight for Cycle
        identity_weight: 10 # Weight for Identity
        cam_weight: 1000 # Weight for CAM
        faceid_weight: 1 # Weight for Face ID
        in_hw: [112,112] # input image hw
        face_model_path: tmp/face_model.h5 # face model path
  """

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
    super()._BaseTrainingLoop__init_kwargs(kwargs)
    super()._BaseTrainingLoop__init_metrics()

  def set_metrics_dict(self):
    d = {
        'train': {
            'g_loss': tf.keras.metrics.Mean('g_loss', dtype=tf.float32),
            'd_loss': tf.keras.metrics.Mean('d_loss', dtype=tf.float32),
        },
        'val': {},
        'debug': {}
    }
    return d

  def local_variables_init(self):
    self.face_model: k.Model = k.models.load_model(self.hparams.face_model_path)

  def face_id_loss(self, img_a, img_b):
    if self.hparams.in_hw[0] > 112:
      img_a = tf.image.resize(img_a, (112, 112))
      img_b = tf.image.resize(img_b, (112, 112))
    emba = self.face_model(img_a, training=False)
    embb = self.face_model(img_b, training=False)
    emba = tf.nn.l2_normalize(emba, -1)
    embb = tf.nn.l2_normalize(embb, -1)
    return tf.reduce_mean(tf.square(emba - embb))

  @staticmethod
  def discriminator_loss(real_logit, fake_logit):
    return tf.reduce_mean(tf.losses.mse(real_logit, tf.ones_like(real_logit)) +
                          tf.losses.mse(fake_logit, tf.zeros_like(fake_logit)))

  @staticmethod
  def generator_loss(fake_logit):
    return tf.reduce_mean(tf.losses.mse(fake_logit, tf.ones_like(fake_logit)))

  @staticmethod
  def bce_loss(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    # def step_fn(inputs: dict):
    #   real_A, real_B = inputs['realA'], inputs['realB']

    #   with tf.GradientTape() as d_tape:
    #     # train discriminator
    #     fake_A2B, _, _ = self.genA2B(real_A, training=True)
    #     fake_B2A, _, _ = self.genB2A(real_B, training=True)

    #     real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A, training=True)
    #     real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A, training=True)
    #     real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B, training=True)
    #     real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B, training=True)

    #     fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A, training=True)
    #     fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A, training=True)
    #     fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B, training=True)
    #     fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B, training=True)
    #     D_ad_loss_GA = self.discriminator_loss(real_GA_logit, fake_GA_logit)

    #     D_ad_cam_loss_GA = self.discriminator_loss(real_GA_cam_logit, fake_GA_cam_logit)
    #     D_ad_loss_LA = self.discriminator_loss(real_LA_logit, fake_LA_logit)
    #     D_ad_cam_loss_LA = self.discriminator_loss(real_LA_cam_logit, fake_LA_cam_logit)
    #     D_ad_loss_GB = self.discriminator_loss(real_GB_logit, fake_GB_logit)
    #     D_ad_cam_loss_GB = self.discriminator_loss(real_GB_cam_logit, fake_GB_cam_logit)
    #     D_ad_loss_LB = self.discriminator_loss(real_LB_logit, fake_LB_logit)
    #     D_ad_cam_loss_LB = self.discriminator_loss(real_LB_cam_logit, fake_LB_cam_logit)

    #     D_loss_A = (self.hparams.adv_weight * (D_ad_loss_GA +
    #                                            D_ad_cam_loss_GA +
    #                                            D_ad_loss_LA +
    #                                            D_ad_cam_loss_LA))
    #     D_loss_B = (self.hparams.adv_weight * (D_ad_loss_GB +
    #                                            D_ad_cam_loss_GB +
    #                                            D_ad_loss_LB +
    #                                            D_ad_cam_loss_LB))
    #     Discriminator_loss = tf.reduce_mean(
    #         tf.reduce_sum(D_loss_A + D_loss_B, [1, 2]))

    #   scaled_d_loss = self.optimizer_minimize(Discriminator_loss, d_tape,
    #                                           self.d_optimizer,
    #                                           [self.disGA,
    #                                            self.disGB,
    #                                            self.disLA,
    #                                            self.disLB])

    #   with tf.GradientTape() as g_tape:
    #     # train generator
    #     fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A, training=True)
    #     fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B, training=True)

    #     fake_A2B2A, _, _ = self.genB2A(fake_A2B, training=True)
    #     fake_B2A2B, _, _ = self.genA2B(fake_B2A, training=True)

    #     fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A, training=True)
    #     fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B, training=True)

    #     fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A, training=True)
    #     fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A, training=True)
    #     fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B, training=True)
    #     fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B, training=True)

    #     G_ad_loss_GA = self.generator_loss(fake_GA_logit)
    #     G_ad_cam_loss_GA = self.generator_loss(fake_GA_cam_logit)
    #     G_ad_loss_LA = self.generator_loss(fake_LA_logit)
    #     G_ad_cam_loss_LA = self.generator_loss(fake_LA_cam_logit)
    #     G_ad_loss_GB = self.generator_loss(fake_GB_logit)
    #     G_ad_cam_loss_GB = self.generator_loss(fake_GB_cam_logit)
    #     G_ad_loss_LB = self.generator_loss(fake_LB_logit)
    #     G_ad_cam_loss_LB = self.generator_loss(fake_LB_cam_logit)

    #     G_recon_loss_A = tf.abs(fake_A2B2A - real_A)
    #     G_recon_loss_B = tf.abs(fake_B2A2B - real_B)

    #     G_identity_loss_A = tf.abs(fake_A2A - real_A)
    #     G_identity_loss_B = tf.abs(fake_B2B - real_B)

    #     G_id_loss_A = self.face_id_loss(real_A, fake_A2B)
    #     G_id_loss_B = self.face_id_loss(real_B, fake_B2A)

    #     G_cam_loss_A = (self.bce_loss(tf.ones_like(fake_B2A_cam_logit),
    #                                   fake_B2A_cam_logit) +
    #                     self.bce_loss(tf.zeros_like(fake_A2A_cam_logit),
    #                                   fake_A2A_cam_logit))
    #     G_cam_loss_B = (self.bce_loss(tf.ones_like(fake_A2B_cam_logit),
    #                                   fake_A2B_cam_logit) +
    #                     self.bce_loss(tf.zeros_like(fake_B2B_cam_logit),
    #                                   fake_B2B_cam_logit))

    #     G_loss_A = (self.hparams.adv_weight * tf.reduce_sum(
    #         G_ad_loss_GA + G_ad_cam_loss_GA +
    #         G_ad_loss_LA + G_ad_cam_loss_LA, [1, 2]) +
    #         self.hparams.cycle_weight * tf.reduce_sum(G_recon_loss_A, [1, 2, 3]) +
    #         self.hparams.identity_weight * tf.reduce_sum(G_identity_loss_A, [1, 2, 3]) +
    #         self.hparams.cam_weight * tf.reduce_sum(G_cam_loss_A, [1]) +
    #         self.hparams.faceid_weight * G_id_loss_A)

    #     G_loss_B = (self.hparams.adv_weight * tf.reduce_sum(
    #         G_ad_loss_GB + G_ad_cam_loss_GB +
    #         G_ad_loss_LB + G_ad_cam_loss_LB, [1, 2]) +
    #         self.hparams.cycle_weight * tf.reduce_sum(G_recon_loss_B, [1, 2, 3]) +
    #         self.hparams.identity_weight * tf.reduce_sum(G_identity_loss_B, [1, 2, 3]) +
    #         self.hparams.cam_weight * tf.reduce_sum(G_cam_loss_B, [1]) +
    #         self.hparams.faceid_weight * G_id_loss_B)

    #     Generator_loss = tf.reduce_mean(G_loss_A + G_loss_B)

    #   scaled_g_loss = self.optimizer_minimize(Generator_loss, g_tape,
    #                                           self.g_optimizer,
    #                                           [self.genA2B,
    #                                            self.genB2A])

    #   if self.hparams.ema.enable:
    #     self.ema.update()

    #   metrics.g_loss.update_state(scaled_g_loss)
    #   metrics.d_loss.update_state(scaled_d_loss)

    def step_fn(inputs: dict):
      real_A, real_B = inputs['realA'], inputs['realB']

      with tf.GradientTape(persistent=True) as tape:
        # train discriminator
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A, training=True)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B, training=True)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A, training=True)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A, training=True)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B, training=True)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B, training=True)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A, training=True)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A, training=True)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B, training=True)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B, training=True)

        D_ad_loss_GA = self.discriminator_loss(real_GA_logit, fake_GA_logit)
        D_ad_cam_loss_GA = self.discriminator_loss(real_GA_cam_logit, fake_GA_cam_logit)
        D_ad_loss_LA = self.discriminator_loss(real_LA_logit, fake_LA_logit)
        D_ad_cam_loss_LA = self.discriminator_loss(real_LA_cam_logit, fake_LA_cam_logit)
        D_ad_loss_GB = self.discriminator_loss(real_GB_logit, fake_GB_logit)
        D_ad_cam_loss_GB = self.discriminator_loss(real_GB_cam_logit, fake_GB_cam_logit)
        D_ad_loss_LB = self.discriminator_loss(real_LB_logit, fake_LB_logit)
        D_ad_cam_loss_LB = self.discriminator_loss(real_LB_cam_logit, fake_LB_cam_logit)

        D_loss_A = (self.hparams.adv_weight * (D_ad_loss_GA +
                                               D_ad_cam_loss_GA +
                                               D_ad_loss_LA +
                                               D_ad_cam_loss_LA))
        D_loss_B = (self.hparams.adv_weight * (D_ad_loss_GB +
                                               D_ad_cam_loss_GB +
                                               D_ad_loss_LB +
                                               D_ad_cam_loss_LB))
        Discriminator_loss = D_loss_A + D_loss_B

        # train generator

        fake_A2B2A, _, _ = self.genB2A(fake_A2B, training=True)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A, training=True)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A, training=True)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B, training=True)

        G_ad_loss_GA = self.generator_loss(fake_GA_logit)
        G_ad_cam_loss_GA = self.generator_loss(fake_GA_cam_logit)
        G_ad_loss_LA = self.generator_loss(fake_LA_logit)
        G_ad_cam_loss_LA = self.generator_loss(fake_LA_cam_logit)
        G_ad_loss_GB = self.generator_loss(fake_GB_logit)
        G_ad_cam_loss_GB = self.generator_loss(fake_GB_cam_logit)
        G_ad_loss_LB = self.generator_loss(fake_LB_logit)
        G_ad_cam_loss_LB = self.generator_loss(fake_LB_cam_logit)
        # 循环一致损失
        G_recon_loss_A = tf.abs(fake_A2B2A - real_A)
        G_recon_loss_B = tf.abs(fake_B2A2B - real_B)
        # 一致损失
        G_identity_loss_A = tf.abs(fake_A2A - real_A)
        G_identity_loss_B = tf.abs(fake_B2B - real_B)

        # 人脸id损失
        G_id_loss_A = self.face_id_loss(real_A, fake_A2B)
        G_id_loss_B = self.face_id_loss(real_B, fake_B2A)

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

      scaled_d_loss = self.optimizer_minimize(Discriminator_loss, tape,
                                              self.d_optimizer,
                                              [self.disGA,
                                               self.disGB,
                                               self.disLA,
                                               self.disLB])

      scaled_g_loss = self.optimizer_minimize(Generator_loss, tape,
                                              self.g_optimizer,
                                              [self.genA2B,
                                               self.genB2A])

      if self.hparams.ema.enable:
        self.ema.update()

      metrics.g_loss.update_state(scaled_g_loss)
      metrics.d_loss.update_state(scaled_d_loss)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  @staticmethod
  def cam(x, size):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, tuple(size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img.astype(np.uint8)

  def visual_img_cam(self, img: np.ndarray, cam: np.ndarray) -> [np.ndarray, np.ndarray]:
    return [renormalize(img, 127.5, 127.5).astype(np.uint8), self.cam(cam, self.hparams.in_hw)]

  def val_step(self, dataset, metrics):
    A2Bs = []
    B2As = []
    for inputs in dataset.take(5):
      real_A, real_B = inputs['realA'], inputs['realB']

      fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A, training=False)
      fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B, training=False)
      fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A, training=False)
      real_A = real_A.numpy()
      fake_A2B, fake_A2B_heatmap = fake_A2B.numpy(), fake_A2B_heatmap.numpy()
      fake_A2B2A, fake_A2B2A_heatmap = fake_A2B2A.numpy(), fake_A2B2A_heatmap.numpy()
      fake_A2A, fake_A2A_heatmap = fake_A2A.numpy(), fake_A2A_heatmap.numpy()

      A2B = np.concatenate(
          self.visual_img_cam(real_A[0], fake_A2A_heatmap[0]) +
          self.visual_img_cam(fake_A2A[0], fake_A2B_heatmap[0]) +
          self.visual_img_cam(fake_A2B[0], fake_A2B2A_heatmap[0]) +
          [renormalize(fake_A2B2A[0], 127.5, 127.5).astype(np.uint8)], axis=1)

      fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B, training=False)
      fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A, training=False)
      fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B, training=False)
      real_B = real_B.numpy()
      fake_B2A, fake_B2A_heatmap = fake_B2A.numpy(), fake_B2A_heatmap.numpy()
      fake_B2A2B, fake_B2A2B_heatmap = fake_B2A2B.numpy(), fake_B2A2B_heatmap.numpy()
      fake_B2B, fake_B2B_heatmap = fake_B2B.numpy(), fake_B2B_heatmap.numpy()

      B2A = np.concatenate(
          self.visual_img_cam(real_B[0], fake_B2B_heatmap[0]) +
          self.visual_img_cam(fake_B2B[0], fake_B2A_heatmap[0]) +
          self.visual_img_cam(fake_B2A[0], fake_B2A2B_heatmap[0]) +
          [renormalize(fake_B2A2B[0], 127.5, 127.5).astype(np.uint8)], axis=1)

      A2Bs.append(A2B)
      B2As.append(B2A)
    self.summary.save_images({'A2B': np.stack(A2Bs)}, 5)
    self.summary.save_images({'B2A': np.stack(B2As)}, 5)
