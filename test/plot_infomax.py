import tensorflow as tf
import sys
import os
sys.path.insert(0, os.getcwd())
import cycler
from tools.kerasdataset import InfoMaxLoop, InfoMaxSslV2Loop, KerasDatasetHelper
from tools.dcasetask5 import DCASETask5FixMatchSSLHelper
from transforms.image.ops import renormalize
from models.semisupervised import cifar_infomax_ssl_v1, ReShuffle
from models.networks import imageclassifierCNN13, compose
import numpy as np
from tools.plot_utils import plot_emmbeding, build_ball
from tools.prob_utils import js_divergence, kl_divergence
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.gridspec import GridSpec
from typing import List, Tuple
import scipy.stats as st

np.set_printoptions(suppress=True)


def save_some_data_for_plot():
  """ 先生成一些数据为绘图做准备 """
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  h = KerasDatasetHelper(**{'dataset': 'cifar10',
                            'label_ratio': 0.05,
                            'unlabel_dataset_ratio': 6,
                            'mixed_precision_dtype': 'float32',
                            'augment_kwargs': {
                                'name': 'ctaugment',
                                'kwarg': {'num_layers': 3,
                                          'confidence_threshold': 0.8,
                                          'decay': 0.99,
                                          'epsilon': 0.001,
                                          'prob_to_apply': None,
                                          'num_levels': 10}}})

  h.set_dataset(32, True, True)
  h.augmenter
  tf.train.Checkpoint()

  iters = iter(h.val_dataset)

  datadict = {'val': {}, 'train': {}}
  for i in range(16):
    d = next(iters)
    datadict['val'].setdefault(i, d)

  iters = iter(h.train_dataset)
  d = next(iters)
  for i in range(16):
    d = next(iters)
    datadict['train'].setdefault(i, d)

  np.save('tmp/some_data.npy', datadict, allow_pickle=True)
  # d.keys()
  # ['probe_op_indices', 'probe_op_args', 'probe_data', 'data', 'label', 'unsup_data', 'unsup_aug_data']


def plot_infomatch_train_emmbeding():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  infer_model, val_model, train_model = cifar_infomax_ssl_v1(
      input_shape=[32, 32, 3],
      nclasses=10,
      softmax=False,
      z_dim=256,
      weight_decay=0.0005)

  train_model.load_weights(
      'log/default_cifar10_infomaxv2_ctaug_exp/train_model_99.h5', by_name=True)

  datadict = np.load('tmp/some_data.npy', allow_pickle=True)[()]
  d = datadict['train'][0]
  # d.keys() ['probe_op_indices', 'probe_op_args', 'probe_data', 'data', 'label', 'unsup_data', 'unsup_aug_data', 'unsup_label']

  (suplogits, supz_mean, supz_log_sigma, supzz_true_scores, supzz_false_scores, supzz_label,
   supzf_true_scores, supzf_false_scores, supzf_label) = train_model.predict(d['data'])

  (unsuplogits, unsupz_mean, unsupz_log_sigma, unsupzz_true_scores, unsupzz_false_scores, unsupzz_label,
   unsupzf_true_scores, unsupzf_false_scores, unsupzf_label) = train_model.predict(d['unsup_data'])

  (auglogits, augz_mean, augz_log_sigma, augzz_true_scores, augzz_false_scores, augzz_label,
   augzf_true_scores, augzf_false_scores, augzf_label) = train_model.predict(d['unsup_aug_data'])

  pca = PCA(3)
  pca_unsuplogits = pca.fit_transform(tf.nn.softmax(unsuplogits, -1).numpy())
  pca_unsuplogits = pca_unsuplogits / np.linalg.norm(pca_unsuplogits, ord=2, axis=-1, keepdims=True)

  pca_auglogits = pca.transform(tf.nn.softmax(auglogits, -1).numpy())
  pca_auglogits = pca_auglogits / np.linalg.norm(pca_auglogits, ord=2, axis=-1, keepdims=True)

  # fig = plt.figure(figsize=[4, 4])
  # ax: axes3d.Axes3D = fig.add_subplot(1, 1, 1, projection='3d')  # type: Axes3D
  # ax.view_init(elev=25., azim=120.)
  # build_ball(ax)
  # ax.scatter(pca_auglogits[:, 0], pca_auglogits[:, 1], pca_auglogits[:, 2], label='1')
  # ax.scatter(pca_unsuplogits[:, 0], pca_unsuplogits[:, 1], pca_unsuplogits[:, 2], label='2')
  # plt.tight_layout()
  # plt.show()

  plt.hist(unsuplogits.ravel(), bins=30)
  plt.hist(auglogits.ravel(), bins=30)

  plt.hist(unsupz_mean.ravel(), bins=200, alpha=0.3)
  plt.hist(augz_mean.ravel(), bins=200, alpha=0.3)

  # logits.shape  # [32,10]
  # z_mean.shape  # [32,10]
  # z_mean.numpy().mean()  # -0.0003257175
  # z_mean.numpy().std()  # 0.72906566
  # unsupz_mean.mean()
  # unsupz_mean.std()

  # js_divergence(unsuplogits, auglogits)
  np.save('tmp/infomatch_hidden.npy', {
      'unsupz_mean': unsupz_mean.ravel(),
      'augz_mean': augz_mean.ravel(),
      'unsup_data': d['unsup_data'],
      'unsup_aug_data': d['unsup_aug_data'],
      'unsupzf_true_scores': unsupzf_true_scores,
      'unsupzf_false_scores': unsupzf_false_scores,
      'unsupzf_label': unsupzf_label,
      'augzf_true_scores': augzf_true_scores,
      'augzf_false_scores': augzf_false_scores,
      'augzf_label': augzf_label
  })


def plot_mixmatch_train_emmbeding():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  infer_model, val_model, train_model = imageclassifierCNN13(
      input_shape=[32, 32, 3],
      nclasses=10,
      filters=32)

  train_model.load_weights(
      'log/default_cifar10_mixmatch_ctaug_exp/train_model_99.h5', by_name=True)

  datadict = np.load('tmp/some_data.npy', allow_pickle=True)[()]
  # d.keys() ['probe_op_indices', 'probe_op_args', 'probe_data', 'data', 'label', 'unsup_data', 'unsup_aug_data', 'unsup_label']

  encoder = tf.keras.backend.function(
      train_model.inputs[0], train_model.get_layer(name='lambda').output)

  with tf.keras.backend.learning_phase_scope(0):
    unsupz_means = []
    augz_means = []
    for i in range(8):
      d = datadict['train'][i]
      supz_mean = encoder(d['data'])
      unsupz_mean = encoder(d['unsup_data'])
      augz_mean = encoder(d['unsup_aug_data'])
      unsupz_means.append(unsupz_mean.ravel())
      augz_means.append(augz_mean.ravel())

  unsupz_means = np.array(unsupz_means).ravel()
  augz_means = np.array(augz_means).ravel()
  # plt.hist(unsupz_means, bins=200, alpha=0.3)
  # plt.hist(augz_means, bins=200, alpha=0.3)

  np.save('tmp/mixmatch_hidden.npy', {'unsupz_mean': unsupz_means,
                                      'augz_mean': augz_means})


def plot_uda_train_emmbeding():
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  infer_model, val_model, train_model = imageclassifierCNN13(
      input_shape=[32, 32, 3],
      nclasses=10,
      filters=32)

  train_model.load_weights(
      'log/default_cifar10_uda_ctaug_exp/train_model_99.h5', by_name=True)

  datadict = np.load('tmp/some_data.npy', allow_pickle=True)[()]
  # d.keys() ['probe_op_indices', 'probe_op_args', 'probe_data', 'data', 'label', 'unsup_data', 'unsup_aug_data', 'unsup_label']

  encoder = tf.keras.backend.function(
      train_model.inputs[0], train_model.get_layer(name='lambda').output)

  with tf.keras.backend.learning_phase_scope(0):
    unsupz_means = []
    augz_means = []
    for i in range(8):
      d = datadict['train'][i]
      supz_mean = encoder(d['data'])
      unsupz_mean = encoder(d['unsup_data'])
      augz_mean = encoder(d['unsup_aug_data'])
      unsupz_means.append(unsupz_mean.ravel())
      augz_means.append(augz_mean.ravel())

  unsupz_means = np.array(unsupz_means).ravel()
  augz_means = np.array(augz_means).ravel()
  # plt.hist(unsupz_means, bins=200, alpha=0.3)
  # plt.hist(augz_means, bins=200, alpha=0.3)

  np.save('tmp/uda_hidden.npy', {'unsupz_mean': unsupz_means,
                                 'augz_mean': augz_means})


def build_ball(ax):
  # 改变摄像头角度
  xlm = ax.get_xlim3d()
  ylm = ax.get_ylim3d()
  zlm = ax.get_zlim3d()
  ax.set_xlim3d(-.82, 0.82)  # 设置距离
  ax.set_ylim3d(-.82, 0.82)  # 设置距离
  ax.set_zlim3d(-.82, 0.82)  # 设置距离
  # 关闭网格线
  # First remove fill
  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = False

  # Now set color to white (or whatever is "invisible")
  ax.xaxis.pane.set_edgecolor('w')
  ax.yaxis.pane.set_edgecolor('w')
  ax.zaxis.pane.set_edgecolor('w')

  # Bonus: To get rid of the grid as well:
  ax.grid(False)

  # 设置刻度
  ax.set_xticks([-0.5, 0, 0.5])
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_zticks([-1, -0.5, 0, 0.5, 1])

  # 绘制圆球
  # 球体坐标公式
  φ = np.linspace(0, np.deg2rad(360), 12)
  θ = np.linspace(0, np.deg2rad(180), 12)
  x = 1 * np.outer(np.cos(φ), np.sin(θ))
  y = 1 * np.outer(np.sin(φ), np.sin(θ))
  z = 1 * np.outer(np.ones(np.size(θ)), np.cos(θ))

  ax.plot_wireframe(x, y, z, colors='dimgray', alpha=0.3, linestyles='-', linewidths=1)


def random_neighborhood_point_on_ball(point: np.ndarray, theta: float = 5, phi: float = 5) -> np.ndarray:
  range_theta = np.deg2rad(theta) / 2
  range_phi = np.deg2rad(phi) / 2

  # (x,y,z) -> (r,θ,φ)
  # 1 = np.linalg.norm(point, 2, -1, keepdims=True)
  θ = np.arccos(point[:, 2:3] / 1)
  φ = np.arctan2(point[:, 1:2], point[:, 0:1])

  # random assign point
  nθ = θ + np.random.uniform(low=-range_theta, high=range_theta, size=θ.shape)
  nφ = φ + np.random.uniform(low=-range_phi, high=range_phi, size=φ.shape)

  # (r,θ,φ) -> (x,y,z)
  x = 1 * np.cos(nφ) * np.sin(nθ)
  y = 1 * np.sin(nφ) * np.sin(nθ)
  z = 1 * np.cos(nθ)

  # make differ line
  dθ = np.linspace(θ.ravel(), nθ.ravel(), 8).T
  dφ = np.linspace(φ.ravel(), nφ.ravel(), 8).T
  dx = 1 * np.cos(dφ) * np.sin(dθ)
  dy = 1 * np.sin(dφ) * np.sin(dθ)
  dz = 1 * np.cos(dθ)

  return np.hstack([x, y, z]), [dx, dy, dz]


def plot_fake_ball():
  plt.style.use('seaborn-paper')
  # 使用 cycler 替换scatter的默认颜色循环
  plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', plt.cm.tab10(np.linspace(0, 1, 9)))
  plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', plt.cm.tab10(np.linspace(0, 1, 9)))
  plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示
  plt.rcParams['font.size'] = 13
  plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
  # xx-small, x-small, small, medium, large, x-large, xx-large, smaller, larger
  params = {'legend.fontsize': 'small',
            'axes.labelsize': 'small',
            'axes.titlesize': 'medium',
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small'}
  plt.rcParams.update(params)

  ls: List[Tuple[str, str]] = [
      ('ICT', 'log/ams-0.01'),
      ('Mix Match', 'log/cnn4'),
      ('Info Match', 'log/ams-0.35')
  ]

  fig: plt.Figure = plt.figure(figsize=[12, 12])
  gs = GridSpec(12, 15, figure=fig, top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
  axs: plt.Axes = [[], [], []]
  for i in range(3):
    axs[0].append(fig.add_subplot(gs[0:5, i * 5:(i + 1) * 5], projection='3d'))
    axs[0][i].view_init(elev=25., azim=-150.)
    axs[0][i].dist = 11
    axs[1].append(fig.add_subplot(gs[5:10, i * 5:(i + 1) * 5], projection='3d'))
    axs[1][i].view_init(elev=25., azim=-150.)
    axs[1][i].dist = 11

    build_ball(axs[0][i])
    build_ball(axs[1][i])

  axs[2].append(fig.add_subplot(gs[10:12, 1:4]))
  axs[2].append(fig.add_subplot(gs[10:12, 6:9]))
  axs[2].append(fig.add_subplot(gs[10:12, 11:14]))

  for i in range(3):
    axa, axb, title, path = axs[0][i], axs[1][i], ls[i][0], ls[i][1]
    num = 50
    n = 9
    np.random.seed(1111)
    ams = np.load(f'/home/zqh/workspace/dcase2018-task5/{path}/result.npz', allow_pickle=True)
    y_true = ams['y_true']
    y_pred = ams['y_pred']
    vec = ams['vec']

    # 设置颜色循环

    # NOTE 找到所有预测正确，且y_true等于指定数
    for j in range(n):
      boolmask = np.logical_and((y_pred == y_true), y_true == j)
      try:
        idx = np.random.choice(np.where(boolmask == True)[0], num)
      except ValueError as e:
        idx = []

      res = vec[idx]

      if len(res) > 0:
        if 'caps' in path and 'ams' not in path:
          v = res[:, j] / np.linalg.norm(res[:, j], ord=2,
                                         axis=-1, keepdims=True)
        else:
          v = res / np.linalg.norm(res, ord=2,
                                   axis=-1, keepdims=True)

        axa.scatter(v[:, 0], v[:, 1], v[:, 2], c=f'C{j}', marker='o', label=f'{j}')
        nv, line = random_neighborhood_point_on_ball(v, 25 - i * 5, 25 - i * 5)
        axb.scatter(nv[:, 0], nv[:, 1], nv[:, 2], c=f'C{j}', marker='^', label=f'{j}')
        # for dx, dy, dz in zip(*line):
        #   ax.plot(dx, dy, dz, color='deeppink', linestyle='dashed')

    axa.set_title(title)

  for i, nm in enumerate(['tmp/uda_hidden.npy', 'tmp/mixmatch_hidden.npy', 'tmp/infomatch_hidden.npy']):
    d = np.load(nm, allow_pickle=True)[()]
    unsupz_means = d['unsupz_mean']
    augz_means = d['augz_mean']
    ax = axs[2][i]
    ax.hist(unsupz_means, bins=200, alpha=0.3, label='original')
    ax.hist(augz_means, bins=200, alpha=0.3, label='augmented')
    rvp = st.norm(loc=unsupz_means.mean(), scale=unsupz_means.std())
    rvq = st.norm(loc=augz_means.mean(), scale=augz_means.std())
    # 0.0003450555490592956
    # 0.00020686891358053482
    # 0.00011651471338703982
    print(js_divergence(rvp.pdf(unsupz_means), rvq.pdf(unsupz_means), from_logits=True))

  ax.legend(loc='upper left')

  plt.tight_layout()
  plt.show()
  fig.savefig('/media/zqh/Documents/研究生-各种资料/基于互信息最大化的半监督的音频分类算法/reslut.svg',
              transparent=True, bbox_inches='tight', pad_inches=0)


def show_n_img(datas, nrows=1, ncols=1, figsize=(9, 9)):
  fig: plt.Figure = plt.figure(figsize=figsize)
  axs = fig.subplots(nrows, ncols, squeeze=False)
  for i in range(nrows):
    for j in range(ncols):
      axs[i, j].imshow(datas[i, j])
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])
  fig.tight_layout()


def train_local_disc_model():
  """ 我觉得还是得训练一个局部判别器,不然实在难搞 """
  from transforms.audio import ops
  from models.semisupervised import ReShuffle
  kl = tf.keras.layers
  tf.random.set_seed(10101)
  np.random.seed(10101)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  h = DCASETask5FixMatchSSLHelper(**{'data_ann': 'data/dcasetask5_ann_list.npy',
                                     'in_hw': [40, 501],
                                     'nclasses': 9,
                                     'unlabel_dataset_ratio': 3,
                                     'augment_kwargs': {
                                         'name': 'ctaugment',
                                         'kwarg': {'num_layers': 3,
                                                   'confidence_threshold': 0.8,
                                                   'decay': 0.99,
                                                   'epsilon': 0.001,
                                                   'prob_to_apply': None,
                                                   'num_levels': 10}}})

  h.set_dataset(16, True)
  # ds = h.train_dataset.map(lambda d: tf.image.resize(d['unsup_data'], (128, 512)))
  ds = h.train_dataset.map(lambda d: tf.image.resize(d['unsup_aug_data'], (128, 512)))
  # iters = iter(ds)
  # data = next(iters)
  # unsup = tf.image.resize(data['unsup_data'], (128, 501))[..., 0]

  inputs = tf.keras.Input((128, 512, 1))

  local_fmap = compose(
      kl.Conv2D(32, 7, 2, padding='same'),
      kl.ReLU(6.),
      kl.BatchNormalization(),
      kl.MaxPool2D(),
      kl.Conv2D(256, 3, 2, padding='same'),
      kl.ReLU(6.),
      kl.BatchNormalization(),
      kl.Conv2D(256, 3, 2, padding='same'),
      kl.ReLU(6.),
      kl.BatchNormalization(),
      kl.Conv2D(256, 3, 2, padding='same'),
      kl.ReLU(6.),
      kl.BatchNormalization(),
  )(inputs)

  global_fmap = compose(
      kl.Conv2D(512, 3, 2, padding='same'),
      kl.ReLU(6.),
      kl.BatchNormalization(),
      kl.GlobalAveragePooling2D(),
  )(local_fmap)

  (zf_label, fmap_shuffle) = ReShuffle(name='shuffling_local')(local_fmap)
  z_samples_map = compose(
      kl.RepeatVector(4 * 16),
      kl.Reshape((4, 16, 512)),
  )(global_fmap)

  zf_true = kl.Concatenate()([z_samples_map, local_fmap])  # loacl true pair
  zf_false = kl.Concatenate()([z_samples_map, fmap_shuffle])  # loacl false pair
  localdiscriminator = tf.keras.Sequential([
      kl.InputLayer(input_shape=[None, None, 512 + 256]),
      kl.Dense(512, activation='relu'),
      kl.Dense(512, activation='relu'),
      kl.Dense(512, activation='relu'),
      kl.Dense(1, activation='sigmoid'),
  ])
  zf_true_scores = localdiscriminator(zf_true)
  zf_false_scores = localdiscriminator(zf_false)
  train_model = tf.keras.Model(inputs, [local_fmap, global_fmap])
  train_model.add_loss(InfoMaxLoop.info_loss(zf_true_scores, zf_false_scores, zf_label))

  train_model.compile(optimizer='adam')
  train_model.fit(ds, epochs=20, steps_per_epoch=h.train_epoch_step)

  tf.keras.models.save_model(train_model, 'tmp/local_infomatch.h5')
  tf.keras.models.save_model(localdiscriminator, 'tmp/local_discriminator.h5')


def plot_fuck_loacl_matul_info_v1():
  from transforms.audio import ops
  tf.random.set_seed(112)
  np.random.seed(112)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  h = DCASETask5FixMatchSSLHelper(**{'data_ann': 'data/dcasetask5_ann_list.npy',
                                     'in_hw': [40, 501],
                                     'nclasses': 9,
                                     'unlabel_dataset_ratio': 3,
                                     'augment_kwargs': {
                                         'name': 'ctaugment',
                                         'kwarg': {'num_layers': 3,
                                                   'confidence_threshold': 0.8,
                                                   'decay': 0.99,
                                                   'epsilon': 0.001,
                                                   'prob_to_apply': None,
                                                   'num_levels': 10}}})

  h.set_dataset(16, True)
  iters = iter(h.train_dataset)
  data = next(iters)
  unsup = tf.image.resize(data['unsup_data'], (128, 512))[..., 0]

  train_model = tf.keras.models.load_model(
      'tmp/local_infomatch.h5', custom_objects={'ReShuffle': ReShuffle})
  disc_model = tf.keras.models.load_model(
      'tmp/local_discriminator.h5', custom_objects={'ReShuffle': ReShuffle})

  factor1 = tf.random.uniform([unsup.shape[0]], minval=0.2, maxval=0.5) * 0.5
  factor2 = tf.random.uniform([unsup.shape[0]], minval=0.2, maxval=0.5) * 0.5
  unsupaug = []
  for uns, p1, p2 in zip(unsup, factor1, factor2):
    tmp = uns
    if tf.random.uniform([]) < 0.5:
      tmp = ops.freq_mask(tmp, p1)
    else:
      tmp = ops.time_mask(tmp, p2)
    if tf.random.uniform([]) < 0.2:
      if tf.random.uniform([]) < 0.5:
        tmp = ops.freq_mask(tmp, p2)
      else:
        tmp = ops.time_mask(tmp, p1)
    unsupaug.append(tmp)

  unsupaug = tf.stack(unsupaug)

  unsup_local_fmap, unsup_global_fmap = train_model.predict(unsup[..., None])
  unsupaug_local_fmap, unsupaug_global_fmap = train_model.predict(unsupaug[..., None])

  unsup_samples_map = compose(tf.keras.layers.RepeatVector(4 * 16),
                              tf.keras.layers.Reshape((4, 16, 512)))(unsup_global_fmap)
  unsupaug_samples_map = compose(tf.keras.layers.RepeatVector(4 * 16),
                                 tf.keras.layers.Reshape((4, 16, 512)),
                                 )(unsupaug_global_fmap)

  unsup_zf_true = tf.keras.layers.Concatenate()([unsup_samples_map, unsup_local_fmap])
  unsupaug_zf_true = tf.keras.layers.Concatenate()([unsupaug_samples_map, unsupaug_local_fmap])

  unsup_zf_score = disc_model.predict(unsup_zf_true)
  unsupaug_zf_score = disc_model.predict(unsupaug_zf_true)

  # for i in range(5):
  #   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
  #   ax1.imshow(unsup[i].numpy(), cmap='rainbow')
  #   ax2.imshow(1 - unsup_zf_score[i, ..., 0])
  #   # ax2.set_yticks(np.arange(4))

  #   ax3.imshow(unsupaug[i].numpy(), cmap='rainbow')
  #   ax4.imshow(1 - unsupaug_zf_score[i, ..., 0])
  #   # ax4.set_yticks(np.arange(4))
  #   plt.show()

  fig: plt.Figure = plt.figure(figsize=(3 * 4, 3 * 4))
  gs = GridSpec(4, 4, figure=fig)  # top=1, bottom=0, left=0, right=1, wspace=0, hspace=0
  axs: plt.Axes = [[], [], [], []]
  for i in range(4):
    axs[i].append(fig.add_subplot(gs[i, 0]))
    axs[i].append(fig.add_subplot(gs[i, 1]))
    axs[i].append(fig.add_subplot(gs[i, 2]))
    axs[i].append(fig.add_subplot(gs[i, 3]))

  axs[0][0].set_title('Mel Spectrogram')
  axs[0][1].set_title('Local MI Score')
  axs[0][2].set_title('Mel Spectrogram')
  axs[0][3].set_title('Local MI Score')

  for i, j, ax in zip([1, 4, 6, 9], [10, 12, 13, 15], axs):
    ax[0].pcolor(unsup[i].numpy(), cmap='rainbow')
    ax[1].pcolor(1 - unsup_zf_score[i, ..., 0])
    ax[2].pcolor(unsup[j].numpy(), cmap='rainbow')
    ax[3].pcolor(1 - unsup_zf_score[j, ..., 0])

  # fig.set_tight_layout('')
  # plt.title('Original')
  plt.tight_layout()
  # plt.show()
  plt.savefig('/media/zqh/Documents/研究生-各种资料/基于互信息最大化的半监督的音频分类算法/local_info_original.png',
              transparent=True, bbox_inches='tight', pad_inches=0)
  plt.clf()

  fig: plt.Figure = plt.figure(figsize=(3 * 4, 3 * 4))
  gs = GridSpec(4, 4, figure=fig)  # top=1, bottom=0, left=0, right=1, wspace=0, hspace=0
  axs: plt.Axes = [[], [], [], []]
  for i in range(4):
    axs[i].append(fig.add_subplot(gs[i, 0]))
    axs[i].append(fig.add_subplot(gs[i, 1]))
    axs[i].append(fig.add_subplot(gs[i, 2]))
    axs[i].append(fig.add_subplot(gs[i, 3]))

  axs[0][0].set_title('Mel Spectrogram')
  axs[0][1].set_title('Local MI Score')
  axs[0][2].set_title('Mel Spectrogram')
  axs[0][3].set_title('Local MI Score')

  for i, j, ax in zip([1, 4, 6, 9], [10, 12, 13, 15], axs):
    ax[0].pcolor(unsupaug[i].numpy(), cmap='rainbow')
    ax[1].pcolor(1 - unsupaug_zf_score[i, ..., 0])
    ax[2].pcolor(unsupaug[j].numpy(), cmap='rainbow')
    ax[3].pcolor(1 - unsupaug_zf_score[j, ..., 0])

  # fig.set_tight_layout('')
  # plt.title('Mask Augmented')
  plt.tight_layout()
  # plt.show()
  plt.savefig('/media/zqh/Documents/研究生-各种资料/基于互信息最大化的半监督的音频分类算法/local_info_masked.png',
              transparent=True, bbox_inches='tight', pad_inches=0)
  plt.clf()

  # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
  # # ax4.set_yticks(np.arange(4))
  # plt.show()

  # zfscore = tf.image.resize(unsup[..., None], [4, 16])[..., 0].numpy()
  # zf_augscore = tf.image.resize(unsupaug[..., None], [4, 16])[..., 0].numpy()
  # unsup.numpy()[0].max()

  # def process(mat):
  #   maxx = np.max(mat)
  #   minx = np.min(mat)
  #   zfs = (mat - minx) / (maxx - minx)
  #   s_zfs = np.power(zfs, 1.1 + (5 * np.random.rand(*zfs.shape)))
  #   s_zfs /= np.sum(s_zfs)
  #   return s_zfs

  # for i in range(5):
  #   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
  #   ax1.imshow(process(mat=zfscore[i]))
  #   ax2.imshow(process(mat=zf_augscore[i]))
  #   ax3.imshow(unsup[i].numpy())
  #   ax4.imshow(unsupaug[i].numpy())
  #   plt.show()


if __name__ == "__main__":
  # train_local_disc_model()
  # save_some_data_for_plot()
  # plot_infomatch_train_emmbeding()
  # plot_mixmatch_train_emmbeding()
  # plot_uda_train_emmbeding()
  # plot_fake_ball()
  plot_fuck_loacl_matul_info_v1()
