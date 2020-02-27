import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
kls = tf.keras.losses
kc = tf.keras.constraints
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
from tools.base import INFO, ERROR, NOTE, BaseHelper
from tools.training_engine import BaseTrainingLoop
import imgaug.augmenters as iaa
from typing import List, Dict, Callable, Iterable, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn.metrics import auc
from tqdm import tqdm


class FcaeRecHelper(BaseHelper):

  def __init__(self,
               image_ann: str,
               in_hw: tuple,
               embedding_size: int,
               val_dataset: str = 'lfw,cfp_fp,agedb_30',
               use_softmax: bool = True):
    """ face recogintion helper

        Parameters
        ----------
        BaseHelper : [type]

        image_ann : str

            image annotation file path

        in_hw : tuple

            in image height width

        embedding_size : int

            embedding size

        use_softmax : bool, optional

            Note Use Softmax loss set to `True`, Use Triplet loss set to `False`, by default True

        """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    if image_ann == None:
      self.train_list: np.ndarray = None
      self.val_list: np.ndarray = None
      self.test_list: np.ndarray = None
    else:
      img_ann_list: dict = np.load(image_ann, allow_pickle=True)[()]
      self.train_list: List[List[Tuple[str, int]]] = img_ann_list['train_data']
      self.train_total_data: int = img_ann_list['train_num']
      self.val_list: List[str] = []
      self.test_list: List[str] = []
      self.val_total_data: int = 0
      self.test_total_data: int = 0
      for name in val_dataset.split(','):
        self.val_list.append(img_ann_list[f'{name}_val_data'])
        self.test_list.append(img_ann_list[f'{name}_val_data'])

        self.val_total_data += img_ann_list[f'{name}_val_num']
        self.test_total_data += img_ann_list[f'{name}_val_num']
      self.dataset_root: tf.Tensor = tf.constant(
          '/'.join(str.split(self.val_list[0], '/')[:-1]), tf.string)
    del img_ann_list
    self.in_hw = np.array(in_hw)
    self.embedding_size = embedding_size
    self.use_softmax = use_softmax

  def augment_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    img = tf.image.random_flip_left_right(img)

    l = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 4, tf.int32)[0]
    img = tf.cond(l[0] == 1, lambda: img, lambda: tf.image.random_hue(img, 0.15))
    img = tf.cond(
        l[1] == 1, lambda: img, lambda: tf.image.random_saturation(img, 0.6, 1.6))
    img = tf.cond(
        l[2] == 1, lambda: img, lambda: tf.image.random_brightness(img, 0.1))
    img = tf.cond(
        l[3] == 1, lambda: img, lambda: tf.image.random_contrast(img, 0.7, 1.3))
    return img, ann

  def build_train_datapipe(self, image_ann_list: List[str], batch_size: int,
                           is_augment: bool, is_normlize: bool,
                           is_training: bool) -> tf.data.Dataset:
    print(INFO, 'data augment is ', str(is_augment))

    img_shape = list(self.in_hw) + [3]
    fnames: tf.RaggedTensor = tf.ragged.constant(image_ann_list['fnames'],
                                                 tf.string)
    labels: tf.RaggedTensor = tf.ragged.constant(image_ann_list['lables'],
                                                 tf.int32)

    nclass = len(image_ann_list['lables'])
    del image_ann_list

    if self.use_softmax:

      def parser(fname: tf.Tensor, label: tf.Tensor):
        # load image
        fname = tf.add(tf.add(self.dataset_root, '/'), fname)
        contents = tf.io.read_file(fname)
        raw_img: tf.Tensor = tf.image.decode_jpeg(contents, 3)
        if is_augment:
          raw_img, _ = self.augment_img(raw_img, None)
        # normlize image
        if is_normlize is True:
          img = self.normlize_img(raw_img)  # type:tf.Tensor
        else:
          img = tf.cast(raw_img, tf.float32)
        img.set_shape(img_shape)
        label.set_shape([])
        # Note y_true shape will be [batch]
        return (img), (label)

      ds_trans = lambda fname, label: tf.data.Dataset.from_tensor_slices((
          fname, label)).shuffle(100).repeat()
      ds = tf.data.Dataset.from_tensor_slices((fnames, labels)).interleave(
          ds_trans, cycle_length=nclass).shuffle(batch_size * 400).repeat().map(
              parser, -1).batch(batch_size, True).prefetch(-1)

    else:

      def parser(fname: tf.Tensor, label: tf.Tensor):
        fname = tf.reshape(fname, (-1,))
        label = tf.reshape(label, (-1,))[:3]
        fname = tf.add(tf.add(self.dataset_root, '/'), fname)
        raw_imgs0: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[0]), 3)
        raw_imgs1: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[1]), 3)
        raw_imgs2: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[2]), 3)
        # imgs do same augment ~
        if is_augment:
          raw_imgs0, _ = self.augment_img(raw_imgs0, None)
          raw_imgs1, _ = self.augment_img(raw_imgs1, None)
          raw_imgs2, _ = self.augment_img(raw_imgs2, None)
        # normlize image
        if is_normlize:
          imgs0: tf.Tensor = self.normlize_img(raw_imgs0)
          imgs1: tf.Tensor = self.normlize_img(raw_imgs1)
          imgs2: tf.Tensor = self.normlize_img(raw_imgs2)
        else:
          imgs0 = tf.cast(raw_imgs0, tf.float32)
          imgs1 = tf.cast(raw_imgs1, tf.float32)
          imgs2 = tf.cast(raw_imgs2, tf.float32)
        imgs0.set_shape(img_shape)
        imgs1.set_shape(img_shape)
        imgs2.set_shape(img_shape)
        label.set_shape([3])
        # Note y_true shape will be [batch,3]
        return (raw_imgs0, raw_imgs1, raw_imgs2), (label)

      ds_trans = lambda fname, label: tf.data.Dataset.from_tensor_slices((
          fname, label)).shuffle(100).repeat()

      ds = tf.data.Dataset.from_tensor_slices((fnames, labels)).interleave(
          ds_trans, cycle_length=nclass,
          block_length=2).batch(2, True).shuffle(batch_size * 400).batch(
              2, True).map(parser, -1).batch(batch_size, True).prefetch(-1)

    return ds

  def build_val_datapipe(self, image_ann_list: str, batch_size: int,
                         is_augment: bool, is_normlize: bool,
                         is_training: bool) -> tf.data.Dataset:

    img_shape = list(self.in_hw) + [3]

    def parser(stream: bytes):
      example = tf.io.parse_single_example(
          stream, {
              'img_a': tf.io.FixedLenFeature([], tf.string),
              'img_b': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.int64),
          })  # type:dict
      raw_img_a: tf.Tensor = tf.image.decode_image(example['img_a'], channels=3)
      raw_img_b: tf.Tensor = tf.image.decode_image(example['img_b'], channels=3)
      label: tf.Tensor = tf.cast(example['label'], tf.bool)

      if is_normlize:
        img_a: tf.Tensor = self.normlize_img(raw_img_a)
        img_b: tf.Tensor = self.normlize_img(raw_img_b)
      else:
        img_a = tf.cast(raw_img_a, tf.float32)
        img_b = tf.cast(raw_img_b, tf.float32)

      img_a.set_shape(img_shape)
      img_b.set_shape(img_shape)
      label.set_shape(())
      # Note y_true shape will be [batch]
      return (img_a, img_b), (label)

    if is_training:
      ds_trans = lambda x: tf.data.TFRecordDataset(x).repeat()
      ds = (
          tf.data.Dataset.from_tensor_slices(image_ann_list).interleave(
              ds_trans).map(parser, -1).batch(batch_size, True).prefetch(-1))
    else:
      ds_trans = lambda x: tf.data.TFRecordDataset(x)
      ds = (
          tf.data.Dataset.from_tensor_slices(image_ann_list).interleave(
              ds_trans).map(parser, -1).batch(batch_size, True).prefetch(-1))

    return ds

  def set_dataset(self,
                  batch_size: int,
                  is_augment: bool = True,
                  is_normlize: bool = True,
                  is_training: bool = True):
    self.batch_size = batch_size
    if is_training:
      self.train_dataset = self.build_train_datapipe(self.train_list, batch_size,
                                                     is_augment, is_normlize,
                                                     is_training)
      self.val_dataset = self.build_val_datapipe(self.val_list, batch_size, False,
                                                 is_normlize, is_training)

      self.train_epoch_step = self.train_total_data // self.batch_size
      self.val_epoch_step = self.val_total_data // self.batch_size
    else:
      self.test_dataset = self.build_val_datapipe(self.test_list, batch_size,
                                                  False, is_normlize, is_training)
      self.test_epoch_step = self.test_total_data // self.batch_size


def l2distance(embedd_a: tf.Tensor, embedd_b: tf.Tensor,
               is_norm: bool = True) -> tf.Tensor:
  """ l2 distance

    Parameters
    ----------
    embedd_a : tf.Tensor

        shape : [batch,embedding size]

    embedd_b : tf.Tensor

        shape : [batch,embedding size]

    is_norm : bool

        is norm == True will normilze(embedd_a)

    Returns
    -------
    [tf.Tensor]

        distance shape : [batch]
    """
  if is_norm:
    embedd_a = tf.math.l2_normalize(embedd_a, -1)
    embedd_b = tf.math.l2_normalize(embedd_b, -1)
  return tf.reduce_sum(tf.square(tf.math.subtract(embedd_a, embedd_b)), -1)


distance_register = {'l2': l2distance}


class TripletLoss(kls.Loss):

  def __init__(self,
               target_distance: float,
               distance_fn: str = 'l2',
               reduction='auto',
               name=None):
    """ Triplet Loss:

            When using l2 diatance , target_distance ∈ [0,4]

        Parameters
        ----------
        target_distance : float

            target distance threshold

        distance_fn : str, optional

            distance_fn name , by default 'l2'

        """
    super().__init__(reduction=reduction, name=name)
    self.target_distance = target_distance
    self.distance_str = distance_fn
    self.distance_fn: l2distance = distance_register[distance_fn]
    self.triplet_acc: tf.Variable = tf.compat.v1.get_variable(
        'triplet_acc',
        shape=(),
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    a, p, n = tf.split(y_pred, 3, axis=-1)
    ap = self.distance_fn(a, p, is_norm=True)  # [batch]
    an = self.distance_fn(a, n, is_norm=True)  # [batch]
    total_loss = tf.reduce_mean(tf.nn.relu(ap - an +
                                           self.target_distance))  # [batch]
    self.triplet_acc.assign(
        tf.reduce_mean(
            tf.cast(
                tf.equal(ap + self.target_distance < an,
                         tf.ones_like(ap, tf.bool)), tf.float32)))
    return total_loss


class Sparse_SoftmaxLoss(kls.Loss):

  def __init__(self, scale=30, reduction='auto', name=None):
    """ sparse softmax loss with scale

        Parameters
        ----------
        scale : int, optional

            loss scale, by default 30

        reduction : [type], optional

            by default 'auto'

        name : str, optional

            by default None

        """
    super().__init__(reduction=reduction, name=name)
    self.scale = scale

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return k.backend.sparse_categorical_crossentropy(y_true, self.scale * y_pred,
                                                     True)


class Sparse_AmsoftmaxLoss(kls.Loss):

  def __init__(self,
               batch_size: int,
               scale: int = 30,
               margin: int = 0.35,
               reduction='auto',
               name=None):
    """ sparse addivate margin softmax

        Parameters
        ----------

        scale : int, optional

            by default 30

        margin : int, optional

            by default 0.35

        """
    super().__init__(reduction=reduction, name=name)
    self.scale = scale
    self.margin = margin
    self.batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32),
                                     1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ loss calc

        Parameters
        ----------
        y_true : tf.Tensor

            shape = [batch,1] type = tf.int32

        y_pred : tf.Tensor

            shape = [batch,class num] type = tf.float32

        Returns
        -------

        tf.Tensor

            loss
        """
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)
    y_true_pred = tf.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - self.margin
    _Z = tf.concat([y_pred, y_true_pred_margin], 1)
    _Z = _Z * self.scale
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ))
    return -y_true_pred_margin * self.scale + logZ


class Sparse_AsoftmaxLoss(kls.Loss):

  def __init__(self,
               batch_size: int,
               scale: int = 30,
               margin: int = 0.35,
               reduction='auto',
               name=None):
    """ sparse addivate softmax

        Parameters
        ----------

        scale : int, optional

            by default 30

        margin : int, optional

            by default 0.35

        """
    super().__init__(reduction=reduction, name=name)
    self.scale = scale
    self.batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32), 1)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)  # find the y_pred
    y_true_pred = tf.expand_dims(y_true_pred, 1)
    y_true_pred_margin = 1 - 8 * tf.square(y_true_pred) + 8 * tf.square(
        tf.square(y_true_pred))
    # min(y_true_pred, y_true_pred_margin)
    y_true_pred_margin = y_true_pred_margin - tf.nn.relu(y_true_pred_margin -
                                                         y_true_pred)
    _Z = tf.concat([y_pred, y_true_pred_margin], 1)
    _Z = _Z * self.scale  # use scale expand value range
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ)
                             )  # Z - exp(scale * y_true_pred)
    return -y_true_pred_margin * self.scale + logZ


class FacerecValidation(k.callbacks.Callback):

  def __init__(self, validation_model: k.Model, validation_data: tf.data.Dataset,
               validation_steps: int, trian_steps: int, accuracy_var: tf.Variable,
               batch_size: int, distance_fn: str, threshold: float):
    self.val_model = validation_model
    self.val_iter: Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]] = iter(
        validation_data)
    self.val_step = int(validation_steps)
    self.trian_step = int(trian_steps)
    self.acc_var = accuracy_var
    self.distance_fn: l2distance = distance_register[distance_fn]
    self.threshold = threshold
    self.batch_size = batch_size

  def on_train_batch_begin(self, batch, logs=None):
    if batch == self.trian_step:

      def fn(i: tf.Tensor):
        x, actual_issame = next(self.val_iter)  # actual_issame:tf.Bool
        y_pred: Tuple[tf.Tensor] = self.val_model(x, training=False)
        dist: tf.Tensor = self.distance_fn(*y_pred, is_norm=True)  # [batch]

        pred_issame = tf.less(dist, self.threshold)

        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(pred_issame, actual_issame), tf.float32))
        fp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(pred_issame, tf.logical_not(actual_issame)),
                tf.float32))
        tn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_not(pred_issame), tf.logical_not(actual_issame)),
                tf.float32))
        fn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(tf.logical_not(pred_issame), actual_issame),
                tf.float32))

        tpr = tf.math.divide_no_nan(tp, tp + fn)
        fpr = tf.math.divide_no_nan(fp, fp + tn)
        return (tp+tn) / tf.cast(tf.shape(dist)[0], tf.float32)

      k.backend.set_value(
          self.acc_var,
          tf.reduce_mean(tf.map_fn(fn, tf.range(self.val_step), tf.float32)))


class FaceTripletTrainingLoop(BaseTrainingLoop):

  def __init__(self,
               train_model,
               val_model,
               target_distance: float,
               distance_fn: str = 'l2'):
    super().__init__(train_model, val_model)
    self.target_distance = target_distance
    self.distance_fn: l2distance = distance_register[distance_fn]

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss': k.metrics.Mean('train_loss', dtype=tf.float32),
            'acc': k.metrics.Mean('train_acc', dtype=tf.float32)
        },
        'val': {
            'acc': k.metrics.Accuracy('val_acc', dtype=tf.float32)
        }
    }
    return d

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        y_pred = self.train_model(images, training=True)
        a, p, n = tf.split(y_pred, 3, axis=-1)
        ap = self.distance_fn(a, p, is_norm=True)  # [batch]
        an = self.distance_fn(a, n, is_norm=True)  # [batch]
        loss_tp = tf.reduce_mean(tf.nn.relu(ap - an + self.target_distance))
        loss_wd = tf.reduce_sum(self.train_model.losses)
        loss = loss_tp + loss_wd

      acc = tf.reduce_mean(
          tf.cast(
              tf.equal(ap + self.target_distance < an, tf.ones_like(ap, tf.bool)),
              tf.float32))

      grads = tape.gradient(loss, self.train_model.trainable_variables)
      self.optimizer.apply_gradients(
          zip(grads, self.train_model.trainable_variables))

      metrics.loss.update_state(loss)
      metrics.acc.update_state(acc)

    for _ in tf.range(num_steps_to_run):
      step_fn(next(iterator))

    @tf.function
    def val_step(self, dataset, metrics):

      def step_fn(inputs):
        """Per-Replica training step function."""
        images, actual_issame = inputs
        y_pred: Tuple[tf.Tensor] = self.val_model(images, training=False)
        dist: tf.Tensor = self.distance_fn(*y_pred, is_norm=True)  # [batch]
        pred_issame = tf.less(dist, self.threshold)
        metrics.acc.update_state(actual_issame, pred_issame)

      for inputs in dataset:
        step_fn(inputs)


def calculate_accuracy(threshold: float, dist: np.ndarray, issame: np.ndarray):
  predict_issame = np.less(dist, threshold)
  tp = np.sum(np.logical_and(predict_issame, issame))
  fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
  tn = np.sum(
      np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
  fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))

  tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
  fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
  acc = float(tp + tn) / dist.size
  return tpr, fpr, acc


class LFold:

  def __init__(self, n_splits=2, shuffle=False):
    self.n_splits = n_splits
    if self.n_splits > 1:
      self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

  def split(self, indices):
    if self.n_splits > 1:
      return self.k_fold.split(indices)
    else:
      return [(indices, indices)]


def normalize(arr):
  return arr / np.linalg.norm(arr, 2, -1, keepdims=True)


def calculate_roc(thresholds, embedds_a, embedds_b, issame, nrof_folds=10, pca=0):
  nrof_pairs = len(issame)
  nrof_thresholds = len(thresholds)
  k_fold = LFold(n_splits=nrof_folds, shuffle=False)

  tprs = np.zeros((nrof_folds, nrof_thresholds))
  fprs = np.zeros((nrof_folds, nrof_thresholds))
  accuracy = np.zeros((nrof_folds))
  indices = np.arange(nrof_pairs)

  if pca == 0:
    embedds_a = normalize(embedds_a)
    embedds_b = normalize(embedds_b)
    diff = np.subtract(embedds_a, embedds_b)
    dist = np.sum(np.square(diff), 1)

  for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
    if pca > 0:
      print(NOTE, 'doing pca on', fold_idx)
      embed1_train = embedds_a[train_set]
      embed2_train = embedds_b[train_set]
      _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
      pca_model = PCA(n_components=pca)
      pca_model.fit(_embed_train)
      embed1 = pca_model.transform(embedds_a)
      embed2 = pca_model.transform(embedds_b)
      embed1 = normalize(embed1)
      embed2 = normalize(embed2)
      diff = np.subtract(embed1, embed2)
      dist = np.sum(np.square(diff), 1)

    # Find the best threshold for the fold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
      _, _, acc_train[threshold_idx] = calculate_accuracy(
          threshold, dist[train_set], issame[train_set])
    best_threshold_index = np.argmax(acc_train)
    for threshold_idx, threshold in enumerate(thresholds):
      tprs[fold_idx,
           threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
               threshold, dist[test_set], issame[test_set])
    _, _, accuracy[fold_idx] = calculate_accuracy(
        thresholds[best_threshold_index], dist[test_set], issame[test_set])

  tpr = np.mean(tprs, 0)
  fpr = np.mean(fprs, 0)
  return tpr, fpr, accuracy, thresholds[best_threshold_index]


def calculate_val_far(threshold: float, dist: np.ndarray, issame: np.ndarray):
  predict_issame = np.less(dist, threshold)
  true_accept = np.sum(np.logical_and(predict_issame, issame))
  false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
  n_same = np.sum(issame)
  n_diff = np.sum(np.logical_not(issame))
  val = float(true_accept) / float(n_same)
  far = float(false_accept) / float(n_diff)
  return val, far


def calculate_val(thresholds,
                  embedds_a,
                  embedds_b,
                  issame,
                  far_target,
                  nrof_folds=10):
  nrof_pairs = len(issame)
  nrof_thresholds = len(thresholds)
  k_fold = LFold(n_splits=nrof_folds, shuffle=False)

  val = np.zeros(nrof_folds)
  far = np.zeros(nrof_folds)

  diff = np.subtract(embedds_a, embedds_b)
  dist = np.sum(np.square(diff), 1)
  indices = np.arange(nrof_pairs)

  for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
      _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set],
                                                      issame[train_set])
    if np.max(far_train) >= far_target:
      f = interpolate.interp1d(far_train, thresholds, kind='slinear')
      threshold = f(far_target)
    else:
      threshold = 0.0

    val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set],
                                                     issame[test_set])

  val_mean = np.mean(val)
  far_mean = np.mean(far)
  val_std = np.std(val)
  return val_mean, val_std, far_mean


def calculate_evaluate_metric(embedds_a, embedds_b, issame, nrof_folds=10, pca=0):
  # Calculate evaluation metrics
  thresholds = np.arange(0, 4, 0.01)
  tpr, fpr, accuracy, best_threshold = calculate_roc(
      thresholds, embedds_a, embedds_b, issame, nrof_folds=nrof_folds, pca=pca)
  thresholds = np.arange(0, 4, 0.001)
  val, val_std, far = calculate_val(
      thresholds, embedds_a, embedds_b, issame, 1e-3, nrof_folds=nrof_folds)
  return tpr, fpr, accuracy, best_threshold, val, val_std, far


def facerec_eval(infer_model: k.Model,
                 h: FcaeRecHelper,
                 nrof_folds: int,
                 pca: int,
                 batch_size: int,
                 is_plot: bool = True):
  h.set_dataset(batch_size, is_training=False)
  embedds_a = []
  embedds_b = []
  issame = []
  for (img_a, img_b), y_true in tqdm(
      h.test_dataset, total=int(h.test_epoch_step)):
    embedds_a.append(infer_model.predict(img_a))
    embedds_b.append(infer_model.predict(img_b))
    issame.append(y_true.numpy())

  embedds_a = np.vstack(embedds_a)
  embedds_b = np.vstack(embedds_b)
  issame = np.hstack(issame)

  (tpr, fpr, accuracy, best_threshold, val, val_std,
   far) = calculate_evaluate_metric(embedds_a, embedds_b, issame, nrof_folds, pca)

  auc_area = auc(fpr, tpr)
  print(NOTE, f'Best thresh: {best_threshold:.2f}')
  print(NOTE, f'Accuracy: {np.mean(accuracy):.2f} +- {np.std(accuracy):.2f}')

  if is_plot:
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (area = {auc_area:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
