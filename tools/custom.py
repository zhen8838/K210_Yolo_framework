import tensorflow as tf
from tensorflow.python.ops.math_ops import reduce_mean, reduce_sum,\
    sigmoid, sqrt, square, logical_and, cast, logical_not, div_no_nan, add
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper, Sum
keras = tf.keras
Callback = tf.keras.callbacks.Callback
K = tf.keras.backend
kls = tf.keras.losses
import signal
from tools.base import NOTE, colored, ERROR, INFO
from toolz import reduce
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import os


class DummyMetric(MeanMetricWrapper):

  def __init__(self, var: ResourceVariable, name: str, dtype=tf.float32):
    """ Dummy_Metric from MeanMetricWrapper

        Parameters
        ----------
        var : ResourceVariable

            a variable from loss
            NOTE only support shape : ()

        name : str

            dummy metric name

        dtype : [type], optional

            by default None
        """
    super().__init__(lambda y_true, y_pred, v: v, name=name, dtype=dtype, v=var)


class DummyOnceMetric(Metric):

  def __init__(self, var: ResourceVariable, name=None, dtype=None, **kwargs):
    super().__init__(name=name, dtype=dtype, **kwargs)
    self.var = var

  def update_state(self, *args, **kwargs):
    pass

  def result(self):
    return self.var.read_value()

  def reset_states(self):
    self.var.assign(0.)


class PFLDMetric(Metric):

  def __init__(self,
               calc_fr: bool,
               landmark_num: int,
               batch_size: int,
               name=None,
               dtype=None):
    """ PFLD metric ， calculate landmark error and failure rate

        Parameters
        ----------
        Metric : [type]

        calc_fr : bool
            wether calculate failure rate， NOTE if `False` just return failure rate
        landmark_num : int

        batch_size : int

        name :  optional
            by default None
        dtype : optional
            by default None
        """
    super().__init__(name=name, dtype=dtype)
    self.calc_fr = calc_fr
    if self.calc_fr == False:
      self.landmark_num = landmark_num
      self.batch_size = batch_size
      # NOTE if calculate landmark error , this variable will be use,
      # When calculate failure rate , just return failure rate .
      self.landmark_error = self.add_weight(
          'LE', initializer=tf.zeros_initializer())  # type: ResourceVariable

      self.failure_num = self.add_weight(
          'FR', initializer=tf.zeros_initializer())  # type: ResourceVariable

      self.total = self.add_weight(
          'total', initializer=tf.zeros_initializer())  # type: ResourceVariable

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.calc_fr == False:
      true_landmarks = y_true[:, :self.landmark_num * 2]
      pred_landmarks = y_pred[:, :self.landmark_num * 2]

      pred_landmarks = sigmoid(pred_landmarks)

      # calc landmark error
      error_all_points = reduce_sum(
          sqrt(
              reduce_sum(
                  square(
                      reshape(pred_landmarks, (self.batch_size, self.landmark_num,
                                               2)) -
                      reshape(true_landmarks, (self.batch_size, self.landmark_num,
                                               2))), [2])), 1)

      # use interocular distance calc landmark error
      if self.landmark_num == 98:
        left_eye = 60
        right_eye = 72
      else:
        left_eye = 36
        right_eye = 45
      interocular_distance = sqrt(
          reduce_sum(
              square((true_landmarks[:, left_eye * 2:(left_eye + 1) * 2] -
                      true_landmarks[:, right_eye * 2:(right_eye + 1) * 2])),
              1))

      error_norm = error_all_points / (interocular_distance * self.landmark_num)

      self.landmark_error.assign_add(reduce_sum(error_norm))

      # error norm > 0.1 ===> failure_number + 1
      self.failure_num.assign_add(reduce_sum(cast(error_norm > 0.1, self.dtype)))

      self.total.assign_add(self.batch_size)

  def result(self):
    if self.calc_fr == False:
      return div_no_nan(self.landmark_error, self.total)
    else:
      return div_no_nan(self.failure_num, self.total)


class SignalStopping(Callback):
  '''Stop training when an interrupt signal (or other) was received
            # Arguments
            sig: the signal to listen to. Defaults to signal.SIGINT.
            doubleSignalExits: Receiving the signal twice exits the python
                    process instead of waiting for this epoch to finish.
            patience: number of epochs with no improvement
                    after which training will be stopped.
            verbose: verbosity mode.
    '''

  def __init__(self, sig=signal.SIGINT, doubleSignalExits=True):
    super(SignalStopping, self).__init__()
    self.signal_received = False
    self.doubleSignalExits = doubleSignalExits

    def signal_handler(sig, frame):
      if self.doubleSignalExits:
        print(f'\n {NOTE} Received SIGINT twice to stop. Exiting..')
        self.doubleSignalExits = False
      else:
        self.signal_received = True
        print(
            f"\n {NOTE} Received SIGINT to stop now. {colored('Please Wait !','red')}"
        )

    signal.signal(sig, signal_handler)

  def on_batch_end(self, batch, logs=None):
    if self.signal_received:
      self.model.stop_training = True

  def on_epoch_end(self, epoch, logs=None):
    if self.signal_received:
      self.model.stop_training = True


class LRCallback(Callback):
  """ LRCallback for compat keras callback and custom training callback """

  def __init__(self, outside_optimizer: str = None):
    super().__init__()
    self.outside_optimizer: str = outside_optimizer

  def set_optimizer(self, optimizer):
    if isinstance(optimizer, tf.optimizers.Optimizer):
      self.optimizer = optimizer

  def set_lr(self, new_lr):
    if self.outside_optimizer:
      K.set_value(self.optimizer.lr, new_lr)
    else:
      K.set_value(self.model.optimizer.lr, new_lr)

  def get_lr(self):
    if self.outside_optimizer:
      lr = K.get_value(self.optimizer.lr)
    else:
      lr = K.get_value(self.model.optimizer.lr)
    return lr


class StepLR(LRCallback):

  def __init__(self, rates: list, steps: list, outside_optimizer: str = None):
    """ Step learning rate setup callback

        eg. steps = [100, 200, 300]
            rates = [0.5, 0.1, 0.8]

            in epoch  0  ~ 100 lr=0.5
            in epoch 100 ~ 200 lr=0.1

        Parameters
        ----------
        rates : list

        steps : list

        """
    super().__init__(outside_optimizer)
    assert len(rates) == len(
        steps), f'{ERROR} the len(rates) must equal len(steps)'
    assert steps[0] > 0, f'{ERROR} the steps[0] can\'t <= 0'
    self.rates = []
    steps.insert(0, 0)
    for i in range(len(rates)):
      self.rates += [rates[i]] * (steps[i + 1] - steps[i])

  def on_epoch_begin(self, epoch, logs=None):
    if epoch < len(self.rates):
      self.set_lr(self.rates[epoch])
    else:
      self.set_lr(self.rates[-1])

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = self.get_lr()


class CosineLR(LRCallback):

  def __init__(self,
               init_lr: float,
               decay_steps: int,
               lowest_lr: float,
               outside_optimizer: str = None):
    """ 
        Applies cosine to the learning rate.

        See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
        with Warm Restarts. https://arxiv.org/abs/1608.03983

        eg. CosineLR(0.001,30,0.0001)

            init_lr=0.001

            decay_steps=30

            lowest_lr=0.0001

        lr will be :
              0.001   | _       __       _    
                      |  \     /  \     /    
                      |   \   /    \   /    
            0.001*0.9 |    \_/      \_/    
              epochs  :    30        60
        """
    super().__init__(outside_optimizer)
    self.init_lr = init_lr
    self.decay_steps = decay_steps
    lowest_lr = 0.0001
    assert lowest_lr < init_lr, 'lowest_lr must smaller than init_lr'
    self.alpha = lowest_lr / self.init_lr

  def decayed_learning_rate(self, step: tf.Tensor):
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
    decayed = (1 - self.alpha) * cosine_decay + self.alpha
    return self.init_lr * decayed

  def on_epoch_begin(self, epoch, logs=None):
    self.set_lr(self.decayed_learning_rate(epoch))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = self.get_lr()


class ScheduleLR(LRCallback):
  """Configurable learning rate schedule."""

  def __init__(self,
               base_lr: float,
               use_warmup: bool,
               warmup_epochs: int,
               decay_rate: float,
               decay_epochs: int,
               outside_optimizer: str = None):
    """ Schedule lr

      When warmup is used, LR will increase linearly from 0 to the peak value of warmup epochs. 
      After that, LR is reduced once per decade according to the decade rate.

      eg. ScheduleLR(
            base_lr=0.1,
            use_warmup=True,
            warmup_epochs=5,
            decay_rate=0.5,
            decay_epochs=5)

      lr will be :
           0.1   |         _____
                 |       /      |
           0.5   |     /        |_____
           0.25  |   /                |_____
            0    | /
          epochs : 0       5     10    10

    Args:
        base_lr (float): base learning rate
        use_warmup (bool): if `True` will use warmup
        warmup_epochs (int): warmup epochs
        decay_rate (float): each decay rate
        decay_epochs (int): how many epochs decay once
    """
    super().__init__(outside_optimizer)
    self.base_lr = base_lr
    self.use_warmup = use_warmup
    self.warmup_epochs = warmup_epochs
    self.decay_rate = decay_rate
    self.decay_epochs = decay_epochs
    if isinstance(self.decay_epochs, (list, tuple)):
      lr_values = [
          self.base_lr * (self.decay_rate**k)
          for k in range(len(self.decay_epochs) + 1)
      ]
      self.lr_schedule_no_warmup = (
          keras.optimizers.schedules.PiecewiseConstantDecay(
              self.decay_epochs, lr_values))
    else:
      self.lr_schedule_no_warmup = (
          keras.optimizers.schedules.ExponentialDecay(
              self.base_lr, self.decay_epochs, self.decay_rate, staircase=True))

  def on_epoch_begin(self, epoch, logs=None):
    if self.use_warmup:
      lr = tf.cond(
          epoch < self.warmup_epochs, lambda: tf.cast(epoch, tf.float32) / self.
          warmup_epochs * self.base_lr, lambda: self.lr_schedule_no_warmup(epoch))
    else:
      lr = self.lr_schedule_no_warmup(epoch)
    self.set_lr(lr)


class VariableCheckpoint(Callback):

  def __init__(self,
               log_dir: str,
               variable_dict: dict,
               monitor='val_loss',
               mode='auto'):
    super().__init__()
    self.log_dir = str(log_dir)
    self.auto_save_dirs = os.path.join(self.log_dir, 'auto_save')
    self.final_save_dirs = os.path.join(self.log_dir, 'final_save')
    self.saver = tf.train.Checkpoint(**variable_dict)
    self.auto_save_manager = tf.train.CheckpointManager(
        self.saver, directory=self.auto_save_dirs, max_to_keep=20)
    self.final_save_manager = tf.train.CheckpointManager(
        self.saver, directory=self.final_save_dirs, max_to_keep=None)
    self.monitor = monitor
    self.save_best_only = True

    if mode not in ['auto', 'min', 'max', 'all']:
      raise ValueError(
          'ModelCheckpoint mode %s is unknown, '
          'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    elif mode == 'all':
      self.monitor_op = lambda a, b: True
      self.best = 0
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def load_checkpoint(self, pre_checkpoint: str):
    if pre_checkpoint:
      self.saver.restore(tf.train.latest_checkpoint(pre_checkpoint))
      print(INFO, f' Load Checkpoint From {pre_checkpoint}')
      return
    else:
      latest_checkpoint = (
          tf.train.latest_checkpoint(self.auto_save_dirs) or
          tf.train.latest_checkpoint(self.final_save_dirs))
      if latest_checkpoint:
        # checkpoint.restore must be within a strategy.scope() so that optimizer
        # slot variables are mirrored.
        self.saver.restore(latest_checkpoint)
        print(INFO, f' Load Checkpoint From {latest_checkpoint}')
        return
    print(INFO, f' No pre-Checkpoint Load')

  def _save_variable(self, logs: dict):
    current = logs.get(self.monitor)
    if current is None:
      print('Can save best model only with %s available, '
            'skipping.', self.monitor)
    else:
      if self.monitor_op(current, self.best):
        self.best = current
        self.auto_save_manager.save()

  def on_epoch_end(self, epoch, logs=None):
    self._save_variable(logs)

  def on_train_end(self, logs=None):
    self.final_save_manager.save()


def focal_sigmoid_cross_entropy_with_logits(labels: tf.Tensor,
                                            logits: tf.Tensor,
                                            gamma: float = 2.0,
                                            alpha: float = 0.25):
  pred_sigmoid = tf.nn.sigmoid(logits)
  pt = (1 - pred_sigmoid) * labels + pred_sigmoid * (1 - labels)
  focal_weight = (alpha * labels + (1 - alpha) * (1 - labels)) * tf.math.pow(pt, gamma)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits) * focal_weight
  return loss


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


def cosdistance(embedd_a: tf.Tensor, embedd_b: tf.Tensor,
                is_norm: bool = True) -> tf.Tensor:
  """ cos distance

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

  return tf.squeeze(tf.keras.backend.batch_dot(embedd_a, embedd_b, -1))


distance_register = {'l2': l2distance, 'cos': cosdistance}


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


class SparseSoftmaxLoss(kls.Loss):

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
    return tf.keras.backend.sparse_categorical_crossentropy(
        y_true, self.scale * y_pred, True)


class SparseAmsoftmaxLoss(kls.Loss):

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


class SparseAsoftmaxLoss(kls.Loss):

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


class CircleLoss(kls.Loss):

  def __init__(self,
               gamma: int = 64,
               margin: float = 0.25,
               batch_size: int = None,
               reduction='auto',
               name=None):
    super().__init__(reduction=reduction, name=name)
    self.gamma = gamma
    self.margin = margin
    self.O_p = 1 + self.margin
    self.O_n = -self.margin
    self.Delta_p = 1 - self.margin
    self.Delta_n = self.margin
    if batch_size:
      self.batch_size = batch_size
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity

    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]

    Returns:
        tf.Tensor: loss
    """
    alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
    alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
    # yapf: disable
    y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
              (1 - y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
    # yapf: enable
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class SparseCircleLoss(CircleLoss):

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity

    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]

    Returns:
        tf.Tensor: loss
    """

    # idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    # sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)

    # alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(sp))
    # alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
    # alpha_n_for_p = tf.expand_dims(tf.gather_nd(alpha_n, idxs), 1)

    # r_sp_m = alpha_p * (sp - self.Delta_p)
    # r_sn_m = alpha_n * (y_pred - self.Delta_n)
    # _Z = tf.concat([r_sn_m, r_sp_m], 1)
    # _Z = _Z * self.gamma
    # # sum all similarity
    # logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    # # remove sn_p from all sum similarity
    # TODO This line will be numerical overflow, Need a more numerically safe method
    # logZ = logZ + tf.math.log(1 - tf.math.exp(
    #     (alpha_n_for_p * (sp - self.Delta_n)) * self.gamma - logZ))

    # return -r_sp_m * self.gamma + logZ
    """ method 2 """
    # idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    # sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)
    # mask = tf.logical_not(
    #     tf.scatter_nd(idxs, tf.ones(tf.shape(idxs)[0], tf.bool),
    #                   tf.shape(y_pred)))

    # sn = tf.reshape(tf.boolean_mask(y_pred, mask), (self.batch_size, -1))
    """ method 3 """
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)
    last_dim = y_pred[:, -1]
    y_pred = tf.tensor_scatter_nd_update(y_pred, idxs, last_dim)
    sn = y_pred[:, :-1]

    alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(sp))
    alpha_n = tf.nn.relu(tf.stop_gradient(sn) - self.O_n)

    r_sp_m = alpha_p * (sp - self.Delta_p)
    r_sn_m = alpha_n * (sn - self.Delta_n)
    _Z = tf.concat([r_sn_m, r_sp_m], 1)
    _Z = _Z * self.gamma
    # sum all similarity
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    # remove sn_p from all sum similarity
    return -r_sp_m * self.gamma + logZ


class PairCircleLoss(CircleLoss):

  def call(self, sp: tf.Tensor, sn: tf.Tensor) -> tf.Tensor:
    """ use within-class similarity and between-class similarity for loss

    Args:
        sp (tf.Tensor): within-class similarity  shape [batch, K]
        sn (tf.Tensor): between-class similarity shape [batch, L]

    Returns:
        tf.Tensor: loss
    """
    ap = tf.nn.relu(-tf.stop_gradient(sp) + 1 + self.margin)
    an = tf.nn.relu(tf.stop_gradient(sn) + self.margin)

    logit_p = -ap * (sp - self.Delta_p) * self.gamma
    logit_n = an * (sn - self.Delta_n) * self.gamma

    return tf.math.softplus(
        tf.math.reduce_logsumexp(logit_n, axis=-1, keepdims=True) +
        tf.math.reduce_logsumexp(logit_p, axis=-1, keepdims=True))
