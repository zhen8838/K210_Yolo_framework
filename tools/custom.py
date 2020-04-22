import tensorflow as tf
from tensorflow.python.ops.math_ops import reduce_mean, reduce_sum,\
    sigmoid, sqrt, square, logical_and, cast, logical_not, div_no_nan, add
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper, Sum
keras = tf.keras
Callback = tf.keras.callbacks.Callback
K = tf.keras.backend
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
      interocular_distance = sqrt(
          reduce_sum(
              square((true_landmarks[:, 120:122] - true_landmarks[:, 144:146])),
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
  pt = (1-pred_sigmoid) * labels + pred_sigmoid * (1-labels)
  focal_weight = (alpha*labels + (1-alpha) * (1-labels)) * tf.math.pow(pt, gamma)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits) * focal_weight
  return loss
