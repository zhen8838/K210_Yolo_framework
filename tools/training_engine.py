import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from typing import Iterator, Mapping
from tqdm import tqdm
import abc
from tensorflow.python.keras.callbacks import CallbackList
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.backend import get_graph
from tools.base import INFO
import os
import time
import numpy as np
import sys


class EasyDict(object):

  def __init__(self, dicts):
    """ convert dict to object like

        Parameters
        ----------
        object : [type]

        dicts : dict
            dict
        """
    if dicts != None:
      for name, value in dicts.items():
        if isinstance(value, dict):
          setattr(self, name, EasyDict(value))
        else:
          setattr(self, name, value)

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def dicts(self):
    return self.__dict__


class EmaHelper(object):

  def __init__(self, orig_model: k.Model, decay: float):
    """ Helper class for exponential moving average.
    
    eg. 
    ```python
    self.ema = EmaHelper(self.val_model, self.hparams.ema.decay)
    self.ema.update()
    .
    .
    self.ema.model(test_data,training=False)
    ```
    
    Args:
        
        orig model (k.Model): usually be validation model NOTE this model variables must be auto update or update in training loop
        
        decay (float): ema decay rate
    """
    self.decay = decay
    self.orig_model = orig_model
    self.model = k.models.clone_model(orig_model)
    self.initial_ema_vars(self.model.variables, self.orig_model.variables)

  def update(self):
    self.update_ema_vars(self.model.variables, self.orig_model.variables,
                         self.decay)

  @staticmethod
  def initial_ema_vars(ema_variables: dict, initial_values: dict):
    """ Assign EMA variables from initial values.
    
    Args:
        ema_variables (dict): ema model variables
        initial_values (dict): training model variables
    """

    def _assign_one_var_fn(ema_var, value):
      ema_var.assign(value)

    def _assign_all_in_cross_replica_context_fn(strategy, ema_vars, values):
      for ema_var, value in zip(ema_vars, values):
        value = strategy.extended.reduce_to(tf.distribute.ReduceOp.MEAN, value,
                                            ema_var)
        if ema_var.trainable:
          strategy.extended.update(ema_var, _assign_one_var_fn, args=(value,))
        else:
          _assign_one_var_fn(ema_var, value)

    replica_context = tf.distribute.get_replica_context()
    if replica_context:
      replica_context.merge_call(
          _assign_all_in_cross_replica_context_fn,
          args=(ema_variables, initial_values))
    else:
      if tf.distribute.in_cross_replica_context():
        _assign_all_in_cross_replica_context_fn(tf.distribute.get_strategy(),
                                                ema_variables, initial_values)
      else:
        for ema_var, value in zip(ema_variables, initial_values):
          _assign_one_var_fn(ema_var, value)

  @staticmethod
  def update_ema_vars(ema_variables: dict, new_values: dict, ema_decay: float):
    """ Updates EMA variables. 
      
      Update rule is following:
        ema_var := ema_var * ema_decay + var * (1 - ema_decay)
      which is equivalent to:
        ema_var -= (1 - ema_decay) * (ema_var - var)
        
    Args:
        ema_variables (dict): ema model variables
        new_values (dict):  training model variables
        ema_decay (float): ema decay rate
    """

    one_minus_decay = 1.0 - ema_decay

    def _update_one_var_fn(ema_var, value):
      ema_var.assign_sub((ema_var-value) * one_minus_decay)

    def _update_all_in_cross_replica_context_fn(strategy, ema_vars, values):
      for ema_var, value in zip(ema_vars, values):
        value = strategy.extended.reduce_to(tf.distribute.ReduceOp.MEAN, value,
                                            ema_var)
        if ema_var.trainable:
          strategy.extended.update(ema_var, _update_one_var_fn, args=(value,))
        else:
          _update_one_var_fn(ema_var, value)

    replica_context = tf.distribute.get_replica_context()
    if replica_context:
      replica_context.merge_call(
          _update_all_in_cross_replica_context_fn,
          args=(ema_variables, new_values))
    else:
      if tf.distribute.in_cross_replica_context():
        _update_all_in_cross_replica_context_fn(tf.distribute.get_strategy(),
                                                ema_variables, new_values)
      else:
        for ema_var, value in zip(ema_variables, new_values):
          _update_one_var_fn(ema_var, value)


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


class DistributionStrategyHelper(object):

  def __init__(self, tpu=None, strategy='Mirrored'):
    """Creates distribution strategy.

    Returns:
      distribution strategy.

    If flag --tpu is set then TPU distribution strategy will be created,
    otherwise mirrored strategy running on local GPUs will be created.
    """
    if tpu:
      print(INFO, 'Use TPU at %s', tpu)
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      distribution_strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
      if strategy:
        assert strategy in [
            'Mirrored',
            'MultiWorkerMirrored',
            'CentralStorage',
            'ParameterServer',
            'OneDevice',
        ]
        print(INFO, f'Using {strategy}Strategy on local devices.')
        distribution_strategy = eval(f'tf.distribute.{strategy}Strategy()')
      else:
        print(INFO, 'Don\'t Using DistributionStrategy on local devices.')
        distribution_strategy = None
    self.strategy: tf.distribute.MirroredStrategy = distribution_strategy

  def get_strategy_scope(self):
    if self.strategy:
      strategy_scope = self.strategy.scope()
    else:
      strategy_scope = DummyContextManager()

    return strategy_scope

  def get_strategy_dataset(self, *args):
    if self.strategy:
      return (self.strategy.experimental_distribute_dataset(ds) for ds in args)
    else:
      return tuple(args)


class BaseSummaryHelper():

  def __init__(self,
               writer,
               write_dir: str,
               full_write_dir: str,
               is_write_graph: bool,
               profile_batch: int,
               is_tracing: bool = False,
               global_seen: int = 0,
               optimizer: tf.optimizers.Optimizer = None):
    self.writer = writer
    self.write_dir = write_dir
    self.full_write_dir = full_write_dir
    self.is_write_graph = is_write_graph
    self.profile_batch = profile_batch
    self.is_tracing = is_tracing
    self.global_seen = global_seen
    self.optimizer = optimizer

  def current_step(self):
    return self.optimizer.iterations.numpy()

  def write_graph(self, model: k.Model):
    """Sets Keras model and writes graph if specified."""
    if model and self.is_write_graph:
      with self.writer.as_default(), summary_ops_v2.always_record_summaries():
        if not model.run_eagerly:
          summary_ops_v2.graph(get_graph(), step=0)

        summary_writable = (
            model._is_graph_network or  # pylint: disable=protected-access
            model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
        if summary_writable:
          summary_ops_v2.keras_model('keras', model, step=0)

  def enable_trace(self):
    tf.summary.trace_on(graph=True, profiler=True)
    self.is_tracing = True

  def save_trace(self):
    """Saves tensorboard profile to event file."""
    step = self.current_step()
    with self.writer.as_default(), summary_ops_v2.always_record_summaries():
      tf.summary.trace_export(
          name='profile_batch', step=step, profiler_outdir=self.full_write_dir)
    self.is_tracing = False

  def update_seen(self):
    self.global_seen += 1

  def save_metrics(self, metrics: dict):
    """Saves metrics to event file.
    
    Args:
        metrics (dict): {'name':scalar,'name1':scalar}
    """
    step = self.current_step()
    with self.writer.as_default():
      for k, v in metrics.items():
        tf.summary.scalar(k, v, step=step)

  def save_images(self, img_pairs: dict, max_outputs=3):
    """ Saves image to event file.
      
      NOTE: image shape must be [b, h, w, c]
      
    Args:
        metrics (dict): {'name':image,'name1':image1}
    """
    step = self.current_step()
    with self.writer.as_default():
      for k, v in img_pairs.items():
        tf.summary.image(k, v, step=step, max_outputs=max_outputs)


class BaseHelperV2(object):

  def __init__(self,
               dataset_root: str,
               in_hw: list,
               mixed_precision_dtype: str,
               hparams: dict = None):
    """ 
      BaseHelperV2
    
    Args:
        dataset_root (str): dataset dir or somethings
        in_hw (list): default [256,256]
        mixed_precision_dtype (str): 
        hparams (dict): can be any things
    """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    self.train_list: str = None
    self.val_list: str = None
    self.test_list: str = None
    self.unlabel_list: str = None

    self.dataset_root = dataset_root
    self.in_hw: list = in_hw
    self.mixed_precision_dtype = mixed_precision_dtype
    self.hparams = EasyDict(hparams)
    self.set_datasetlist()

  @abc.abstractclassmethod
  def set_datasetlist(self):
    """you must overwrite this function to setup:
       `self.train_list, self.val_list, self.test_list
        self.train_total_data, self.val_total_data, self.test_total_data`
    """
    raise NotImplementedError

  @abc.abstractclassmethod
  def build_train_datapipe(self,
                           batch_size: int,
                           is_augment: bool,
                           is_normalize: bool = True) -> tf.data.Dataset:
    raise NotImplementedError

  @abc.abstractclassmethod
  def build_val_datapipe(self, batch_size: int,
                         is_normalize: bool = True) -> tf.data.Dataset:
    raise NotImplementedError

  def set_dataset(self, batch_size, is_augment, is_normalize: bool = True):
    self.batch_size = batch_size
    self.train_dataset = self.build_train_datapipe(batch_size, is_augment,
                                                   is_normalize)
    self.val_dataset = self.build_val_datapipe(batch_size, is_normalize)
    self.epoch_step = self.train_total_data // self.batch_size
    self.val_epoch_step = self.val_total_data // self.batch_size


class BaseTrainingLoop():

  def __init__(self, train_model: k.Model, val_model: k.Model,
               optimizer: k.optimizers.Optimizer,
               strategy: tf.distribute.Strategy, **kwargs: dict):
    """ Training Loop initial
      
      if use kwargs, must contain `hparams`, NOTE hparams contain all extra features.
      
      1.  exponential moving average:
          
          ema training model variable to validation model.
          NOTE if enable ema, must upate ema in `each train step function`
          Args:
            ema : {
              enable (bool): true or false
              decay (float): ema decay rate, recommend 0.999
            }
            
    Args:
        train_model (k.Model): training model
        val_model (k.Model): validation model
    """
    self.train_model = train_model
    self.val_model = val_model
    self.optimizer = optimizer
    self.strategy = strategy
    self.models_dict: Mapping[str, k.Model] = {
        'train_model': self.train_model,
        'val_model': self.val_model,
    }
    if kwargs:
      assert 'hparams' in kwargs.keys(), 'if use kwargs, must contain hparams !'
      # NOTE hparams contain all extra features
      self.hparams = EasyDict(kwargs['hparams'])
      if 'ema' in self.hparams.keys():
        if self.hparams.ema.enable:
          self.ema = EmaHelper(self.val_model, self.hparams.ema.decay)
          self.models_dict.setdefault('ema_model', self.ema.model)
    self.metrics = EasyDict(self.set_metrics_dict_wraper())

  @abc.abstractclassmethod
  def local_variables_init(self):
    """ init some variables for training """
    pass

  @abc.abstractmethod
  def train_step(self, iterator, num_steps_to_run, metrics: EasyDict):
    """Training StepFn."""
    pass

  @abc.abstractmethod
  def val_step(self, dataset: tf.data.Dataset, metrics: EasyDict):
    """Evaluation StepFn."""
    pass

  def optimizer_minimize(self, loss: tf.Tensor, tape: tf.GradientTape,
                         optimizer: tf.optimizers.Optimizer, model: k.Model):
    """apply gradients
    
    Args:
        loss (tf.Tensor): 
        tape (tf.GradientTape): 
        optimizer (tf.optimizers.Optimizer): 
        model (k.Model):
    """
    with tape:
      scaled_loss = loss / self.strategy.num_replicas_in_sync
      if isinstance(optimizer,
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(loss)

    grad = tape.gradient(scaled_loss, model.trainable_variables)
    if isinstance(optimizer,
                  tf.keras.mixed_precision.experimental.LossScaleOptimizer):
      grad = optimizer.get_unscaled_gradients(grad)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return scaled_loss

  def set_summary_writer(self,
                         write_dir: str,
                         sub_dir: str = 'train',
                         is_write_graph=True,
                         profile_batch=2):

    full_write_dir = os.path.join(write_dir, sub_dir)
    self.summary = BaseSummaryHelper(
        writer=tf.summary.create_file_writer(full_write_dir),
        write_dir=write_dir,
        full_write_dir=full_write_dir,
        is_write_graph=is_write_graph,
        profile_batch=profile_batch,
        is_tracing=False,
        global_seen=0,
        optimizer=self.optimizer)
    assert self.summary.profile_batch > 1, 'NOTE: summary.profile_batch > 1'
    self.summary.write_graph(self.train_model)

  @abc.abstractclassmethod
  def set_metrics_dict(self) -> dict:
    return None

  def set_metrics_dict_wraper(self) -> dict:
    d = self.set_metrics_dict()
    d.setdefault('debug', {})
    return d

  def set_callbacks(self, callbacks: list, model: k.Model = None):
    """Configures callbacks for use in various training loops.
        """
    callback_list = CallbackList(callbacks)
    callback_list.set_model(self.train_model if model == None else model)
    callback_list.model.stop_training = False
    self.callback_list = callback_list

  @staticmethod
  def _make_logs(prefix: str, metrics: EasyDict) -> dict:
    return dict([
        (prefix + '/' + k, v.result().numpy()) for (k, v) in metrics.items()
    ])

  def set_dataset(self, train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset, train_epoch_step: int,
                  val_epoch_step: int):
    self.train_iter: Iterator[tf.Tensor] = iter(train_dataset)
    self.val_dataset = val_dataset
    self.train_epoch_step = train_epoch_step
    self.val_epoch_step = val_epoch_step

  def train_and_eval(self, epochs, initial_epoch=0, steps_per_run=1):
    """ training variables init """
    self.local_variables_init()
    """ training and eval loop """
    train_target_steps = self.train_epoch_step // steps_per_run
    print(
        f'Train {train_target_steps} steps, Validate {self.val_epoch_step} steps')
    """ Set Progbar Log Param"""
    probar_metrics = set(self.metrics.train.keys())
    probar_metrics = probar_metrics.union(
        set(['val_' + m for m in probar_metrics]))
    """ Init Callbacks """
    self.callback_list.on_train_begin()
    """ Start training loop """
    for cur_ep in range(initial_epoch, epochs):
      if self.train_model.stop_training:
        break
      print('Epoch %d/%d' % (cur_ep + 1, epochs))
      self.callback_list.on_epoch_begin(cur_ep + 1)
      probar = MProgbar(train_target_steps + 1, stateful_metrics=probar_metrics)
      for seen in range(1, train_target_steps + 1):
        self.train_step(self.train_iter, tf.constant(steps_per_run),
                        self.metrics.train)
        # write something to tensorboard
        if self.summary.is_tracing:
          self.summary.save_trace()
        elif (not self.summary.is_tracing and
              self.summary.global_seen == self.summary.profile_batch - 1):
          self.summary.enable_trace()

        train_logs = self._make_logs('train', self.metrics.train)
        train_logs['train/lr'] = self.optimizer.learning_rate.numpy()
        self.summary.save_metrics(train_logs)
        debug_logs = self._make_logs('debug', self.metrics.debug)
        self.summary.save_metrics(debug_logs)
        self.summary.update_seen()
        probar.update(seen, probar._make_logs_value(self.metrics.train))
      """ Start Validation """
      self.val_step(self.val_dataset, self.metrics.val)
      val_logs = self._make_logs('val', self.metrics.val)
      self.summary.save_metrics(val_logs)
      probar.update(
          seen + 1,
          probar._make_logs_value(self.metrics.train) +
          probar._make_logs_value(self.metrics.val, 'val/'))
      """ Epoch Callbacks """
      val_logs.update(train_logs)
      self.callback_list.on_epoch_end(cur_ep, val_logs)
      """ Rset Metrics States """
      [v.reset_states() for v in self.metrics.train.values()]
      [v.reset_states() for v in self.metrics.val.values()]
    self.callback_list.on_train_end(val_logs)
    self.summary.writer.close()
    return cur_ep + 1

  def save_models(self, finally_epoch: int):
    """save all models in training loop models_dict
    
    Args:
        finally_epoch (int): finshed epoch
    """
    for key, v in self.models_dict.items():
      save_path = os.path.join(self.summary.write_dir,
                               f'{key}-{finally_epoch}.h5')
      if isinstance(v, k.Model):
        k.models.save_model(v, save_path)
        print(INFO, f'Save {key} as {save_path}')


class GanBaseTrainingLoop(BaseTrainingLoop):

  def __init__(self, generator_model: k.Model, discriminator_model: k.Model,
               val_model: k.Model, generator_optimizer: k.optimizers.Optimizer,
               discriminator_optimizer: k.optimizers.Optimizer,
               strategy: tf.distribute.Strategy, **kwargs):
    """GanBaseTrainingLoop
    
      NOTE: inner class, 
            self.g_model = generator_model
            self.d_model = discriminator_model
            self.g_optimizer = generator_optimizer
            self.d_optimizer = discriminator_optimizer
    
    Args:
        generator_model (k.Model): generator_model
        discriminator_model (k.Model): discriminator_model
        val_model (k.Model): val_model
        generator_optimizer (k.optimizers.Optimizer): generator_optimizer
        discriminator_optimizer (k.optimizers.Optimizer): discriminator_optimizer
        strategy (tf.distribute.Strategy): strategy
    """
    self.train_model = self.g_model = generator_model
    self.d_model = discriminator_model
    self.g_optimizer = self.optimizer = generator_optimizer
    self.d_optimizer = discriminator_optimizer
    self.val_model = val_model
    self.strategy = strategy
    self.models_dict: Mapping[str, k.Model] = {
        'generator_model': self.g_model,
        'discriminator_model': self.d_model,
        'val_model': self.val_model,
    }
    if kwargs:
      assert 'hparams' in kwargs.keys(), 'if use kwargs, must contain hparams !'
      # NOTE hparams contain all extra features
      self.hparams = EasyDict(kwargs['hparams'])
      if 'ema' in self.hparams.keys():
        if self.hparams.ema.enable:
          self.ema = EmaHelper(self.val_model, self.hparams.ema.decay)
          self.models_dict.setdefault('ema_model', self.ema.model)
    self.metrics = EasyDict(self.set_metrics_dict_wraper())


class MProgbar(Progbar):

  @staticmethod
  def _make_logs_value(metrics: EasyDict, prefix='') -> list:
    return [(prefix + k, v.result().numpy()) for (k, v) in metrics.items()]

  def update(self, current, values=None):
    """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        # In the case that progress bar doesn't have a target value in the first
        # epoch, both on_batch_end and on_epoch_end will be called, which will
        # cause 'current' and 'self._seen_so_far' to have the same value. Force
        # the minimal value to 1 here, otherwise stateful_metric will be 0s.
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and self.target is not None and
          current < self.target):
        return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width-1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current

      self._total_width = len(bar)
      sys.stdout.write(bar)

      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600, (eta%3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

      for k in self._values_order:
        info += ' %s: ' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += '%.3f' % avg
          else:
            info += '%.3e' % avg
        else:
          info += '%s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      if self.target is not None and current >= self.target:
        info += '\n'

      sys.stdout.write(info)
      sys.stdout.flush()

    elif self.verbose == 2:
      if self.target is not None and current >= self.target:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.3f' % avg
          else:
            info += ' %.3e' % avg
        info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

    self._last_update = now