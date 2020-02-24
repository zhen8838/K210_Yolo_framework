import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from typing import Iterator
from tqdm import tqdm
import abc
from tensorflow.python.keras.callbacks import CallbackList
from tensorflow.python.keras.utils.generic_utils import Progbar
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


class BaseTrainingLoop():
    def __init__(self, train_model: k.Model, val_model: k.Model):
        self.train_model = train_model
        self.val_model = val_model
        self.optimizer: k.optimizers.Optimizer = train_model.optimizer
        assert self.optimizer is not None, 'train_model must have optimizer!'
        self.metrics = EasyDict(self.set_metrics_dict())

    @abc.abstractmethod
    def train_step(self, iterator, num_steps_to_run, metrics: EasyDict):
        """Training StepFn."""
        pass

    @tf.function
    def val_step(self, dataset: tf.data.Dataset, metrics: EasyDict):
        """Evaluation StepFn."""
        pass

    def get_current_train_step(self) -> int:
        """Returns current training step."""
        return self.optimizer.iterations.numpy()

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def save_metrics(self, metrics: dict):
        """Saves metrics to event file."""
        step = self.get_current_train_step()
        with self.summary_writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=step)

    @abc.abstractclassmethod
    def set_metrics_dict(self):
        return None

    def set_callbacks(self, callbacks: list, model: k.Model = None):
        """Configures callbacks for use in various training loops.
        """
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self.train_model if model == None else model)
        callback_list.model.stop_training = False
        self.callback_list = callback_list

    @staticmethod
    def _make_logs(prefix: str, metrics: EasyDict) -> dict:
        return dict([(prefix + '/' + k, v.result().numpy()) for (k, v) in metrics.items()])

    def set_dataset(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                    train_epoch_step: int, val_epoch_step: int):
        self.train_iter: Iterator[tf.Tensor] = iter(train_dataset)
        self.val_dataset = val_dataset
        self.train_epoch_step = train_epoch_step
        self.val_epoch_step = val_epoch_step

    def train_and_eval(self, epochs, initial_epoch=0, steps_per_run=1):
        """ training and eval loop """
        train_target_steps = self.train_epoch_step // steps_per_run
        print(f'Train {train_target_steps} steps, validate {self.val_epoch_step} steps')

        """ Set Progbar Log Param"""
        probar_metrics = set(self.metrics.train.keys())
        probar_metrics = probar_metrics.union(set(['val_' + m for m in probar_metrics]))
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
                self.train_step(self.train_iter, tf.constant(steps_per_run), self.metrics.train)
                train_logs = self._make_logs('train', self.metrics.train)
                train_logs['train/lr'] = self.optimizer.learning_rate.numpy()
                self.save_metrics(train_logs)
                probar.update(seen, probar._make_logs_value(self.metrics.train))
                if self.train_model.stop_training:
                    break

            """ Start Validation """
            self.val_step(self.val_dataset, self.metrics.val)
            val_logs = self._make_logs('val', self.metrics.val)
            self.save_metrics(val_logs)
            probar.update(seen + 1, probar._make_logs_value(self.metrics.train) +
                          probar._make_logs_value(self.metrics.val, 'val_'))

            """ Epoch Callbacks """
            val_logs.update(train_logs)
            self.callback_list.on_epoch_end(cur_ep, val_logs)
            """ Rset Metrics States """
            [v.reset_states() for v in self.metrics.train.values()]
            [v.reset_states() for v in self.metrics.val.values()]
        self.callback_list.on_train_end(val_logs)


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
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
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
                    bar += ('=' * (prog_width - 1))
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
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
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
                        info += '%.4f' % avg
                    else:
                        info += '%.4e' % avg
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
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now


class ScheduleLR(k.optimizers.schedules.LearningRateSchedule):
    """Configurable learning rate schedule."""

    def __init__(self,
                 steps_per_epoch,
                 base_lr,
                 use_warmup,
                 warmup_epochs,
                 decay_rate,
                 decay_epochs):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs
        if isinstance(self.decay_epochs, (list, tuple)):
            lr_values = [self.base_lr * (self.decay_rate ** k)
                         for k in range(len(self.decay_epochs) + 1)]
            self.lr_schedule_no_warmup = (
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    self.decay_epochs, lr_values))
        else:
            self.lr_schedule_no_warmup = (
                tf.keras.optimizers.schedules.ExponentialDecay(
                    self.base_lr, self.decay_epochs, self.decay_rate, staircase=True))

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'base_lr': self.base_lr,
            'use_warmup': self.use_warmup,
            'warmup_epochs': self.warmup_epochs,
            'decay_rate': self.decay_rate,
            'decay_epochs': self.decay_epochs,
        }

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        if self.use_warmup:
            return tf.cond(lr_epoch < self.warmup_epochs,
                           lambda: lr_epoch / self.warmup_epochs * self.base_lr,
                           lambda: self.lr_schedule_no_warmup(lr_epoch))
        else:
            return self.lr_schedule_no_warmup(lr_epoch)
