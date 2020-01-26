import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.ops.math_ops import reduce_mean, reduce_sum,\
    sigmoid, sqrt, square, logical_and, cast, logical_not, div_no_nan, add
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper, Sum
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import signal
from tools.base import NOTE, colored, ERROR
from toolz import reduce
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


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

# NOTE from https://github.com/bojone/keras_lookahead


class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610)
     functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model: keras.Model):
        has_recompiled = model._recompile_weights_loss_and_weighted_metrics()
        model._check_trainable_weights_consistency()
        if isinstance(model.optimizer, list):
            raise ValueError('The `optimizer` in `compile` should be a single '
                             'optimizer.')
        # If we have re-compiled the loss/weighted metric sub-graphs then create
        # train function even if one exists already. This is because
        # `_feed_sample_weights` list has been updated on re-copmpile.
        if getattr(model, 'train_function', None) is None or has_recompiled:
            current_trainable_state = model._get_trainable_state()
            model._set_trainable_state(model._compiled_trainable_state)

            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if not isinstance(K.symbolic_learning_phase(), int):
                inputs += [K.symbolic_learning_phase()]

            with K.get_graph().as_default():
                with K.name_scope('training'):
                    # Training updates
                    fast_params = model._collected_trainable_weights
                    training_updates = model.optimizer.get_updates(
                        params=fast_params, loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]

                    fast_updates = (
                        training_updates +
                        model.get_updates_for(None) +
                        model.get_updates_for(model.inputs)
                    )
                metrics = model._get_training_eval_metrics()
                metrics_tensors = [
                    m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
                ]

            with K.name_scope('training'):
                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [model.total_loss] + metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                setattr(model, 'train_function', F)
            # Restore the current trainable state
            model._set_trainable_state(current_trainable_state)


class PFLDMetric(Metric):
    def __init__(self, calc_fr: bool, landmark_num: int, batch_size: int, name=None, dtype=None):
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
            error_all_points = reduce_sum(sqrt(reduce_sum(square(
                reshape(pred_landmarks, (self.batch_size, self.landmark_num, 2)) -
                reshape(true_landmarks, (self.batch_size, self.landmark_num, 2))), [2])), 1)

            # use interocular distance calc landmark error
            interocular_distance = sqrt(
                reduce_sum(
                    square((true_landmarks[:, 120:122] -
                            true_landmarks[:, 144:146])), 1))
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
                print(f"\n {NOTE} Received SIGINT to stop now. {colored('Please Wait !','red')}")

        signal.signal(sig, signal_handler)

    def on_batch_end(self, batch, logs=None):
        if self.signal_received:
            self.model.stop_training = True


class StepLR(Callback):
    def __init__(self, rates: list, steps: list):
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
        super().__init__()
        assert len(rates) == len(steps), f'{ERROR} the len(rates) must equal len(steps)'
        assert steps[0] > 0, f'{ERROR} the steps[0] can\'t <= 0'
        self.rates = []
        steps.insert(0, 0)
        for i in range(len(rates)):
            self.rates += [rates[i]] * (steps[i + 1] - steps[i])

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < len(self.rates):
            K.set_value(self.model.optimizer.lr, self.rates[epoch])
        else:
            K.set_value(self.model.optimizer.lr, self.rates[-1])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class CosineLR(Callback):
    def __init__(self, init_lr: float, decay_steps: int, lowest_lr: float):
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
        super().__init__()
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
        K.set_value(self.model.optimizer.lr, self.decayed_learning_rate(epoch))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
