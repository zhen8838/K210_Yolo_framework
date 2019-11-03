import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import reduce_mean, reduce_sum,\
    sigmoid, sqrt, square, logical_and, cast, logical_not, div_no_nan, add
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.metrics import Metric, MeanMetricWrapper
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.keras.optimizers import Optimizer
import numpy as np
from tensorflow.python.ops.variables import RefVariable


YOLO_PRECISION = 0
YOLO_RECALL = 1


class Yolo_P_R(Metric):
    def __init__(self, out_metric: int, thresholds: float, name=None, dtype=None):
        """ yolo out_metric common class , set out_metric to return different metrics.

            landmark to control calc different metrics.

        YOLO_PRECISION = 0
        YOLO_RECALL = 1

        Parameters
        ----------
        Metric : [type]

        out_metric : int
            metric class
        thresholds : float

        name : [type], optional
            by default None
        dtype : [type], optional
            by default None
        """
        super(Yolo_P_R, self).__init__(name=name, dtype=dtype)
        self.out_metric = out_metric
        self.thresholds = thresholds

        if self.out_metric == YOLO_PRECISION:

            self.tp = self.add_weight(
                'tp', initializer=init_ops.zeros_initializer)  # type: RefVariable

            self.fp = self.add_weight(
                'fp', initializer=init_ops.zeros_initializer)  # type: RefVariable

            self.fn = self.add_weight(
                'fn', initializer=init_ops.zeros_initializer)  # type: RefVariable
        else:
            pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        YOLO_PRECISION : will calc PRECISION,RECALL
        YOLO_RECALL : not calc any thing.

        """
        if self.out_metric == YOLO_PRECISION:
            true_confidence = y_true[..., 4:5]
            pred_confidence = y_pred[..., 4:5]
            pred_confidence_sigmoid = sigmoid(pred_confidence)

            values = logical_and(true_confidence > self.thresholds, pred_confidence > self.thresholds)
            values = cast(values, self.dtype)
            self.tp.assign_add(reduce_sum(values))

            values = logical_and(logical_not(true_confidence > self.thresholds),
                                 pred_confidence > self.thresholds)
            values = cast(values, self.dtype)
            self.fp.assign_add(reduce_sum(values))

            values = logical_and(true_confidence > self.thresholds,
                                 logical_not(pred_confidence > self.thresholds))
            values = cast(values, self.dtype)
            self.fn.assign_add(reduce_sum(values))
        else:
            pass

    def result(self):
        if self.out_metric == YOLO_PRECISION:
            return div_no_nan(self.tp, (add(self.tp, self.fp)))
        elif self.out_metric == YOLO_RECALL:
            return div_no_nan(self.tp, (add(self.tp, self.fn)))

    def get_config(self):
        return {'out_metric': self.out_metric, 'thresholds': self.thresholds, 'name': self.name, 'dtype': self.dtype}


class DummyMetric(MeanMetricWrapper):
    def __init__(self, var: RefVariable, name: str, dtype=tf.float32):
        """ Dummy_Metric from MeanMetricWrapper

        Parameters
        ----------
        var : RefVariable

            a variable from yoloalign loss

        name : str

            dummy metric name

        dtype : [type], optional

            by default None
        """
        super().__init__(lambda y_true, y_pred, v: v, name=name, dtype=dtype, v=var.read_value())

# NOTE From https://github.com/bojone/keras_radam


class RAdam(Optimizer):
    """RAdam optimizer.
    Default parameters follow those provided in the original Adam paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [RAdam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1908.03265)
        - [On The Variance Of The Adaptive Learning Rate And Beyond]
          (https://arxiv.org/abs/1908.03265)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)
        rho = 2 / (1 - self.beta_2) - 1
        rho_t = rho - 2 * t * beta_2_t / (1 - beta_2_t)
        r_t = K.sqrt(
            K.relu(rho_t - 4) * K.relu(rho_t - 2) * rho / ((rho - 4) * (rho - 2) * rho_t)
        )
        flag = K.cast(rho_t > 4, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            mhat_t = m_t / (1 - beta_1_t)
            vhat_t = K.sqrt(v_t / (1 - beta_2_t))
            p_t = p - lr * mhat_t * (flag * r_t / (vhat_t + self.epsilon) + (1 - flag))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# NOTE from https://github.com/bojone/keras_lookahead
class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model: keras.models.Model):
        has_recompiled = model._recompile_weights_loss_and_weighted_metrics()
        metrics_tensors = [
            model._all_metrics_tensors[m] for m in model.metrics_names[1:]
        ]
        model._check_trainable_weights_consistency()
        if getattr(model, 'train_function') is None or has_recompiled:
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
            # NOTE if calculate landmark error , this variable will be use, When calculate failure rate , just return failure rate .
            self.landmark_error = self.add_weight(
                'LE', initializer=init_ops.zeros_initializer)  # type: RefVariable

            self.failure_num = self.add_weight(
                'FR', initializer=init_ops.zeros_initializer)  # type: RefVariable

            self.total = self.add_weight(
                'total', initializer=init_ops.zeros_initializer)  # type: RefVariable

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
