from tensorflow.python import keras
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.keras.optimizers import Optimizer
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


class Yolo_Precision(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None):
        super(Yolo_Precision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds

        default_threshold = 0.5

        self.thresholds = default_threshold if thresholds is None else thresholds

        self.true_positives = self.add_weight(
            'tp', initializer=init_ops.zeros_initializer)  # type: ResourceVariable

        self.false_positives = self.add_weight(
            'fp', initializer=init_ops.zeros_initializer)  # type: ResourceVariable

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_confidence = y_true[..., 4:5]
        pred_confidence = y_pred[..., 4:5]
        pred_confidence_sigmoid = math_ops.sigmoid(pred_confidence)

        values = math_ops.logical_and(true_confidence > self.thresholds, pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.true_positives.assign_add(math_ops.reduce_sum(values))

        values = math_ops.logical_and(math_ops.logical_not(true_confidence > self.thresholds),
                                      pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.false_positives.assign_add(math_ops.reduce_sum(values))

    def result(self):
        return math_ops.div_no_nan(self.true_positives, (math_ops.add(self.true_positives, self.false_positives)))


class Yolo_Recall(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None):
        super(Yolo_Recall, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds

        default_threshold = 0.5

        self.thresholds = default_threshold if thresholds is None else thresholds

        self.true_positives = self.add_weight(
            'tp', initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'fn', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_confidence = y_true[..., 4:5]
        pred_confidence = y_pred[..., 4:5]
        pred_confidence_sigmoid = math_ops.sigmoid(pred_confidence)

        values = math_ops.logical_and(true_confidence > self.thresholds, pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.true_positives.assign_add(math_ops.reduce_sum(values))  # type: ResourceVariable

        values = math_ops.logical_and(true_confidence > self.thresholds,
                                      math_ops.logical_not(pred_confidence > self.thresholds))
        values = math_ops.cast(values, self.dtype)
        self.false_negatives.assign_add(math_ops.reduce_sum(values))  # type: ResourceVariable

    def result(self):
        return math_ops.div_no_nan(self.true_positives, (math_ops.add(self.true_positives, self.false_negatives)))

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
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()
        metrics_tensors = [
            model._all_metrics_tensors[m] for m in model.metrics_names[1:]
        ]
        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if not isinstance(K.symbolic_learning_phase(), int):
                inputs += [K.symbolic_learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                
                fast_updates = (model.updates +
                                training_updates +
                                model.get_updates_for(None) +
                                model.get_updates_for(model.inputs))

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

                model.train_function = F
