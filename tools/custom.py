from tensorflow.python import keras
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np


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
