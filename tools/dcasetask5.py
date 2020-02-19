""" dcase2018 task5 helper functions """
from tools.base import BaseHelper
import tensorflow as tf
import numpy as np


class DCASETask5Helper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: list):
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None
        if image_ann == None:
            self.train_list: str = None
            self.val_list: str = None
            self.test_list: str = None
            self.unlabel_list: str = None
        else:
            img_ann_list = np.load(image_ann, allow_pickle=True)[()]
            # NOTE can use dict set trian and test dataset
            self.unlabel_list: str = img_ann_list['train_unlabel_data']
            self.train_list: str = img_ann_list['train_label_data']
            self.val_list: str = img_ann_list['vali_data']
            self.test_list: str = None
            self.train_total_data: int = img_ann_list['train_label_num']
            self.val_total_data: int = img_ann_list['vali_num']
            self.test_total_data: int = None
        self.in_hw: tf.Tensor = tf.constant(in_hw, tf.int64)
