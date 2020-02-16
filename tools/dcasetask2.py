import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tools.base import BaseHelper, INFO
from typing import List

k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


class DCASETask5Helper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: list, fold: int):
        self.train_dataset: tf.data.Dataset = None
        self.val_dataset: tf.data.Dataset = None
        self.test_dataset: tf.data.Dataset = None

        self.train_epoch_step: int = None
        self.val_epoch_step: int = None
        self.test_epoch_step: int = None
        self.fold = fold
        if image_ann == None:
            self.train_list: np.ndarray = None
            self.val_list: np.ndarray = None
            self.test_list: np.ndarray = None
        else:
            img_ann_list = np.load(image_ann, allow_pickle=True)[()]
            # NOTE can use dict set trian and test dataset
            self.unlabel_list: str = img_ann_list['unlabel_data']
            self.train_list: str = img_ann_list['train_data'][self.fold]
            self.val_list: str = img_ann_list['val_data'][self.fold]
            self.test_list: str = 0
            self.train_total_data: int = img_ann_list['train_num'][self.fold]
            self.val_total_data: int = img_ann_list['val_num'][self.fold]
            self.test_total_data: int = 0
        self.in_hw: tf.Tensor = tf.constant(in_hw, tf.int64)

    @staticmethod
    def parser_example(stream: bytes) -> [tf.Tensor, tf.Tensor]:
        example = tf.io.parse_single_example(stream, {
            'mel_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.VarLenFeature(tf.int64)})
        return example['mel_raw'], example['label'].values

    @staticmethod
    def decode_img(s: tf.Tensor) -> tf.Tensor:
        return tf.io.parse_tensor(s, tf.float32)

    @staticmethod
    def resize_img(img: tf.Tensor, in_hw: tf.Tensor,
                   ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        """
        resize image and keep ratio

        Parameters
        ----------
        img : tf.Tensor

        ann : tf.Tensor


        Returns
        -------
        [tf.Tensor, tf.Tensor]
            img, ann [ uint8 , float32 ]
        """
        img_hw = tf.shape(img, tf.int64)[:2]
        crop_len = in_hw[1]
        if img_hw[1] < crop_len:
            shift = tf.random.uniform([], 0, crop_len - img_hw[1], dtype=tf.int64)
            new_img = tf.pad(img, [[0, 0, ], [shift, crop_len - img_hw[1] - shift]])
        elif img_hw[1] == crop_len:
            new_img = img
        else:
            shift = tf.random.uniform([], 0, img_hw[1] - crop_len, dtype=tf.int64)
            new_img = img[:, shift:shift + crop_len]

        return new_img, ann

    @staticmethod
    def resize_train_img(img: tf.Tensor, in_hw: tf.Tensor,
                         ann: tf.Tensor, crop_rate=0.25
                         ) -> [tf.Tensor, tf.Tensor]:
        """ when training first crop image and resize image and keep ratio

        Parameters
        ----------
        img : tf.Tensor

        in_hw : tf.Tensor

        ann : tf.Tensor

        Returns
        -------
        [tf.Tensor, tf.Tensor]
            img, ann
        """
        img_hw = tf.shape(img, tf.int64)[:2]
        crop_len = in_hw[1]
        rate = tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                       lambda: tf.ones([]),
                       lambda: tf.random.uniform([], 0, 1., tf.float32) * (1 - crop_rate) + crop_rate)

        if img_hw[1] <= crop_len:
            _len = tf.cast(tf.cast(img_hw[1], tf.float32) * rate, tf.int64)
            if img_hw[1] - _len == 0:
                shift_crop = tf.zeros([], tf.int64)
            else:
                shift_crop = tf.random.uniform([], 0, img_hw[1] - _len, tf.int64)
            img = img[:, shift_crop:shift_crop + _len]
            if crop_len - _len == 0:
                shift = tf.zeros([], tf.int64)
            else:
                shift = tf.random.uniform([], 0, crop_len - _len, tf.int64)
            new_img = tf.pad(img, [[0, 0, ], [shift, crop_len - _len - shift]])
        else:
            shift = tf.random.uniform([], 0, img_hw[1] - crop_len, tf.int64)
            new_img = img[:, shift:shift + crop_len]
            _len = tf.cast(tf.cast(crop_len, tf.float32) * rate, tf.int64)
            if (crop_len - _len) == 0:
                shift_crop = tf.zeros([], tf.int64)
            else:
                shift_crop = tf.random.uniform([], 0, crop_len - _len, tf.int64)
            new_img = tf.concat([tf.zeros_like(new_img[:shift_crop]),
                                 new_img[shift_crop:shift_crop + _len],
                                 tf.zeros_like(new_img[shift_crop + _len:])], 0)
        return new_img, ann

    @staticmethod
    def mixup_img(imga, anna, imgb, annb) -> [tf.Tensor, tf.Tensor]:
        rate = tfp.distributions.Beta(1., 1.).sample([])
        img = imga * rate + imgb * (1 - rate)
        ann = tf.cast(anna, tf.float32) * rate + tf.cast(annb, tf.float32) * (1 - rate)
        return img, ann

    @staticmethod
    def gain_img(img, gainv=0.1) -> tf.Tensor:
        rate = 1 - gainv + tf.random.uniform([], 0, 1) * gainv * 2
        new_img = img * rate
        return new_img

    @staticmethod
    def freqmask_img(img, max_w=26):
        coord = tf.random.uniform([], 0, tf.shape(img, tf.int64)[0], tf.int64)
        width = tf.random.uniform([], 8, max_w, tf.int64)
        cut = tf.stack([coord - width, coord + width])
        cut = tf.clip_by_value(cut, 0, tf.shape(img, tf.int64)[0])
        new_img = tf.concat([img[:cut[0]],
                             tf.zeros_like(img[cut[0]:cut[1]]),
                             img[cut[1]:]], 0)
        return new_img

    @staticmethod
    def power_to_db(img, ref=1.0, amin=1e-10, top_db=80.0):
        magnitude = img
        ref_value = tf.abs(ref)
        log_spec = 10.0 * (tf.math.log(tf.maximum(amin, magnitude)) / tf.math.log(10.))
        log_spec -= 10.0 * (tf.math.log(tf.maximum(amin, ref_value)) / tf.math.log(10.))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    def augment_img(self, imga, anna, imgb, annb) -> [tf.Tensor, tf.Tensor]:
        img, ann = tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                           lambda: self.mixup_img(imga, anna, imgb, annb),
                           lambda: tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                                           lambda: (imga, tf.cast(anna, tf.float32)),
                                           lambda: (imgb, tf.cast(annb, tf.float32))))
        img = tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                      lambda: self.freqmask_img(img),
                      lambda: img)
        return img, ann

    def normlize_img(self, img) -> tf.Tensor:
        img = self.power_to_db(img)
        img = (img - tf.reduce_mean(img)) / (tf.math.reduce_std(img) + 1e-7)
        new_img = img[..., None]
        return new_img

    def process_img(self, imga: tf.Tensor, anna: tf.Tensor,
                    imgb: tf.Tensor, annb: tf.Tensor,
                    in_hw: tf.Tensor, is_augment: bool,
                    is_resize: bool,
                    is_normlize: bool) -> [tf.Tensor, tf.Tensor]:
        """ process image and label , if is training then use data augmenter
        """
        if is_resize and is_augment:
            imga, anna = self.resize_train_img(imga, in_hw, anna)
            imgb, annb = self.resize_train_img(imgb, in_hw, annb)
        elif is_resize:
            imga, anna = self.resize_img(imga, in_hw, anna)
            imgb, annb = self.resize_img(imgb, in_hw, annb)
        if is_augment:
            img, ann = self.augment_img(imga, anna, imgb, annb)
        else:
            img, ann = tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                               lambda: (imga, tf.cast(anna, tf.float32)),
                               lambda: (imgb, tf.cast(annb, tf.float32)))
        if is_normlize:
            img = self.normlize_img(img)
        else:
            img = tf.cast(img, tf.float32)
        return img, ann

    def build_train_datapipe(self, batch_size: int, is_augment: bool,
                             is_normlize: bool) -> tf.data.Dataset:

        def _parser(train_stream: List[tf.Tensor], unlabel_stream: List[tf.Tensor]):
            stream = train_stream
            mel_rawa, anna = self.parser_example(stream[0])
            mel_rawb, annb = self.parser_example(stream[1])
            imga = self.decode_img(mel_rawa)
            imgb = self.decode_img(mel_rawb)
            imga.set_shape((None, None))
            imgb.set_shape((None, None))
            anna.set_shape((None))
            annb.set_shape((None))
            img, label = self.process_img(imga, anna, imgb, annb, self.in_hw,
                                          is_augment, True, is_normlize)
            train_img, train_label = img, label

            stream = unlabel_stream
            mel_rawa, anna = self.parser_example(stream[0])
            mel_rawb, annb = self.parser_example(stream[1])
            imga = self.decode_img(mel_rawa)
            imgb = self.decode_img(mel_rawb)
            imga.set_shape((None, None))
            imgb.set_shape((None, None))
            anna.set_shape((None))
            annb.set_shape((None))
            img, label = self.process_img(imga, anna, imgb, annb, self.in_hw,
                                          is_augment, True, is_normlize)
            unlabel_img, unlabel_label = img, label
            return (train_img, unlabel_img), tf.concat([train_label, unlabel_label], -1)

        ds = (tf.data.Dataset.zip((tf.data.TFRecordDataset(self.train_list, None, None, 4),
                                   tf.data.TFRecordDataset(self.unlabel_list, None, None, 4))).
              repeat().
              batch(2, True).
              map(_parser, -1).
              batch(batch_size, True).
              prefetch(-1))

        return ds

    def build_val_datapipe(self, batch_size: int,
                           is_augment: bool, is_normlize: bool
                           ) -> tf.data.Dataset:

        def _parser(stream: List[bytes]):
            mel_rawa, anna = self.parser_example(stream[0])
            mel_rawb, annb = self.parser_example(stream[1])
            imga = self.decode_img(mel_rawa)
            imgb = self.decode_img(mel_rawb)
            imga.set_shape((None, None))
            imgb.set_shape((None, None))

            img, label = self.process_img(imga, anna, imgb, annb, self.in_hw,
                                          is_augment, True, is_normlize)
            return img, label

        ds = (tf.data.TFRecordDataset(self.val_list,
                                      None, None, 4).
              repeat().
              batch(2, True).
              map(_parser, -1).
              batch(batch_size, True).
              prefetch(-1))

        return ds

    def set_dataset(self, batch_size: int, is_augment: bool = True,
                    is_normlize: bool = True, is_training: bool = True):
        self.batch_size = batch_size
        if is_training:
            self.train_dataset = self.build_train_datapipe(batch_size, is_augment, is_normlize)
            self.val_dataset = self.build_val_datapipe(batch_size, False, is_normlize)

            self.train_epoch_step = self.train_total_data // self.batch_size
            self.val_epoch_step = self.val_total_data // self.batch_size
        else:
            self.test_dataset = self.build_datapipe(self.test_list, batch_size,
                                                    False, is_normlize, is_training)
            self.test_epoch_step = self.test_total_data // self.batch_size


class FixMatchHelper(DCASETask5Helper):
    def __init__(self, image_ann, in_hw, fold):
        super().__init__(image_ann, in_hw, fold)

    def process_img(self, img, ann, in_hw, is_augment, is_resize, is_normlize):
        """ is_augment=0 is none aug
            is_augment=1 is weak aug
            is_augment=2 is strong aug
        """
        if is_resize and is_augment == 0:
            img, ann = self.resize_img(img, in_hw, ann)
        elif is_resize and is_augment == 1:
            img, ann = self.resize_img(img, in_hw, ann)
        elif is_resize and is_augment == 2:
            img, ann = self.resize_train_img(img, in_hw, ann)
        if is_augment == 2:
            img = tf.cond(tf.random.uniform([], 0, 1.) < 0.5,
                          lambda: self.freqmask_img(img),
                          lambda: img)
        if is_normlize:
            img = self.normlize_img(img)
        return img, ann

    def build_train_datapipe(self, batch_size: int, naugment: int, is_normlize: bool) -> tf.data.Dataset:

        def labeled_parser(stream: tf.Tensor):
            mel_raw, ann = self.parser_example(stream)
            img = self.decode_img(mel_raw)
            img.set_shape((None, None))
            ann.set_shape((None))
            img, label = self.process_img(img, ann, self.in_hw,
                                          1, True, is_normlize)
            return img, label

        def unlabel_parser(stream: tf.Tensor):
            mel_raw, ann = self.parser_example(stream)
            img = self.decode_img(mel_raw)
            img.set_shape((None, None))
            ann.set_shape((None))
            weak_img, _ = self.process_img(img, ann, self.in_hw,
                                           1, True, is_normlize)
            strong_img, _ = self.process_img(img, ann, self.in_hw,
                                             2, True, is_normlize)
            return weak_img, strong_img

        ds_labeled = (tf.data.TFRecordDataset(self.train_list).
                      shuffle(200).
                      repeat().
                      map(labeled_parser, -1).
                      batch(batch_size, True).
                      prefetch(-1))

        ds_unlabeled = (tf.data.TFRecordDataset(self.unlabel_list).
                        shuffle(200).
                        repeat().
                        map(unlabel_parser, -1).
                        batch(batch_size * naugment, True).
                        prefetch(-1))

        return tf.data.Dataset.zip((ds_labeled, ds_unlabeled))

    def build_val_datapipe(self, batch_size: int, is_normlize: bool) -> tf.data.Dataset:

        def labeled_parser(stream: tf.Tensor):
            mel_raw, ann = self.parser_example(stream)
            img = self.decode_img(mel_raw)
            img.set_shape((None, None))
            ann.set_shape((None))
            img, label = self.process_img(img, ann, self.in_hw,
                                          0, True, is_normlize)
            return img, label

        ds_labeled = (tf.data.TFRecordDataset(self.val_list).
                      shuffle(200).
                      repeat().
                      map(labeled_parser, -1).
                      batch(batch_size, True).
                      prefetch(-1))

        return ds_labeled


class SemiBCELoss(k.losses.Loss):
    def __init__(self, reduction='auto', name=None):
        super().__init__(reduction=reduction, name=name)
        self.lwlrap = tf.Variable(0., False, dtype=tf.float32, shape=())
        self.lwlrap_noisy = tf.Variable(0., False, dtype=tf.float32, shape=())

    @staticmethod
    def per_class_lwlrap(y_true: tf.Tensor, y_pred: tf.Tensor):
        def one_sample_positive_class_precisions(i: tf.Tensor):
            """Calculate precisions for each true class for a single sample.

            """
            score = y_pred[i]
            truth = y_true[i]
            pos_class_indices = tf.where(truth > 0)

            def false_fn():
                # Retrieval list of classes for this sample.
                retrieved_classes = tf.argsort(score)[::-1]
                # class_rankings[top_scoring_class_index] == 0 etc.
                class_rankings = tf.range(tf.shape(truth)[0])
                class_rankings = tf.gather(class_rankings, retrieved_classes)
                # Which of these is a true label?
                retrieved_class_idx = tf.gather(class_rankings, pos_class_indices)
                retrieved_class_true = tf.scatter_nd(retrieved_class_idx[:, None],
                                                     tf.ones_like(retrieved_class_idx, dtype=tf.bool), tf.shape(truth))

                # Num hits for every truncated retrieval list.
                retrieved_cumulative_hits = tf.cumsum(tf.cast(retrieved_class_true, tf.float32))
                # Precision of retrieval list truncated at each hit, in order of pos_labels.
                precision_at_hits = (
                    tf.gather(retrieved_cumulative_hits, retrieved_class_idx) /
                    (1 + tf.cast(retrieved_class_idx, tf.float32)))

                return tf.scatter_nd(tf.cast(pos_class_indices, tf.int32),
                                     precision_at_hits[:, 0], tf.shape(truth))

            # Only calculate precisions if there are some true classes.
            return tf.cond(tf.equal(tf.size(pos_class_indices), 0),
                           lambda: tf.zeros_like(score, tf.float32),
                           lambda: false_fn())

        # 只填写每个样本的正确类。
        precisions_for_samples_by_classes = tf.map_fn(one_sample_positive_class_precisions,
                                                      tf.range(tf.shape(y_pred)[0]), tf.float32)

        labels_per_class = tf.reduce_sum(tf.cast(y_true > 0, tf.float32), axis=0)
        weight_per_class = labels_per_class / tf.cast(tf.reduce_sum(labels_per_class), tf.float32)
        # 每列的平均值，即指定给特定类别标签的所有精度。
        per_class_lwlrap = (tf.reduce_sum(precisions_for_samples_by_classes, axis=0) /
                            tf.maximum(1., labels_per_class))
        return per_class_lwlrap, weight_per_class

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        target, target_noisy = tf.split(y_true, 2, -1)
        output, output_noisy = tf.split(y_pred, 2, -1)
        pred = tf.sigmoid(output)
        pred_noisy = tf.sigmoid(output_noisy)

        bce = tf.nn.sigmoid_cross_entropy_with_logits(target, output)
        bce_noisy = tf.nn.sigmoid_cross_entropy_with_logits(target_noisy, pred_noisy)

        per_class_lwlrap, weight_per_class = self.per_class_lwlrap(target, pred)
        self.lwlrap.assign(tf.reduce_sum(per_class_lwlrap * weight_per_class))
        per_class_lwlrap, weight_per_class = self.per_class_lwlrap(target_noisy, pred_noisy)
        self.lwlrap_noisy.assign(tf.reduce_sum(per_class_lwlrap * weight_per_class))
        loss = tf.reduce_sum(bce + bce_noisy)
        return loss


class LwlrapValidation(k.callbacks.Callback):
    def __init__(self, validation_model: k.Model, validation_data: tf.data.Dataset,
                 validation_steps: int, trian_steps: int, lwlrap_var: tf.Variable):
        self.val_model = validation_model
        self.val_iter = iter(validation_data)
        self.trian_step = int(trian_steps)
        self.val_step = validation_steps
        self.lwlrap_var = lwlrap_var

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.trian_step:
            def fn(i: tf.Tensor):
                img, y_true = next(self.val_iter)
                y_pred: tf.Tensor = tf.sigmoid(self.val_model(img, training=False))

                per_class_lwlrap, weight_per_class = SemiBCELoss.per_class_lwlrap(y_true, y_pred)
                lwlrap = tf.reduce_sum(per_class_lwlrap * weight_per_class)
                return lwlrap
            K.set_value(self.lwlrap_var, tf.reduce_mean(tf.map_fn(fn, tf.range(self.val_step), tf.float32)))
