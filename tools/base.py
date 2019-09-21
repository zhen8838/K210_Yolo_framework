import tensorflow as tf
import numpy as np
import abc
from termcolor import colored

INFO = colored('[ INFO  ]', 'blue')  # type:str
ERROR = colored('[ ERROR ]', 'red')  # type:str
NOTE = colored('[ NOTE ]', 'green')  # type:str


class BaseHelper(object):
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.train_epoch_step = None
        self.test_epoch_step = None

    @abc.abstractmethod
    def set_dataset(self, batch_size, rand_seed, is_augment=True):
        NotImplementedError('Must be implemented in subclasses.')

    def read_img(self, img_path: str) -> tf.Tensor:
        """ read image """
        return tf.image.decode_image(
            tf.io.read_file(img_path), channels=3,
            dtype=tf.uint8, expand_animations=False)

    @abc.abstractmethod
    def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True):
        NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def resize_img(self, img: np.ndarray, ann: np.ndarray) -> [np.ndarray, np.ndarray]:
        NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def data_augmenter(self, img: np.ndarray, ann: np.ndarray) -> [np.ndarray, np.ndarray]:
        NotImplementedError('Must be implemented in subclasses.')

    def normlize_img(self, img: tf.Tensor) -> tf.Tensor:
        """ normlize img """
        return (tf.cast(img, tf.float32) / 255. - 0.5) / 1

    def process_img(self, img: np.ndarray, ann: np.ndarray,
                    is_augment: bool, is_resize: bool, is_normlize: bool) -> [np.ndarray, np.ndarray]:
        """ process image and true box , if is training then use data augmenter

        Parameters
        ----------
        img : np.ndarray
            image srs
        ann : np.ndarray
            one annotation
        is_augment : bool
            wether to use data augmenter
        is_resize : bool
            wether to resize the image
        is_normlize : bool
            wether to normlize the image

        Returns
        -------
        tuple
            image src , true box
        """
        if is_resize:
            img, ann = self.resize_img(img, ann)
        if is_augment:
            img, ann = self.data_augmenter(img, ann)
        if is_normlize:
            img = self.normlize_img(img)
        return img, ann
