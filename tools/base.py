import tensorflow as tf
import numpy as np
import abc
from termcolor import colored

np.set_printoptions(suppress=True)

INFO = colored('[ INFO  ]', 'blue')  # type:str
ERROR = colored('[ ERROR ]', 'red')  # type:str
NOTE = colored('[ NOTE ]', 'green')  # type:str


class BaseHelper(object):

  def __init__(self, image_ann: str, validation_split: float):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    self.validation_split = validation_split  # type:float
    if image_ann == None:
      self.train_list: np.ndarray = None
      self.val_list: np.ndarray = None
      self.test_list: np.ndarray = None
    else:
      img_ann_list = np.load(image_ann, allow_pickle=True)

      if isinstance(img_ann_list[()], dict):
        # NOTE can use dict set trian and test dataset
        self.train_list = img_ann_list[()]['train_data']  # type:np.ndarray
        self.val_list = img_ann_list[()]['val_data']  # type:np.ndarray
        self.test_list = img_ann_list[()]['test_data']  # type:np.ndarray
      elif isinstance(img_ann_list[()], np.ndarray):
        self.train_list, self.val_list, self.test_list = np.split(
            img_ann_list, [
                int((1 - self.validation_split) * len(img_ann_list)),
                int((1 - self.validation_split / 2) * len(img_ann_list))
            ])
      else:
        raise ValueError(f'{image_ann} data format error!')
      self.train_total_data = len(self.train_list)
      self.val_total_data = len(self.val_list)
      self.test_total_data = len(self.test_list)

  @abc.abstractmethod
  def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                     is_augment: bool, is_normlize: bool,
                     is_training: bool) -> tf.data.Dataset:
    NotImplementedError('Must be implemented in subclasses.')

  def set_dataset(self,
                  batch_size: int,
                  is_augment: bool = True,
                  is_normlize: bool = True,
                  is_training: bool = True):
    self.batch_size = batch_size
    if is_training:
      self.train_dataset = self.build_datapipe(self.train_list, batch_size,
                                               is_augment, is_normlize,
                                               is_training)
      self.val_dataset = self.build_datapipe(self.val_list, batch_size, False,
                                             is_normlize, is_training)

      self.train_epoch_step = self.train_total_data // self.batch_size
      self.val_epoch_step = self.val_total_data // self.batch_size
    else:
      self.test_dataset = self.build_datapipe(self.test_list, batch_size, False,
                                              is_normlize, is_training)
      self.test_epoch_step = self.test_total_data // self.batch_size

  def read_img(self, img_path: str) -> tf.Tensor:
    """ read image """
    return tf.image.decode_image(
        tf.io.read_file(img_path),
        channels=3,
        dtype=tf.uint8,
        expand_animations=False)

  @abc.abstractmethod
  def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True):
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def resize_img(self, img: np.ndarray,
                 ann: np.ndarray) -> [np.ndarray, np.ndarray]:
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def augment_img(self, img: np.ndarray,
                  ann: np.ndarray) -> [np.ndarray, np.ndarray]:
    NotImplementedError('Must be implemented in subclasses.')

  def normlize_img(self, img: tf.Tensor) -> tf.Tensor:
    """ normlize img """
    return (tf.cast(img, tf.float32) / 255. - 0.5) / 1

  def process_img(self, img: np.ndarray, ann: np.ndarray, is_augment: bool,
                  is_resize: bool, is_normlize: bool) -> [tf.Tensor, tf.Tensor]:
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
      img, ann = self.augment_img(img, ann)
    if is_normlize:
      img = self.normlize_img(img)
    return img, ann

