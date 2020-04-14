import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools.base import BaseHelper, INFO, ERROR, NOTE
from tensorflow.python.keras.losses import LossFunctionWrapper
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import imgaug as ia
import imgaug.augmenters as iaa
from typing import List, Dict
from pathlib import Path
from tqdm import trange, tqdm
k = tf.keras
kl = tf.keras.layers


class ImgnetHelper(BaseHelper):

  def __init__(self,
               image_ann: str,
               class_num: int,
               in_hw: list,
               mixup: bool = False):
    """ ImgnetHelper

        Parameters
        ----------
        image_ann : str

            `**.npy` file path

        class_num : int

            class num

        in_hw : list

            input height weight

        """
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    self.class_num: int = class_num
    self.in_hw = in_hw
    self.mixup = mixup
    self.meta: dict = np.load(image_ann, allow_pickle=True)[()]
    self.class_name: Dict[int, str] = {
        v: k for k, v in self.meta['clas_dict'].items()
    }
    self.train_list = self.meta['train_list']
    self.val_list = self.meta['val_list']
    self.test_list = self.val_list

    self.train_total_data = self.meta['train_num']
    self.val_total_data = self.meta['val_num']
    self.test_total_data = self.val_total_data

    self.iaaseq = iaa.Sequential([
        iaa.SomeOf([1, None], [
            iaa.MultiplyHueAndSaturation(
                mul_hue=(0.7, 1.3), mul_saturation=(0.7, 1.3), per_channel=True),
            iaa.Multiply((0.5, 1.5), per_channel=True),
            iaa.SigmoidContrast((3, 8)),
        ], True),
        iaa.SomeOf([1, None], [
            iaa.Fliplr(0.5),
            iaa.Affine(scale={
                "x": (0.7, 1.3),
                "y": (0.7, 1.3)
            }, backend='cv2'),
            iaa.Affine(
                translate_percent={
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15)
                },
                backend='cv2'),
            iaa.Affine(rotate=(-15, 15), backend='cv2')
        ], True)
    ], True)

  def read_img(self, img_path: tf.Tensor) -> tf.Tensor:
    return tf.image.decode_jpeg(tf.io.read_file(img_path), 3)

  def resize_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
    img = tf.image.resize_with_pad(
        img, self.in_hw[0], self.in_hw[1], method='nearest')
    return img, ann

  # def augment_img(self, img: tf.Tensor, ann: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
  #   img = tf.numpy_function(lambda x: self.iaaseq(image=x,), [img], tf.uint8)

  #   return img, ann
  def iaa_augment_img(self, img: np.ndarray,
                      ann: np.ndarray) -> [np.ndarray, np.ndarray]:
    image_aug = self.iaaseq(image=img)
    return image_aug, ann

  def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                     is_augment: bool, is_normlize: bool,
                     is_training: bool) -> tf.data.Dataset:

    def parser(img_path: tf.Tensor, ann: tf.Tensor):
      img = self.read_img(img_path)
      img, ann = self.resize_img(img, ann)
      if is_augment:
        img, ann = tf.numpy_function(self.iaa_augment_img, [img, ann],
                                     [tf.uint8, tf.int32])
        img.set_shape([self.in_hw[0], self.in_hw[1], 3])
        ann.set_shape([])
      if is_normlize:
        img = self.normlize_img(img)
      if is_training:
        label = tf.one_hot(ann, self.class_num)
      else:
        label = ann
      return img, label

    name_ds = tf.data.Dataset.from_tensor_slices(image_ann_list[0])
    label_ds = tf.data.Dataset.from_tensor_slices(image_ann_list[1])
    if is_training:
      ds = (
          tf.data.Dataset.zip(
              (name_ds, label_ds)).shuffle(1000 * batch_size).repeat().map(
                  parser, -1).batch(batch_size, True).prefetch(-1))

    else:
      ds = (
          tf.data.Dataset.zip(
              (name_ds, label_ds)).map(parser, -1).batch(batch_size, False))

    return ds

  def parser_outputs(self, outputs: tf.Tensor) -> tf.Tensor:
    return tf.argmax(tf.nn.softmax(outputs, -1), -1)

  def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True):

    if is_show:
      plt.imshow((img).astype('uint8'))
      plt.title(self.class_name[ann])
      plt.show()

    return img.astype('uint8')


def imgnet_infer(img_path: Path, infer_model: k.Model, result_path: Path,
                 h: ImgnetHelper):
  print(INFO, f'Load Images from {str(img_path)}')
  if img_path.is_dir():
    img_paths = []
    for suffix in ['bmp', 'jpg', 'JPG', 'jpeg', 'png']:
      img_paths += list(map(str, img_path.glob(f'*.{suffix}')))
  elif img_path.is_file():
    img_paths = [str(img_path)]
  else:
    ValueError(f'{ERROR} img_path `{str(img_path)}` is invalid')

  if result_path is not None:
    print(INFO, f'Load NNcase Results from {str(result_path)}')
    if result_path.is_dir():
      ncc_results: np.ndarray = np.array([
          np.fromfile(
              str(result_path / (Path(img_paths[i]).stem + '.bin')),
              dtype='float32') for i in range(len(img_paths))
      ])
    elif result_path.is_file():
      ncc_results = np.expand_dims(
          np.fromfile(str(result_path), dtype='float32'), 0)  # type:np.ndarray
    else:
      ValueError(f'{ERROR} result_path `{str(result_path)}` is invalid')
  else:
    ncc_results = None

  print(INFO, f'Infer Results')
  orig_hws = []
  det_imgs = []
  for img_path in img_paths:
    img = h.read_img(img_path)
    img, _ = h.resize_img(img, tf.zeros((0)))
    det_img = h.normlize_img(img)
    det_imgs.append(det_img)
  det_imgs = tf.stack(det_imgs)

  outputs = infer_model.predict(det_imgs)
  """ parser batch out """
  results = h.parser_outputs(outputs).numpy()

  if result_path is None:
    """ draw gpu result """
    for img_path, cals in zip(img_paths, results):
      draw_img = h.read_img(img_path)

      h.draw_image(draw_img.numpy(), cals)
  else:
    """ draw gpu result and nncase result """
    ncc_preds = []
    for ncc_result in ncc_results:
      ncc_preds.append(ncc_result)

    ncc_results = h.parser_outputs(ncc_preds).numpy()

    for img_path, cals, ncc_cals in zip(img_paths, results, ncc_results):
      draw_img = h.read_img(img_path)
      gpu_img = h.draw_image(draw_img.numpy(), cals, is_show=False)
      ncc_img = h.draw_image(draw_img.numpy(), ncc_cals, is_show=False)
      fig: plt.Figure = plt.figure(figsize=(8, 3))
      ax1 = plt.subplot(121)  # type:plt.Axes
      ax2 = plt.subplot(122)  # type:plt.Axes
      ax1.imshow(gpu_img)
      ax2.imshow(ncc_img)
      ax1.set_title('GPU ' + h.class_name[cals])
      ax2.set_title('Ncc ' + h.class_name[ncc_cals])
      fig.tight_layout()
      plt.axis('off')
      plt.xticks([])
      plt.yticks([])
      plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          classes: np.ndarray,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if not title:
    if normalize:
      title = 'confusion matrix (normalize)'
    else:
      title = 'confusion matrix'

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  p_calss, r_class, f_class, support_micro = precision_recall_fscore_support(
      y_true, y_pred)

  cm_extend = np.vstack((p_calss, r_class, f_class)) * 100
  # Only use the labels that appear in the data
  classes = classes[unique_labels(y_true, y_pred)]
  yclasses = np.append(classes, np.array(['precision', 'recall', 'F1 score']))

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  fig, ax = plt.subplots(figsize=(8, 8))
  im = ax.imshow(np.vstack((cm, cm_extend)), interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(
      xticks=np.arange(cm.shape[1]),
      yticks=np.arange(cm.shape[0] + cm_extend.shape[0]),
      # ... and label them with the respective list entries
      xticklabels=classes,
      yticklabels=yclasses,
      title=title,
      ylabel='y true',
      xlabel='y pred')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(
          j,
          i,
          format(cm[i, j], fmt),
          ha="center",
          va="center",
          color="white" if cm[i, j] > thresh else "black")

  for i in range(cm_extend.shape[0]):
    for j in range(cm_extend.shape[1]):
      ax.text(
          j,
          i + cm.shape[0],
          format(cm_extend[i, j], '.1f'),
          ha="center",
          va="center",
          color="black",
          fontsize=10)

  fig.tight_layout()
  return fig, cm, cm_extend, yclasses


def imgnet_eval(infer_model: k.Model, h: ImgnetHelper, batch_size: int = 32):
  h.set_dataset(batch_size, False, True, False)
  y_trues = []
  y_preds = []
  for data, y_true in tqdm(h.test_dataset, total=h.test_epoch_step):
    y_pred = infer_model(data, training=False)
    y_preds.append(h.parser_outputs(y_pred).numpy())
    y_trues.append(y_true.numpy())
  y_preds = np.concatenate(y_preds, 0)
  y_trues = np.concatenate(y_trues, 0)

  fig, cm, cm_extend, yclasses = plot_confusion_matrix(
      y_trues, y_preds, np.array(list(h.class_name.values())))

  print('\n'.join([
      f'{s:^20} : {v:.3f}'
      for s, v in zip(yclasses[-3:], np.mean(cm_extend, axis=1))
  ]))
  plt.show()


class ClassifyLoss(LossFunctionWrapper):

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction='auto',
               name='categorical_crossentropy'):
    super().__init__(
        k.losses.categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)
