import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imgaug.augmentables import Keypoint, KeypointsOnImage
import sys
import os
sys.path.insert(0, os.getcwd())

from tools.pfld_v2 import PFLDV2Helper
from tools.pfld import calculate_pitch_yaw_roll


def crop_roi_image(img, landmark, scale: float = 1.2):
  xy = np.min(landmark, axis=0).astype(np.int32)
  zz = np.max(landmark, axis=0).astype(np.int32)
  wh = zz - xy + 1
  center = (xy + wh / 2).astype(np.int32)
  boxsize = int(np.max(wh) * scale)
  xy = center - boxsize // 2
  x1, y1 = xy
  x2, y2 = xy + boxsize
  height, width = img.shape[:2]
  dx = max(0, -x1)
  dy = max(0, -y1)
  x1 = max(0, x1)
  y1 = max(0, y1)

  edx = max(0, x2 - width)
  edy = max(0, y2 - height)
  x2 = min(width, x2)
  y2 = min(height, y2)

  imgT = img[y1:y2, x1:x2]
  if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
    print(dy, edy, dx, edx)
    print(imgT.shape)
    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
    print(imgT.shape)
    # tf.image.pad_to_bounding_box

  return imgT, landmark - xy


def test_dataload(se):
  h = PFLDV2Helper('data/pfld_68_img_ann_list.npy', [112, 112], 68, 6)

  ds = tf.data.Dataset.from_tensor_slices(h.train_list).shuffle(1000)
  iters = iter(ds)
  for i in range(200):
    path, im_hw, landmark, attr = next(iters)
    img = tf.image.decode_image(tf.io.read_file(path), channels=3)
    # landmark = landmark.numpy()
    # crop image
    # nimg, nladmark = crop_roi_image(img, landmark, scale=1.2)
    nimg, nladmark = h.crop_img(img, im_hw, landmark, scale=1.2)
    nimg, nladmark = nimg.numpy(), nladmark.numpy()
    # nimg, nladmark = img, landmark

    # augment image
    nimg, nladmark = h.iaaseq(image=nimg, keypoints=nladmark[None, ...])
    nladmark = nladmark[0]

    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    pitch, yaw, roll = calculate_pitch_yaw_roll(nladmark[TRACKED_POINTS])

    # resize image

    # show image
    for (x, y) in nladmark.astype(np.int32):
      cv2.circle(nimg, (x, y), 3, (255, 0, 0), 1)
    plt.imshow(nimg)
    plt.show()
    print(path)


def test_PFLDV2_train_dataset():
  h = PFLDV2Helper('data/pfld_68_img_ann_list.npy', [112, 112], 68, 6)
  # h.set_dataset(4, False, False, False)
  # ds = h.test_dataset
  h.set_dataset(4, True, False, True)
  ds = h.train_dataset

  iters = iter(ds)

  for j in range(20):
    imgs, labels = next(iters)
    for i in range(4):
      label = labels[i].numpy()
      img = imgs[i].numpy()
      h.draw_image(img.astype(np.uint8), label)


def test_PFLDV2_test_dataset():
  h = PFLDV2Helper('data/pfld_68_img_ann_list.npy', [112, 112], 68, 6)
  h.set_dataset(4, False, False, False)
  ds = h.test_dataset

  iters = iter(ds)

  for j in range(20):
    imgs, labels = next(iters)
    for i in range(4):
      label = labels[i].numpy()
      img = imgs[i].numpy()
      h.draw_image(img.astype(np.uint8), label)


def test_PFLDV2_train_attr_weight():
  h = PFLDV2Helper('data/pfld_68_img_ann_list.npy', [112, 112], 68, 6)
  h.set_dataset(96, True, False, True)
  ds = h.train_dataset

  iters = iter(ds)

  for j in range(20):
    imgs, labels = next(iters)
    true_landmark, true_a_w, true_eular = tf.split(labels, [h.landmark_num * 2, 1, 3], 1)
    print(true_a_w)

  # attr = tf.zeros((96, 6))
  # attr = tf.one_hot(tf.random.uniform((96,), 0, 6, dtype=tf.int32), 6)
  # batch_size = 96
  # attribute_num = 6
  # mat_ratio = tf.reduce_mean(attr, axis=0, keepdims=True)
  # mat_ratio = tf.where(mat_ratio > 0, 1. / mat_ratio,
  #                      tf.ones([1, attribute_num]) * batch_size)
  # attribute_weight = tf.matmul(attr, mat_ratio, transpose_b=True)  # [n,1]
  # attribute_weight = tf.where(
  #     attribute_weight == 0, tf.ones_like(attribute_weight), attribute_weight)


if __name__ == "__main__":
  # test_PFLDV2_train_dataset()
  # test_PFLDV2_test_dataset()
  # test_PFLDV2_train_attr_weight()
  pass
