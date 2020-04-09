import tensorflow as tf
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.imgnet import ImgnetHelper
import matplotlib.pyplot as plt


def test_data_process():
  h = ImgnetHelper('data/mosquitoes_img_ann.npy', 6, [224, 224])

  h.set_dataset(1, True, False, True)

  iters = iter(h.train_dataset)

  for i in range(10):
    imgs, anns = next(iters)
    plt.imshow(imgs[0])
    plt.show()
