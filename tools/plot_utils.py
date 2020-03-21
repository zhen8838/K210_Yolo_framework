import numpy as np
import matplotlib.pyplot as plt


def plot_images(datas, nrows=1, ncols=1, figsize=(9, 9), is_show: bool = True):
  fig: plt.Figure = plt.figure(figsize=figsize)
  axs = fig.subplots(nrows, ncols, squeeze=False)
  for i in range(nrows):
    for j in range(ncols):
      axs[i, j].imshow(datas[i, j])
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])
  fig.tight_layout()
  if is_show:
    plt.show()