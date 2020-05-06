import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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


def build_ball(ax):
  xlm = ax.get_xlim3d()
  ylm = ax.get_ylim3d()
  zlm = ax.get_zlim3d()
  ax.set_xlim3d(-.82, 0.82)
  ax.set_ylim3d(-.82, 0.82)
  ax.set_zlim3d(-.82, 0.82)
  # First remove fill
  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = False

  # Now set color to white (or whatever is "invisible")
  ax.xaxis.pane.set_edgecolor('w')
  ax.yaxis.pane.set_edgecolor('w')
  ax.zaxis.pane.set_edgecolor('w')

  # Bonus: To get rid of the grid as well:
  ax.grid(False)

  ax.set_xticks([-0.5, 0, 0.5])
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_zticks([-1, -0.5, 0, 0.5, 1])

  u = np.linspace(0, 2 * np.pi, 15)
  v = np.linspace(0, np.pi, 20)
  x = 1 * np.outer(np.cos(u), np.sin(v))
  y = 1 * np.outer(np.sin(u), np.sin(v))
  z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
  ax.plot_wireframe(
      x, y, z, colors='dimgray', alpha=0.6, linestyles='-', linewidths=1)


def plot_emmbeding(datas: np.ndarray, is_show: bool = True):
  """plot emmbeding in 3D

  Args:
      datas (np.ndarray): shape [n,3] 
      is_show (bool, optional): Defaults to True.
  """
  fig = plt.figure(figsize=[4, 4])
  ax: axes3d.Axes3D = fig.add_subplot(1, 1, 1, projection='3d')  # type: Axes3D
  ax.view_init(elev=25., azim=120.)
  build_ball(ax)
  ax.scatter(datas[:, 0], datas[:, 1], datas[:, 2])
  if is_show:
    plt.tight_layout()
    plt.show()
