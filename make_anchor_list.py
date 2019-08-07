import numpy as np
from tools.utils import Helper, INFO, ERROR, NOTE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
import argparse
import tensorflow as tf


def tf_fake_iou(X: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """ calc the fake iou between x and centroids

    Parameters
    ----------
    X : tf.Tensor
        dataset array, shape = [?,2]
    centroids : tf.Tensor
        centroids,shape = [?,2]

    Returns
    -------
    tf.Tensor
        iou score, shape = [?,1]
    """
    a_maxes = X / 2.
    a_mins = -a_maxes

    b_maxes = centroids / 2.
    b_mins = -b_maxes

    iner_mins = tf.maximum(a_mins, b_mins)
    iner_maxes = tf.minimum(a_maxes, b_maxes)
    iner_wh = tf.maximum(iner_maxes - iner_mins, 0.)
    iner_area = iner_wh[..., 0] * iner_wh[..., 1]

    s1 = X[..., 0] * X[..., 1]
    s2 = centroids[..., 0] * centroids[..., 1]

    return 1 - iner_area / (s1 + s2 - iner_area)


def findClosestCentroids(X: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """ find close centroids

    Parameters
    ----------
    X : tf.Tensor
        dataset array, shape = [?,2]
    centroids : tf.Tensor
        centroids array, shape = [?,2]

    Returns
    -------
    tf.Tensor
        idx, shape = [?,]    
    """
    idx = tf.argmin(tf_fake_iou(X, centroids), axis=1)
    return idx


def computeCentroids(X: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """ use idx calc the new centroids

    Parameters
    ----------
    X : np.ndarray
        shape = [?,2]
    idx : np.ndarray
        shape = [?,]
    k : int
        the centroids num

    Returns
    -------
    np.ndarray
        new centroids
    """
    m, n = np.shape(X)
    centroids = np.zeros((k, n))
    for i in range(k):
        centroids[i, :] = np.mean(X[np.nonzero(idx == i)[0], :], axis=0)
    return centroids


def plotDataPoints(X, idx, K):
    plt.scatter(X[:, 0], X[:, 1], c=idx)


def plotProgresskMeans(X, centroids_history, idx, K, i):
    plotDataPoints(X, idx, K)
    # Plot the centroids as black x's
    for i in range(len(centroids_history) - 1):
        plt.plot(centroids_history[i][:, 0], centroids_history[i][:, 1], 'rx')
        plt.plot(centroids_history[i + 1][:, 0], centroids_history[i + 1][:, 1], 'bx')
        # Plot the history of the centroids with lines
        for j in range(K):
            # matplotlib can't draw line like [x1,y1] to [x2,y2]
            # it have to write like [x1,x2] to [y1,y2] f**k!
            plt.plot(np.r_[centroids_history[i + 1][j, 0], centroids_history[i][j, 0]],
                     np.r_[centroids_history[i + 1][j, 1], centroids_history[i][j, 1]], 'k--')
    # Title
    plt.title('Iteration number {}'.format(i + 1))


def tile_x(x: np.ndarray, k: int):
    # tile the array
    x = x[:, np.newaxis, :]
    x = np.tile(x, (1, k, 1))
    return x


def tile_c(initial_centroids: np.ndarray, m: int):
    c = initial_centroids[np.newaxis, :, :]
    c = np.tile(c, (m, 1, 1))
    return c


def build_kmeans_graph(new_x: np.ndarray, new_c: np.ndarray):
    """ build calc kmeans graph

    Parameters
    ----------
    new_x : np.ndarray
        shape= [?,5,2]
    new_c : np.ndarray
        shape = [?,5,2]

    Returns
    -------
    tuple
    in_x : x placeholder
    in_c : c placeholder
    out_idx : output idx tensor, shape [?,]
    """
    in_x = tf.placeholder(tf.float64, shape=np.shape(new_x), name='in_x')
    in_c = tf.placeholder(tf.float64, shape=np.shape(new_c), name='in_c')
    out_idx = findClosestCentroids(in_x, in_c)

    return in_x, in_c, out_idx


def runkMeans(X: np.ndarray, initial_centroids: np.ndarray, max_iters: int,
              plot_progress=False):
    # init value
    m, _ = X.shape
    k, _ = initial_centroids.shape

    # history list
    centroid_history = []

    # save history
    centroids = initial_centroids.copy()
    centroid_history.append(centroids.copy())

    # build tensorflow graph
    new_x, new_c = tile_x(X, k), tile_c(initial_centroids, m)
    assert new_x.shape == new_c.shape
    in_x, in_c, idx = build_kmeans_graph(new_x, new_c)

    """ run kmeans """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    for i in range(max_iters):
        idx_ = sess.run(idx, feed_dict={in_x: new_x, in_c: new_c})
        new_centrois = computeCentroids(X, idx_, k)
        centroid_history.append(new_centrois.copy())
        new_c = tile_c(new_centrois, m)

    sess.close()
    if plot_progress:
        plt.figure()
        plotProgresskMeans(X, centroid_history, idx_, k, max_iters)
        plt.show()

    return new_centrois, idx_


def main(train_set: str, max_iters: int, in_hw: tuple, out_hw: tuple,
         anchor_num: int, is_random: bool, is_plot: bool, low: list, high: list):
    X = np.load(f'data/{train_set}_img_ann.npy', allow_pickle=True)
    in_wh = np.array(in_hw[::-1])
    low = np.array(low)
    high = np.array(high)
    # NOTE correct boxes
    for i in range(len(X)):
        # X[i, 1], X[i, 2]
        img_wh = X[i, 2][::-1]

        """ calculate the affine transform factor """
        scale = in_wh / img_wh  # NOTE affine tranform sacle is [w,h]
        scale[:] = np.min(scale)
        # NOTE translation is [w offset,h offset]
        translation = ((in_wh - img_wh * scale) / 2).astype(int)

        """ calculate the box transform matrix """
        X[i, 1][:, 1:3] = (X[i, 1][:, 1:3] * img_wh * scale + translation) / in_wh
        X[i, 1][:, 3:5] = (X[i, 1][:, 3:5] * img_wh * scale) / in_wh

    x = np.vstack(X[:, 1])
    x = x[:, 3:]
    layers = len(out_hw) // 2
    if is_random == 'True':
        initial_centroids = np.hstack((np.random.uniform(low[0], high[0], (layers * anchor_num, 1)),
                                       np.random.uniform(low[1], high[1], (layers * anchor_num, 1))))
    else:
        initial_centroids = np.vstack((np.linspace(0.05, 0.3, num=layers * anchor_num), np.linspace(0.05, 0.5, num=layers * anchor_num)))
        initial_centroids = initial_centroids.T
    centroids, idx = runkMeans(x, initial_centroids, 10, is_plot)
    # NOTE : sort by descending , bigger value for layer 0 .
    centroids = np.array(sorted(centroids, key=lambda x: (-x[0])))
    centroids = np.reshape(centroids, (layers, anchor_num, 2))
    for l in range(layers):
        centroids[l] = centroids[l]  # grid_wh[l]  # NOTE centroids是相对于全局的0-1
    if np.any(np.isnan(centroids)):
        print(ERROR, 'Result have NaN value please Rerun!')
    else:
        print(NOTE, f'Now anchors are :\n{centroids}')
        np.save(f'data/{train_set}_anchor.npy', centroids)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('train_set', type=str, help=NOTE + 'this is train dataset name , the output *.npy file will be {train_set}_anchors.list')
    parser.add_argument('--max_iters', type=int, help='kmeans max iters', default=10)
    parser.add_argument('--is_random', type=str, help='wether random generate the center', choices=['True', 'False'], default='True')
    parser.add_argument('--is_plot', type=str, help='wether show the figure', choices=['True', 'False'], default='True')
    parser.add_argument('--in_hw', type=int, help='net work input image size', default=(224, 320), nargs='+')
    parser.add_argument('--out_hw', type=int, help='net work output image size', default=(7, 10, 14, 20), nargs='+')
    parser.add_argument('--low', type=float, help='Lower bound of random anchor, (x,y)', default=(0.0, 0.0), nargs='+')
    parser.add_argument('--high', type=float, help='Upper bound of random anchor, (x,y)', default=(1.0, 1.0), nargs='+')
    parser.add_argument('--anchor_num', type=int, help='single layer anchor nums', default=3)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.train_set, args.max_iters, args.in_hw, args.out_hw, args.anchor_num, args.is_random, args.is_plot, args.low, args.high)
