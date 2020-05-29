import cv2
import numpy as np
from collections import namedtuple
from typing import List, Tuple
import math
import os
import sys
import tensorflow as tf
import scipy.stats as st
from tools.openpose.openopse_agument import CocoPart, CocoPairs, CocoPairsRender, CocoColors
try:
  from tools.openpose.pafprocess import pafprocess
except ModuleNotFoundError as e:
  print(e)
  print('you need to build c++ library for pafprocess. goto file://tools/openpose/pafprocess/README.md ')
  exit(-1)


def _round(v):
  return int(round(v))


def _include_part(part_list, part_idx):
  for part in part_list:
    if part_idx == part.part_idx:
      return True, part
  return False, None


class Human:
  """
  body_parts: list of BodyPart
  """
  __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

  def __init__(self, pairs):
    self.pairs = []
    self.uidx_list = set()
    self.body_parts = {}
    for pair in pairs:
      self.add_pair(pair)
    self.score = 0.0

  @staticmethod
  def _get_uidx(part_idx, idx):
    return '%d-%d' % (part_idx, idx)

  def add_pair(self, pair):
    self.pairs.append(pair)
    self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                               pair.part_idx1,
                                               pair.coord1[0], pair.coord1[1], pair.score)
    self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                               pair.part_idx2,
                                               pair.coord2[0], pair.coord2[1], pair.score)
    self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
    self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

  def is_connected(self, other):
    return len(self.uidx_list & other.uidx_list) > 0

  def merge(self, other):
    for pair in other.pairs:
      self.add_pair(pair)

  def part_count(self):
    return len(self.body_parts.keys())

  def get_max_score(self):
    return max([x.score for _, x in self.body_parts.items()])

  def get_face_box(self, img_w, img_h, mode=0):
    """
    Get Face box compared to img size (w, h)
    :param img_w:
    :param img_h:
    :param mode:
    :return:
    """
    # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
    _NOSE = CocoPart.Nose
    _NECK = CocoPart.Neck
    _REye = CocoPart.REye
    _LEye = CocoPart.LEye
    _REar = CocoPart.REar
    _LEar = CocoPart.LEar

    _THRESHOLD_PART_CONFIDENCE = 0.2
    parts = [part for idx, part in self.body_parts.items() if part.score >
             _THRESHOLD_PART_CONFIDENCE]

    is_nose, part_nose = _include_part(parts, _NOSE)
    if not is_nose:
      return None

    size = 0
    is_neck, part_neck = _include_part(parts, _NECK)
    if is_neck:
      size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

    is_reye, part_reye = _include_part(parts, _REye)
    is_leye, part_leye = _include_part(parts, _LEye)
    if is_reye and is_leye:
      size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
      size = max(size,
                 img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

    if mode == 1:
      if not is_reye and not is_leye:
        return None

    is_rear, part_rear = _include_part(parts, _REar)
    is_lear, part_lear = _include_part(parts, _LEar)
    if is_rear and is_lear:
      size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

    if size <= 0:
      return None

    if not is_reye and is_leye:
      x = part_nose.x * img_w - (size // 3 * 2)
    elif is_reye and not is_leye:
      x = part_nose.x * img_w - (size // 3)
    else:  # is_reye and is_leye:
      x = part_nose.x * img_w - size // 2

    x2 = x + size
    if mode == 0:
      y = part_nose.y * img_h - size // 3
    else:
      y = part_nose.y * img_h - _round(size / 2 * 1.2)
    y2 = y + size

    # fit into the image frame
    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w - x, x2 - x) + x
    y2 = min(img_h - y, y2 - y) + y

    if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
      return None
    if mode == 0:
      return {"x": _round((x + x2) / 2),
              "y": _round((y + y2) / 2),
              "w": _round(x2 - x),
              "h": _round(y2 - y)}
    else:
      return {"x": _round(x),
              "y": _round(y),
              "w": _round(x2 - x),
              "h": _round(y2 - y)}

  def get_upper_body_box(self, img_w, img_h):
    """
    Get Upper body box compared to img size (w, h)
    :param img_w:
    :param img_h:
    :return:
    """

    if not (img_w > 0 and img_h > 0):
      raise Exception("img size should be positive")

    _NOSE = CocoPart.Nose
    _NECK = CocoPart.Neck
    _RSHOULDER = CocoPart.RShoulder
    _LSHOULDER = CocoPart.LShoulder
    _THRESHOLD_PART_CONFIDENCE = 0.3
    parts = [part for idx, part in self.body_parts.items() if part.score >
             _THRESHOLD_PART_CONFIDENCE]
    part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                   part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

    if len(part_coords) < 5:
      return None

    # Initial Bounding Box
    x = min([part[0] for part in part_coords])
    y = min([part[1] for part in part_coords])
    x2 = max([part[0] for part in part_coords])
    y2 = max([part[1] for part in part_coords])

    # # ------ Adjust heuristically +
    # if face points are detcted, adjust y value

    is_nose, part_nose = _include_part(parts, _NOSE)
    is_neck, part_neck = _include_part(parts, _NECK)
    torso_height = 0
    if is_nose and is_neck:
      y -= (part_neck.y * img_h - y) * 0.8
      torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
    #
    # # by using shoulder position, adjust width
    is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
    is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
    if is_rshoulder and is_lshoulder:
      half_w = x2 - x
      dx = half_w * 0.15
      x -= dx
      x2 += dx
    elif is_neck:
      if is_lshoulder and not is_rshoulder:
        half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
        x = min(part_neck.x * img_w - half_w, x)
        x2 = max(part_neck.x * img_w + half_w, x2)
      elif not is_lshoulder and is_rshoulder:
        half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
        x = min(part_neck.x * img_w - half_w, x)
        x2 = max(part_neck.x * img_w + half_w, x2)

    # ------ Adjust heuristically -

    # fit into the image frame
    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w - x, x2 - x) + x
    y2 = min(img_h - y, y2 - y) + y

    if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
      return None
    return {"x": _round((x + x2) / 2),
            "y": _round((y + y2) / 2),
            "w": _round(x2 - x),
            "h": _round(y2 - y)}

  def __str__(self):
    return ' \n'.join([str(x) for x in self.body_parts.values()])

  def __repr__(self):
    return self.__str__()


class BodyPart:
  """
  part_idx : part index(eg. 0 for nose)
  x, y: coordinate of body part
  score : confidence score
  """
  __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

  def __init__(self, uidx, part_idx, x, y, score):
    self.uidx = uidx
    self.part_idx = part_idx
    self.x, self.y = x, y
    self.score = score

  def get_part_name(self):
    return CocoPart.name_list[self.part_idx]

  def __str__(self):
    return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

  def __repr__(self):
    return self.__str__()


class Peak:
  __slots__ = ('x', 'y', 'score', 'id')

  def __init__(self, x, y, score, id):
    self.x = x
    self.y = y
    self.score = score
    self.id = id

  def __str__(self):
    return f'x:{self.x:d} y:{self.y:d} score:{self.score:.2f} id:{self.id}\n'

  def __repr__(self):
    return self.__str__()


class VectorXY:
  __slots__ = ('x', 'y')

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __str__(self):
    return f'x:{self.x:.2f} y:{self.y:.2f}'

  def __repr__(self):
    return self.__str__()


class ConnectionCandidate:
  __slots__ = ('idx1', 'idx2', 'score', 'etc')

  def __init__(self, idx1, idx2, score, etc):
    super().__init__()
    self.idx1 = idx1
    self.idx2 = idx2
    self.score = score
    self.etc = etc

  def __str__(self):
    return f'CC {self.idx1}->{self.idx2} {self.score:.2f}'

  def __repr__(self):
    return self.__str__()


class Connection:
  __slots__ = ('cid1', 'cid2', 'score', 'peak_id1', 'peak_id2')

  def __init__(self, cid1, cid2, score, peak_id1, peak_id2):
    super().__init__()
    self.cid1 = cid1
    self.cid2 = cid2
    self.score = score
    self.peak_id1 = peak_id1
    self.peak_id2 = peak_id2

  def __str__(self):
    return f'C {self.cid1}-{self.cid2} {self.score:.2f} {self.peak_id1}-{self.peak_id2}'

  def __repr__(self):
    return self.__str__()


class Pafprocess():

  def __init__(self):
    super().__init__()
    self.THRESH_HEAT = 0.05
    self.NUM_PART = 18
    self.COCOPAIRS_SIZE = 19
    self.STEP_PAF = 10
    self.THRESH_VECTOR_SCORE = 0.05
    self.THRESH_VECTOR_CNT1 = 6
    self.THRESH_PART_CNT = 4
    self.THRESH_HUMAN_SCORE = 0.3
    self.COCOPAIRS_NET = [
        [12, 13], [20, 21], [14, 15], [16, 17], [22, 23],
        [24, 25], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [10, 11], [28, 29], [30, 31], [34, 35], [32, 33],
        [36, 37], [18, 19], [26, 27]
    ]

    self.COCOPAIRS = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
        [12, 13], [1, 0], [0, 14], [14, 16], [0, 15],
        [15, 17], [2, 16], [5, 17]
    ]
    self.subset: List[List[float]] = [[]]
    self.peak_infos_line: List[Peak] = []

  def get_paf_vectors(self, pafmap: np.ndarray, ch_id1: int, ch_id2: int, peak1: Peak, peak2: Peak) -> List[VectorXY]:
    def roundpaf(v): return (int)(v + 0.5)

    paf_vectors: List[VectorXY] = []
    STEP_X = (peak2.x - peak1.x) / self.STEP_PAF
    STEP_Y = (peak2.y - peak1.y) / self.STEP_PAF
    for i in range(self.STEP_PAF):
      location_x = roundpaf(peak1.x + i * STEP_X)
      location_y = roundpaf(peak1.y + i * STEP_Y)

      v = VectorXY(x=pafmap[location_y, location_x, ch_id1],
                   y=pafmap[location_y, location_x, ch_id2])
      paf_vectors.append(v)
    return paf_vectors

  def process(self, peaks, heat_mat, paf_mat):
    peak_infos: List[List[Peak]] = [[] for i in range(self.NUM_PART)]   # Peak
    peak_cnt = 0

    # 寻找峰值大于阈值的,对于大于阈值的构建peak对象 NOTE 一定得按part的顺序索引
    part_ids, ys, xs = np.where(np.transpose(peaks, [2, 0, 1]) > self.THRESH_HEAT)
    for y, x, part_id in zip(ys, xs, part_ids):
      if part_id < self.NUM_PART:
        info = Peak(x=x, y=y, score=heat_mat[y, x, part_id], id=peak_cnt)
        peak_cnt += 1
        peak_infos[part_id].append(info)

    self.peak_infos_line.clear()
    self.peak_infos_line.extend(sum(peak_infos, []))
    # 开始连接
    connection_all: List[List[Connection]] = [[] for i in range(self.COCOPAIRS_SIZE)]   # Connection
    for pair_id in range(self.COCOPAIRS_SIZE):
      candidates: List[ConnectionCandidate] = []  # ConnectionCandidate
      peak_a_list = peak_infos[self.COCOPAIRS[pair_id][0]]
      peak_b_list = peak_infos[self.COCOPAIRS[pair_id][1]]
      if len(peak_a_list) == 0 or len(peak_b_list) == 0:
        continue
      for peak_a_id, peak_a in enumerate(peak_a_list):
        for peak_b_id, peak_b in enumerate(peak_b_list):
          vec = VectorXY(x=peak_b.x - peak_a.x, y=peak_b.y - peak_a.y)
          norm = np.sqrt(vec.x * vec.x + vec.y * vec.y)
          if (norm < 1e-12):
            continue
          vec.x = vec.x / norm
          vec.y = vec.y / norm
          paf_vecs = self.get_paf_vectors(paf_mat, self.COCOPAIRS_NET[pair_id][0],
                                          self.COCOPAIRS_NET[pair_id][1],
                                          peak_a, peak_b)
          # 标准1 分数阈值计数
          criterion1 = 0
          scores = 0.
          for i in range(self.STEP_PAF):
            score = vec.x * paf_vecs[i].x + vec.y * paf_vecs[i].y
            scores += score

            if score > self.THRESH_VECTOR_SCORE:
              criterion1 += 1

          criterion2 = scores / self.STEP_PAF + min(0.0, (0.5 * heat_mat.shape[0] / norm) - 1.0)
          # 标准2
          if criterion1 > self.THRESH_VECTOR_CNT1 and criterion2 > 0:
            candidate = ConnectionCandidate(
                idx1=peak_a_id, idx2=peak_b_id,
                score=criterion2, etc=criterion2 + peak_a.score + peak_b.score)
            candidates.append(candidate)

        conns = connection_all[pair_id]
        sorted(candidates, key=lambda x: x.score, reverse=True)

        for c_id in range(len(candidates)):
          candidate = candidates[c_id]
          assigned = False
          for conn_id in range(len(conns)):
            if (conns[conn_id].peak_id1 == candidate.idx1):
              # already assigned
              assigned = True
              break
            if assigned:
              break
            if conns[conn_id].peak_id2 == candidate.idx2:
              # already assigned
              assigned = True
              break
            if assigned:
              break
          if assigned:
            continue

          conn = Connection(
              peak_id1=candidate.idx1, peak_id2=candidate.idx2,
              score=candidate.score,
              cid1=peak_a_list[candidate.idx1].id,
              cid2=peak_b_list[candidate.idx2].id)

          conns.append(conn)

    self.subset.clear()
    for pair_id in range(self.COCOPAIRS_SIZE):
      conns = connection_all[pair_id]
      part_id1 = self.COCOPAIRS[pair_id][0]
      part_id2 = self.COCOPAIRS[pair_id][1]

      for conn_id in range(len(conns)):
        found = 0
        subset_idx1 = 0
        subset_idx2 = 0
        for subset_id in range(len(self.subset)):
          if (self.subset[subset_id][part_id1] == conns[conn_id].cid1 or
                  self.subset[subset_id][part_id2] == conns[conn_id].cid2):
            if found == 0:
              subset_idx1 = subset_id
            if found == 1:
              subset_idx2 = subset_id
            found += 1
        if found == 1:
          if self.subset[subset_idx1][part_id2] != conns[conn_id].cid2:
            self.subset[subset_idx1][part_id2] = conns[conn_id].cid2
            self.subset[subset_idx1][19] += 1
            self.subset[subset_idx1][18] += (self.peak_infos_line[conns[conn_id].cid2].score +
                                             conns[conn_id].score)
        elif (found == 2):
          membership = 0
          for subset_id in range(18):
            if (self.subset[subset_idx1][subset_id] > 0 and self.subset[subset_idx2][subset_id] > 0):
              membership = 2
          if membership == 0:
            for subset_id in range(18):
              self.subset[subset_idx1][subset_id] += (self.subset[subset_idx2][subset_id] + 1)
            self.subset[subset_idx1][19] += self.subset[subset_idx2][19]
            self.subset[subset_idx1][18] += self.subset[subset_idx2][18]
            self.subset[subset_idx1][18] += conns[conn_id].score
            self.subset.pop(subset_idx2)
          else:
            self.subset[subset_idx1][part_id2] = conns[conn_id].cid2
            self.subset[subset_idx1][19] += 1
            self.subset[subset_idx1][18] += (self.peak_infos_line[conns[conn_id].cid2].score +
                                             conns[conn_id].score)
        elif (found == 0 and pair_id < 17):
          row = np.ones((20)) * -1
          row[part_id1] = conns[conn_id].cid1
          row[part_id2] = conns[conn_id].cid2
          row[19] = 2
          row[18] = (self.peak_infos_line[conns[conn_id].cid1].score +
                     self.peak_infos_line[conns[conn_id].cid2].score +
                     conns[conn_id].score)
          self.subset.append(row)

    for i in range(len(self.subset))[::-1]:
      if (self.subset[i][19] < self.THRESH_PART_CNT or
              (self.subset[i][18] / self.subset[i][19]) < self.THRESH_HUMAN_SCORE):
        self.subset.pop(i)

  def get_num_humans(self):
    return len(self.subset)

  def get_part_cid(self, human_id, part_id):
    return self.subset[human_id][part_id]

  def get_score(self, human_id):
    return self.subset[human_id][18] / self.subset[human_id][19]

  def get_part_x(self, cid):
    return self.peak_infos_line[cid].x

  def get_part_y(self, cid):
    return self.peak_infos_line[cid].y

  def get_part_score(self, cid):
    return self.peak_infos_line[cid].score


def draw_humans(npimg, humans: List[Human], imgcopy=False):
  if imgcopy:
    npimg = np.copy(npimg)
  image_h, image_w = npimg.shape[:2]
  centers = {}
  for human in humans:
    # draw point
    for i in range(CocoPart.Background):
      if i not in human.body_parts.keys():
        continue

      body_part = human.body_parts[i]
      center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
      centers[i] = center
      cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

    # draw line
    for pair_order, pair in enumerate(CocoPairsRender):
      if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
        continue
      cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

  return npimg


def estimate_paf(peaks, heat_mat, paf_mat):
  # todo 我复现的python版后处理还是有问题,暂时用他的版本
  # pafprocess = Pafprocess()
  # pafprocess.process(peaks, heat_mat, paf_mat)
  pafprocess.process_paf(peaks, heat_mat, paf_mat)
  humans = []
  for human_id in range(pafprocess.get_num_humans()):
    human = Human([])
    is_added = False

    for part_idx in range(18):
      c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
      if c_idx < 0:
        continue

      is_added = True
      human.body_parts[part_idx] = BodyPart(
          '%d-%d' % (human_id, part_idx), part_idx,
          float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
          float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
          pafprocess.get_part_score(c_idx)
      )

    if is_added:
      score = pafprocess.get_score(human_id)
      human.score = score
      humans.append(human)
  return humans


def layer(op):
  def layer_decorated(self, *args, **kwargs):
    # Automatically set a name if not provided.
    name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
    # Figure out the layer inputs.
    if len(self.terminals) == 0:
      raise RuntimeError('No input variables found for layer %s.' % name)
    elif len(self.terminals) == 1:
      layer_input = self.terminals[0]
    else:
      layer_input = list(self.terminals)
    # Perform the operation and get the output.
    layer_output = op(self, layer_input, *args, **kwargs)
    # Add to layer LUT.
    self.layers[name] = layer_output
    # This output is now the input for the next layer.
    self.feed(layer_output)
    # Return self for chained calls.
    return self

  return layer_decorated


class Smoother(object):
  def __init__(self, inputs, filter_size, sigma, heat_map_size=0):
    self.inputs = inputs
    self.terminals = []
    self.layers = dict(inputs)
    self.filter_size = filter_size
    self.sigma = sigma
    self.heat_map_size = heat_map_size
    self.setup()

  def setup(self):
    self.feed('data').conv(name='smoothing')

  def get_unique_name(self, prefix):
    ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
    return '%s_%d' % (prefix, ident)

  def feed(self, *args):
    assert len(args) != 0
    self.terminals = []
    for fed_layer in args:
      if isinstance(fed_layer, str):
        try:
          fed_layer = self.layers[fed_layer]
        except KeyError:
          raise KeyError('Unknown layer name fed: %s' % fed_layer)
      self.terminals.append(fed_layer)
    return self

  def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter

  def make_gauss_var(self, name, size, sigma, c_i):
    # with tf.device("/cpu:0"):
    kernel = self.gauss_kernel(size, sigma, c_i)
    var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
    return var

  def get_output(self):
    '''Returns the smoother output.'''
    return self.terminals[-1]

  @layer
  def conv(self,
           input,
           name,
           padding='SAME'):
    # Get the number of channels in the input
    if self.heat_map_size != 0:
      c_i = self.heat_map_size
    else:
      c_i = input.get_shape().as_list()[3]
    # Convolution for a given input and kernel
    def convolve(i, k): return tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding=padding)
    kernel = self.make_gauss_var('gauss_weight', self.filter_size, self.sigma, c_i)
    output = convolve(input, kernel)
    return output

