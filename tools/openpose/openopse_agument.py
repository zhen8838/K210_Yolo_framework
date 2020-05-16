import numpy as np
import cv2
import math


class ImageMeta(object):
  def __init__(self, img, joint_list, in_hw, coco_parts, coco_vecs, sigma):
    super().__init__()
    self.joint_list: np.ndarray = joint_list
    self.joint_list_mask: np.ndarray = np.logical_not(
        np.all(joint_list == -1000, -1, keepdims=True))
    self.img: np.ndarray = img
    self.height, self.width = img.shape[:2]
    self.coco_parts: np.ndarray = coco_parts
    self.coco_vecs: np.ndarray = coco_vecs
    self.network_h = in_hw[0]
    self.network_w = in_hw[1]
    self.sigma = sigma

  def get_heatmap(self, target_hw):
    heatmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.float32)

    for joints in self.joint_list:
      for idx, point in enumerate(joints):
        if point[0] < 0 or point[1] < 0:
          continue
        ImageMeta.put_heatmap(heatmap, idx, point, self.sigma)

    heatmap = heatmap.transpose((1, 2, 0))

    # background
    heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

    if target_hw:
      heatmap = cv2.resize(heatmap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

    return heatmap

  def get_heatmap_v(self, target_hw):
    # NOTE channel first will be faster
    heatmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.float32)
    for (joints, masks) in zip(self.joint_list, self.joint_list_mask):
      for idx, (point, mask) in enumerate(zip(joints, masks)):
        if mask:
          th = 4.6052
          delta = np.sqrt(th * 2)
          p0 = np.maximum(0., point - delta * self.sigma).astype('int32')
          p1 = np.minimum([self.width, self.height], point + delta * self.sigma).astype('int32')

          x = np.arange(p0[0], p1[0])
          y = np.arange(p0[1], p1[1])

          xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

          exp = (((xv - point[0]) ** 2 + (yv - point[1]) ** 2) /
                 (2.0 * self.sigma * self.sigma))

          boolmask = exp < th
          yidx = yv[boolmask]
          xidx = xv[boolmask]
          exp_valid = exp[boolmask]

          heatmap[idx, yidx, xidx] = np.minimum(
              np.maximum(heatmap[idx, yidx, xidx],
                         np.exp(-exp_valid)), 1.0)

    heatmap = heatmap.transpose((1, 2, 0))

    # background
    heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

    if target_hw:
      heatmap = cv2.resize(heatmap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

    return heatmap

  @staticmethod
  def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x, center_y = center  # point
    _, height, width = heatmap.shape[:3]  # 热图大小

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    cnt = 0
    uncnt = 0
    for y in range(y0, y1):
      for x in range(x0, x1):
        d = (x - center_x) ** 2 + (y - center_y) ** 2
        exp = d / 2.0 / sigma / sigma
        # NOTE 如果这个点不是靠近图像边沿,exp就是th的两倍
        if exp > th:
          continue
        heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
        heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

  def get_vectormap_v(self, target_hw):
    vectormap = np.zeros((self.coco_parts * 2, self.height, self.width), dtype=np.float32)
    countmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.int16)
    threshold = 8.
    for joints, masks in zip(self.joint_list, self.joint_list_mask):
      for idx, jidx in enumerate(self.coco_vecs):
        if np.alltrue(masks[jidx]):
          center_from, center_to = joints[jidx]
          # put_vectormap
          vector = center_to - center_from
          p0 = np.maximum(0,
                          (np.minimum(center_from, center_to) - threshold).astype(np.int32))
          p1 = np.minimum([self.width, self.height],
                          (np.maximum(center_from, center_to) + threshold).astype(np.int32))

          norm = np.sqrt(np.sum(np.square(vector), axis=-1))
          if norm == 0:
            continue
          vector = vector / norm
          # p --> x,y
          x = np.arange(p0[0], p1[0])
          y = np.arange(p0[1], p1[1])
          xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
          bec_x = xv - center_from[0]
          bec_y = yv - center_from[1]
          dist = np.abs(bec_x * vector[1] - bec_y * vector[0])
          boolmask = dist <= threshold
          yidx = yv[boolmask]
          xidx = xv[boolmask]
          countmap[idx, yidx, xidx] += 1
          vectormap[idx * 2 + 0, yidx, xidx] = vector[0]
          vectormap[idx * 2 + 1, yidx, xidx] = vector[1]

    countmap = np.repeat(countmap, 2, axis=0)
    boolmask = (countmap > 0)
    vectormap[boolmask] = vectormap[boolmask] / countmap[boolmask]
    vectormap = vectormap.transpose((1, 2, 0))

    if target_hw:
      vectormap = cv2.resize(vectormap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

    return vectormap

  def get_vectormap(self, target_hw):
    vectormap = np.zeros((self.coco_parts * 2, self.height, self.width), dtype=np.float32)
    countmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.int16)
    for joints in self.joint_list:
      for plane_idx, (j_idx1, j_idx2) in enumerate(self.coco_vecs):
        # j_idx1 -= 1
        # j_idx2 -= 1

        center_from = joints[j_idx1]
        center_to = joints[j_idx2]

        if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
          continue

        ImageMeta.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

    vectormap = vectormap.transpose((1, 2, 0))
    nonzeros = np.nonzero(countmap)
    for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
      if countmap[p][y][x] <= 0:
        continue
      vectormap[y][x][p * 2 + 0] /= countmap[p][y][x]
      vectormap[y][x][p * 2 + 1] /= countmap[p][y][x]

    if target_hw:
      vectormap = cv2.resize(vectormap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

    return vectormap

  @staticmethod
  def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
    _, height, width = vectormap.shape[:3]

    vec_x = center_to[0] - center_from[0]
    vec_y = center_to[1] - center_from[1]

    min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
    min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

    max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
    max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm == 0:
      return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
      for x in range(min_x, max_x):
        bec_x = x - center_from[0]
        bec_y = y - center_from[1]
        dist = abs(bec_x * vec_y - bec_y * vec_x)

        if dist > threshold:
          continue

        countmap[plane_idx][y][x] += 1

        vectormap[plane_idx * 2 + 0][y][x] = vec_x
        vectormap[plane_idx * 2 + 1][y][x] = vec_y


class CocoPart(object):
  Nose = 0
  Neck = 1
  RShoulder = 2
  RElbow = 3
  RWrist = 4
  LShoulder = 5
  LElbow = 6
  LWrist = 7
  RHip = 8
  RKnee = 9
  RAnkle = 10
  LHip = 11
  LKnee = 12
  LAnkle = 13
  REye = 14
  LEye = 15
  REar = 16
  LEar = 17
  Background = 18
  flip_list = np.array([
      Nose, Neck, LShoulder, LElbow, LWrist,
      RShoulder, RElbow, RWrist,
      LHip, LKnee, LAnkle, RHip, RKnee, RAnkle,
      LEye, REye, LEar, REar, Background], dtype=np.int32)


def pose_random_scale(meta: ImageMeta):
  scalew = np.random.uniform(0.8, 1.2)
  scaleh = np.random.uniform(0.8, 1.2)
  neww = int(meta.width * scalew)
  newh = int(meta.height * scaleh)
  dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

  # adjust meta data
  # adjust_joint_list = []
  # for joint in meta.joint_list:
  #   adjust_joint = []
  #   for point in joint:
  #     if point[0] < -100 or point[1] < -100:
  #       adjust_joint.append((-1000, -1000))
  #       continue
  #     adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
  #   adjust_joint_list.append(adjust_joint)
  # meta.joint_list = adjust_joint_list

  meta.joint_list = meta.joint_list * np.array([scalew, scaleh], dtype=np.float32) + 0.5
  meta.width, meta.height = neww, newh
  meta.img = dst
  return meta


def pose_resize_shortestedge_fixed(meta: ImageMeta):
  ratio_w = meta.network_w / meta.width
  ratio_h = meta.network_h / meta.height
  ratio = max(ratio_w, ratio_h)
  return pose_resize_shortestedge(meta, int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5)))


def pose_resize_shortestedge_random(meta: ImageMeta):
  ratio_w = meta.network_w / meta.width
  ratio_h = meta.network_h / meta.height
  ratio = min(ratio_w, ratio_h)
  target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
  target_size = int(target_size * np.random.uniform(0.95, 1.6))
  # target_size = int(min(meta.network_w, meta.network_h) * random.uniform(0.7, 1.5))
  return pose_resize_shortestedge(meta, target_size)


def pose_resize_shortestedge(meta, target_size):
  img = meta.img

  # adjust image
  scale = target_size / min(meta.height, meta.width)
  if meta.height < meta.width:
    newh, neww = target_size, int(scale * meta.width + 0.5)
  else:
    newh, neww = int(scale * meta.height + 0.5), target_size

  dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

  pw = ph = 0
  if neww < meta.network_w or newh < meta.network_h:
    pw = max(0, (meta.network_w - neww) // 2)
    ph = max(0, (meta.network_h - newh) // 2)
    mw = (meta.network_w - neww) % 2
    mh = (meta.network_h - newh) % 2
    color = np.random.randint(0, 255)
    dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw,
                             cv2.BORDER_CONSTANT, value=(color, 0, 0))

  # adjust meta data
  # adjust_joint_list = []
  # for joint in meta.joint_list:
  #   adjust_joint = []
  #   for point in joint:
  #     if point[0] < -100 or point[1] < -100:
  #       adjust_joint.append((-1000, -1000))
  #       continue
  #     adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
  #   adjust_joint_list.append(adjust_joint)

  # meta.joint_list = adjust_joint_list
  meta.joint_list = meta.joint_list * scale + 0.5 + np.array([pw, ph], dtype=np.float32)
  meta.width, meta.height = neww + pw * 2, newh + ph * 2
  meta.img = dst
  return meta


def pose_crop_center(meta: ImageMeta):
  target_size = (meta.network_w, meta.network_h)
  x = (meta.width - target_size[0]) // 2 if meta.width > target_size[0] else 0
  y = (meta.height - target_size[1]) // 2 if meta.height > target_size[1] else 0

  return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop_random(meta: ImageMeta):
  target_size = (meta.network_w, meta.network_h)

  for _ in range(50):
    x = np.random.randint(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
    y = np.random.randint(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0

    # check whether any face is inside the box to generate a reasonably-balanced datasets
    for joint in meta.joint_list:
      if x <= joint[CocoPart.Nose][0] < x + target_size[0] and y <= joint[CocoPart.Nose][1] < y + target_size[1]:
        break

  return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop(meta, x, y, w, h):
  # adjust image
  target_size = (w, h)

  img = meta.img
  resized = img[y:y + target_size[1], x:x + target_size[0], :]

  # adjust meta data
  # adjust_joint_list = []
  # for joint in meta.joint_list:
  #   adjust_joint = []
  #   for point in joint:
  #     if point[0] < -100 or point[1] < -100:
  #       adjust_joint.append((-1000, -1000))
  #       continue
  #     new_x, new_y = point[0] - x, point[1] - y
  #     adjust_joint.append((new_x, new_y))
  #   adjust_joint_list.append(adjust_joint)

  meta.joint_list = meta.joint_list - np.array([x, y], dtype=np.float32)
  meta.width, meta.height = target_size
  meta.img = resized
  return meta


def pose_flip(meta: ImageMeta):
  r = np.random.uniform(0, 1.0)
  if r > 0.5:
    return meta
  img = meta.img
  img = cv2.flip(img, 1)

  # flip meta

  # adjust_joint_list = []
  # for joint in meta.joint_list:
  #   adjust_joint = []
  #   for cocopart in flip_list:
  #     point = joint[cocopart]
  #     if point[0] < -100 or point[1] < -100:
  #       adjust_joint.append((-1000, -1000))
  #       continue
  #     # if point[0] <= 0 or point[1] <= 0:
  #     #     adjust_joint.append((-1, -1))
  #     #     continue
  #     adjust_joint.append((meta.width - point[0], point[1]))
  #   adjust_joint_list.append(adjust_joint)
  if meta.joint_list.shape[0] > 0:
    adjust_joint_list = np.array([joint[CocoPart.flip_list]
                                  for joint in meta.joint_list], dtype=np.float32)
    adjust_joint_list[..., 0] = meta.width - adjust_joint_list[..., 0]
    meta.joint_list = adjust_joint_list
    meta.joint_list_mask = np.array([joint[CocoPart.flip_list]
                                     for joint in meta.joint_list_mask], dtype=np.float32)
  meta.img = img
  return meta


# def pose_rotation(meta: ImageMeta):
#   deg = random.uniform(-15.0, 15.0)
#   img = meta.img

#   center = (img.shape[1] * 0.5, img.shape[0] * 0.5)       # x, y
#   rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
#   ret = cv2.warpAffine(img, rot_m, img.shape[1::-1],
#                        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
#   if img.ndim == 3 and ret.ndim == 2:
#     ret = ret[:, :, np.newaxis]
#   neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
#   neww = min(neww, ret.shape[1])
#   newh = min(newh, ret.shape[0])
#   newx = int(center[0] - neww * 0.5)
#   newy = int(center[1] - newh * 0.5)
#   # print(ret.shape, deg, newx, newy, neww, newh)
#   img = ret[newy:newy + newh, newx:newx + neww]

#   # adjust meta data
#   adjust_joint_list = []
#   for joint in meta.joint_list:
#     adjust_joint = []
#     for point in joint:
#       if point[0] < -100 or point[1] < -100:
#         adjust_joint.append((-1000, -1000))
#         continue

#       x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
#       adjust_joint.append((x, y))
#     adjust_joint_list.append(adjust_joint)

#   meta.joint_list = adjust_joint_list
#   meta.width, meta.height = neww, newh
#   meta.img = img

#   return meta


def _rotate_coord(shape, newxy, point, angle):
  angle = -1 * angle / 180.0 * math.pi

  ox, oy = shape
  px, py = point

  ox /= 2
  oy /= 2

  qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
  qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

  new_x, new_y = newxy

  qx += ox - new_x
  qy += oy - new_y

  return int(qx + 0.5), int(qy + 0.5)
