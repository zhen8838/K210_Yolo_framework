import tensorflow as tf
import numpy as np
from tools.retinaface import RetinaFaceHelper
from tools.training_engine import BaseTrainingLoop
import imgaug as ia
import imgaug.augmenters as iaa
from tools.bbox_utils import tf_bbox_iou
k = tf.keras
kl = tf.keras.layers


def safe_scale_center(box, scale, limit_x, limit_y):
  cxy = np.clip((box[:2] + box[2:]) * 0.5 * scale, 0, [limit_x - 1, limit_y - 1]).astype('int32')
  return cxy


def truncate_radius(sizes: np.ndarray, stride: np.ndarray) -> np.ndarray:
  return sizes / (stride * 4.0)


def gaussian_truncate_2d(shape, sigma_x=1, sigma_y=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))

  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h


def draw_truncate_gaussian(heatmap: np.ndarray, center: np.ndarray,
                           wh_radius: np.ndarray,
                           k: float = 1) -> np.ndarray:
  w_radius, h_radius = np.ceil(wh_radius).astype('int32')
  h, w = 2 * h_radius + 1, 2 * w_radius + 1
  sigma_x = w / 6
  sigma_y = h / 6
  gaussian = gaussian_truncate_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, w_radius), min(width - x, w_radius + 1)
  top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return gaussian


def gaussian_2d(shape: np.ndarray, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h


def draw_gaussian(heatmap: np.ndarray, center: np.ndarray, radius: float, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
    # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


@np.vectorize
def log(v):
  base = np.exp(1)
  if abs(v) < base:
    return v / base
  if v > 0:
    return np.log(v)
  else:
    return -np.log(-v)


@np.vectorize
def exp(v):
  gate = 1
  if abs(v) < gate:
    return v * np.exp(gate)
  if v > 0:
    return np.exp(v)
  else:
    return -np.exp(-v)


class DBFaceHelper(RetinaFaceHelper):
  colormap = [
      (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112),
      (0, 255, 133), (255, 0, 0), (255, 163, 0), (255, 102, 0),
      (194, 255, 0), (0, 143, 255), (51, 255, 0), (0, 82, 255),
      (0, 255, 41), (0, 255, 173), (10, 0, 255), (173, 255, 0),
      (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
      (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
      (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
      (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
      (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
      (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
      (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
      (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255),
      (8, 255, 214), (7, 255, 224), (255, 184, 6), (10, 255, 71),
      (255, 41, 10), (7, 255, 255), (224, 255, 8), (102, 8, 255),
      (255, 61, 6), (255, 194, 7), (255, 122, 8), (0, 255, 20),
      (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
      (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15),
      (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0),
      (153, 255, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255),
      (0, 173, 255), (31, 0, 255), (11, 200, 200), (61, 230, 250)]

  def __init__(self,
               image_ann: str,
               in_hw: tuple,
               nlandmark: int = 5,
               num_parallel_calls: int = -1):
    self.train_dataset: tf.data.Dataset = None
    self.val_dataset: tf.data.Dataset = None
    self.test_dataset: tf.data.Dataset = None

    self.train_epoch_step: int = None
    self.val_epoch_step: int = None
    self.test_epoch_step: int = None

    img_ann_list = np.load(image_ann, allow_pickle=True)[()]

    # NOTE can use dict set trian and test dataset
    self.train_list: Iterable[
        Tuple[np.ndarray, np.ndarray]] = img_ann_list['train']
    self.val_list: Iterable[Tuple[np.ndarray, np.ndarray]] = img_ann_list['val']
    if 'test' in img_ann_list.keys():
      self.test_list: Iterable[Tuple[np.ndarray, np.ndarray]] = img_ann_list['test']
      self.test_total_data: int = len(self.test_list)
    else:
      self.test_list = self.val_list
      self.test_total_data: int = len(self.val_list)
    self.train_total_data: int = len(self.train_list)
    self.val_total_data: int = len(self.val_list)
    self.num_parallel_calls = num_parallel_calls
    self.org_in_hw: np.ndarray = np.array(in_hw)
    self.in_hw: _EagerTensorBase = tf.Variable(self.org_in_hw, trainable=False)
    self.nlandmark: int = nlandmark
    self.iaaseq = iaa.SomeOf([1, 3], [
        iaa.Fliplr(0.5),
        iaa.Affine(
            scale={
                "x": (0.8, 1.2),
                "y": (0.8, 1.2)
            },
            backend='cv2',
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL),
        iaa.Affine(
            translate_percent={
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2)
            },
            backend='cv2',
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL),
        iaa.Affine(
            rotate=(-30, 30),
            backend='cv2',
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL),
        iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))
    ], True)

  def ann_to_label(self, bbox, landm, clses, haslandmark):
    posweight_radius = 2
    stride = 4
    keepsize = 8
    fm_width = self.org_in_hw[1] // stride
    fm_height = self.org_in_hw[0] // stride
    # 热图gt
    heatmap_gt = np.zeros((1, fm_height, fm_width), np.float32)
    # 热图权重
    heatmap_posweight = np.zeros((1, fm_height, fm_width), np.float32)
    # 不知道
    keep_mask = np.ones((1, fm_height, fm_width), np.float32)
    # bbox回归
    reg_tlrb = np.zeros((1 * 4, fm_height, fm_width), np.float32)
    # bbox mask
    reg_mask = np.zeros((1, fm_height, fm_width), np.float32)
    # 不知道
    distance_map = np.zeros((1, fm_height, fm_width), np.float32) + 1000
    # landmark 回归
    landmark_gt = np.zeros((1 * self.nlandmark * 2, fm_height, fm_width), np.float32)
    # landmark mask
    landmark_mask = np.zeros((1, fm_height, fm_width), np.float32)

    hassmall = False

    for box, landmark, clse, haslandm in zip(bbox, landm,
                                             clses.ravel(),
                                             haslandmark.ravel()):
      area = np.prod(box[2:] - box[:2])
      cx, cy = safe_scale_center(box, 1 / stride, fm_width, fm_height)
      reg_box = box / stride  # xyxy
      isSmallObj = area < keepsize * keepsize
      box_wh = box[2:] - box[:2]
      w, h = box_wh / stride
      x0 = int(np.clip(cx - w // 2, 0, fm_width - 1))
      y0 = int(np.clip(cy - h // 2, 0, fm_height - 1))
      x1 = int(np.clip(cx + w // 2, 0, fm_width - 1) + 1)
      y1 = int(np.clip(cy + h // 2, 0, fm_height - 1) + 1)

      if isSmallObj:
        cx, cy = safe_scale_center(box, 1 / stride, fm_width, fm_height)
        keep_mask[clse, cy, cx] = 0  # 对于过小的目标中心需要mask掉
        # 不但中心
        if x1 - x0 > 0 and y1 - y0 > 0:
          keep_mask[0, y0:y1, x0:x1] = 0
        hassmall = True
        if area >= 5 * 5:
          # distance_map 设 0
          distance_map[clse, cy, cx] = 0
          # 这个classes的回归box
          reg_tlrb[clse * 4:(clse + 1) * 4, cy, cx] = reg_box
          # 这个classes mask掉
          reg_mask[clse, cy, cx] = 1
        continue

      if x1 - x0 > 0 and y1 - y0 > 0:
        # print(x0, y0, x1, y1)  # 这个实际上就是reg_box的int化,就是人脸缩放后的区域大小
        keep_mask[0, y0:y1, x0:x1] = 1
      # 上面那个是整个目标的区域
      # 这里是这个目标区域的中高斯部分的半径,也就是再缩小4倍
      wh_radius = truncate_radius(box_wh, stride)
      # 将高斯部分绘制在其中
      gaussian_map = draw_truncate_gaussian(heatmap_gt[clse, :, :],
                                            (cx, cy), wh_radius)
      # plt.imshow(heatmap_gt[0])
      # plt.imshow(gaussian_map)
      # 最大和最小目标
      mxface = 300
      miface = 25
      mxline = np.max(box_wh)  # 选择长边
      # gamma我没懂干啥的
      gamma = (mxline - miface) / (mxface - miface) * 10
      gamma = np.minimum(np.maximum(0, gamma), 10) + 1
      # draw_gaussian 和 draw_truncate_gaussian 有什么关系???
      xx = draw_gaussian(heatmap_posweight[clse, :, :], (cx, cy), posweight_radius, k=gamma)
      # plt.imshow(heatmap_posweight[0])
      range_expand_x, range_expand_y = np.ceil(wh_radius).astype('int32')
      min_expand_size = 3
      range_expand_x = max(min_expand_size, range_expand_x)
      range_expand_y = max(min_expand_size, range_expand_y)

      icx, icy = cx, cy
      reg_landmark = None
      fill_threshold = 0.3

      if haslandm:
        # classes大于0表示有landmark
        xxyy_cat_landmark = np.concatenate((landmark[::2], landmark[1::2]))
        reg_landmark = xxyy_cat_landmark / stride
        xxyy = np.repeat([cx, cy], 5)
        # 计算landmark的坐标到中心的偏移
        rvalue = (reg_landmark - xxyy)
        # 这里的log太tm奇怪了...
        landmark_gt[clse * 10:(clse + 1) * 10, cy, cx] = log(rvalue) / 4
        landmark_mask[clse, cy, cx] = 1

      rotate = False
      if not rotate:
        # 整个方形区域里面每个像素都算一遍distance
        for cx in range(icx - range_expand_x, icx + range_expand_x + 1):
          for cy in range(icy - range_expand_y, icy + range_expand_y + 1):
            if cx < fm_width and cy < fm_height and cx >= 0 and cy >= 0:

              my_gaussian_value = 0.9
              gy, gx = cy - icy + range_expand_y, cx - icx + range_expand_x
              if gy >= 0 and gy < gaussian_map.shape[0] and gx >= 0 and gx < gaussian_map.shape[1]:
                my_gaussian_value = gaussian_map[gy, gx]

              distance = np.sqrt((cx - icx)**2 + (cy - icy)**2)
              if my_gaussian_value > fill_threshold or distance <= min_expand_size:
                already_distance = distance_map[clse, cy, cx]
                my_mix_distance = (1 - my_gaussian_value) * distance

                if my_mix_distance > already_distance:
                  continue

                distance_map[clse, cy, cx] = my_mix_distance
                # plt.imshow(distance_map[0])
                # 这里也是所有可能的位置都赋值reg_box
                reg_tlrb[clse * 4:(clse + 1) * 4, cy, cx] = reg_box
                reg_mask[clse, cy, cx] = 1

    return heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, keep_mask

  def normlize_img(self, img: tf.Tensor) -> tf.Tensor:
    """ normlize img """
    return (tf.cast(img, tf.float32) / 255. - 0.5) / 0.5

  def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                     is_augment: bool, is_normlize: bool,
                     is_training: bool) -> tf.data.Dataset:

    def _wrapper(i: tf.Tensor) -> tf.Tensor:
      path, ann = tf.numpy_function(lambda idx: tuple(image_ann_list[idx]), [i],
                                    [tf.string, tf.float32])

      img = self.read_img(path)
      img, *ann = self.process_img(img, ann, self.in_hw, is_augment, True,
                                   is_normlize)
      bbox, landm, haslandmark = ann
      clas = tf.zeros_like(haslandmark, dtype=tf.int32)
      (heatmap_gt, heatmap_posweight, reg_tlrb,
       reg_mask, landmark_gt, landmark_mask,
       keep_mask) = tf.numpy_function(self.ann_to_label,
                                      [bbox, landm, clas, haslandmark],
                                      [tf.float32, tf.float32, tf.float32,
                                       tf.float32, tf.float32, tf.float32,
                                       tf.float32], name='ann_to_label')
      num_objs = tf.shape(bbox)[0]
      img.set_shape((None, None, 3))
      fm_width = self.org_in_hw[1] // 4
      fm_height = self.org_in_hw[0] // 4
      heatmap_gt.set_shape((1, fm_height, fm_width))
      heatmap_posweight.set_shape((1, fm_height, fm_width))
      reg_tlrb.set_shape((1 * 4, fm_height, fm_width))
      reg_mask.set_shape((1, fm_height, fm_width))
      landmark_gt.set_shape((1 * self.nlandmark * 2, fm_height, fm_width))
      landmark_mask.set_shape((1, fm_height, fm_width))
      keep_mask.set_shape((1, fm_height, fm_width))

      return (img, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask,
              landmark_gt, landmark_mask, num_objs, keep_mask)

    if is_training:
      dataset = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                 shuffle(batch_size * 500).
                 repeat().
                 map(_wrapper, self.num_parallel_calls).
                 batch(batch_size, True).
                 prefetch(-1))
    else:
      dataset = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list))).
                 map(_wrapper, -1).
                 batch(batch_size, True).
                 prefetch(-1))
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    return dataset

  def colors_img(self, heatmap: np.ndarray) -> np.ndarray:
    color = np.array([np.array(self.colormap, dtype=np.float32)[c][np.newaxis, np.newaxis, :] *
                      heatmap[:, :, c:c + 1] for c in range(self.class_num)])
    return cv2.resize(np.sum(color, 0), tuple(self.org_in_hw))

  def blend_img(self, raw_img: np.ndarray, colors_img: np.ndarray, factor: float = 0.6) -> np.ndarray:
    return (np.clip(raw_img + colors_img * (1 + factor), 0, 255)).astype('uint8')

  def draw_image(self, img, ann, heatmap=None, is_show=True):
    bbox, landm, clses = ann
    if not isinstance(img, np.ndarray):
      img = img.numpy()
      bbox = bbox.numpy()
      landm = landm.numpy()
      clses = clses.numpy()
    for i, flag in enumerate(clses):
      if flag == 1:
        cv2.rectangle(img, tuple(bbox[i][:2].astype(int)),
                      tuple(bbox[i][2:].astype(int)), (255, 0, 0))
        for ldx, ldy in zip(landm[i][0::2].astype(int),
                            landm[i][1::2].astype(int)):
          cv2.circle(img, (ldx, ldy), 3, (0, 0, 255), 1)

    if heatmap is not None:
      if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.numpy()

      img = self.blend_img(img, self.colors_img(heatmap))

    if is_show:
      plt.tight_layout()
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      plt.show()
    return img


class DBfaceTrainingLoop(BaseTrainingLoop):
  """ 
    iou_method: ciou
  """

  def set_metrics_dict(self):
    d = {
        'train': {
            'loss': tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'hm': tf.keras.metrics.Mean('hm_loss', dtype=tf.float32),
            'reg': tf.keras.metrics.Mean('reg_loss', dtype=tf.float32),
            'ldmk': tf.keras.metrics.Mean('landmark_loss', dtype=tf.float32),
        },
        'val': {
            'loss': tf.keras.metrics.Mean('vloss', dtype=tf.float32),
            'hm': tf.keras.metrics.Mean('vhm_loss', dtype=tf.float32),
            'reg': tf.keras.metrics.Mean('vreg_loss', dtype=tf.float32),
            'ldmk': tf.keras.metrics.Mean('vlandmark_loss', dtype=tf.float32),
        }
    }
    return d

  @staticmethod
  def focal_loss(pred, gt, pos_weights, keep_mask=None):
    pos_inds = tf.cast(tf.equal(gt, 1), tf.float32)
    neg_inds = tf.cast(tf.less(gt, 1), tf.float32)

    neg_weights = tf.pow(1 - gt, 4)
    pos_loss = tf.math.log(pred) * tf.pow(1 - pred, 2) * pos_weights
    neg_loss = tf.math.log(1 - pred) * tf.pow(pred, 2) * neg_weights * neg_inds

    if keep_mask is not None:
      pos_loss = tf.reduce_sum(pos_loss * keep_mask)
      neg_loss = tf.reduce_sum(neg_loss * keep_mask)
    else:
      pos_loss = tf.reduce_sum(pos_loss)
      neg_loss = tf.reduce_sum(neg_loss)
    return -(pos_loss + neg_loss)

  @staticmethod
  def smooth_l1_loss(x, t, weight, sigma=1):
    sigma2 = sigma ** 2

    diff = weight * (x - t)
    abs_diff = tf.abs(diff)
    flag = tf.cast(abs_diff < (1. / sigma2), tf.float32)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return tf.reduce_sum(y)

  @staticmethod
  def wing_loss(x, t, weight, sigma=1, w=10, e=2):
    C = w - w * np.log(1 + w / e)

    diff = weight * (x - t)
    abs_diff = tf.abs(diff)
    flag = tf.cast(abs_diff < w, tf.float32)
    y = (flag * w * tf.math.log(1 + abs_diff / e) +
         (1 - flag) * (abs_diff - C))
    return tf.reduce_sum(y)

  @staticmethod
  def iou_loss(pred, gt, weight, method='ciou'):
    # pred is   b, h, w, 4
    # gt is     b, h, w, 4
    # mask is   b, h, w, 1
    # 4 channel is x, y, r, b - cx
    h, w = pred.shape[1:3]
    weight = tf.squeeze(weight, -1)
    mask = weight > 0
    weight = tf.boolean_mask(weight, mask)
    avg_factor = tf.reduce_sum(weight)
    # if tf.equal(avg_factor, 0.):
    #   return tf.stop_gradient(tf.constant(0., tf.float32))
    # else:
    x = tf.range(0, w)
    y = tf.range(0, h)
    shift_y, shift_x = tf.meshgrid(y, x)
    shift = tf.cast(tf.stack((shift_x, shift_y), -1), tf.float32)

    pred_boxes = tf.concat((shift - pred[..., :2], shift + pred[..., 2:]), -1)

    pred_boxes = tf.boolean_mask(pred_boxes, mask)
    gt_boxes = tf.boolean_mask(gt, mask)
    ious = tf_bbox_iou(pred_boxes, gt_boxes, method=method)
    iou_distance = 1 - ious
    return tf.math.divide_no_nan(tf.reduce_sum(iou_distance * weight), avg_factor)

  @tf.function
  def train_step(self, iterator, num_steps_to_run, metrics):

    def step_fn(inputs):
      """Per-Replica training step function."""
      (images, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask,
       landmark_gt, landmark_mask, num_objs,
       keep_mask) = inputs

      with tf.GradientTape() as tape:
        batch_objs = tf.reduce_sum(num_objs)
        hm, tlrb, landmark = self.train_model(images, training=True)
        hm = tf.nn.sigmoid(hm)
        hm = tf.clip_by_value(hm, 1e-4, 1 - 1e-4)
        tlrb = tf.exp(tlrb)

        heatmap_gt = tf.transpose(heatmap_gt, [0, 2, 3, 1])
        heatmap_posweight = tf.transpose(heatmap_posweight, [0, 2, 3, 1])
        reg_tlrb = tf.transpose(reg_tlrb, [0, 2, 3, 1])
        reg_mask = tf.transpose(reg_mask, [0, 2, 3, 1])
        landmark_gt = tf.transpose(landmark_gt, [0, 2, 3, 1])
        landmark_mask = tf.transpose(landmark_mask, [0, 2, 3, 1])
        keep_mask = tf.transpose(keep_mask, [0, 2, 3, 1])

        hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight,
                                  keep_mask=keep_mask) / tf.cast(batch_objs, tf.float32)
        reg_loss = self.iou_loss(tlrb, reg_tlrb, reg_mask, self.hparams.iou_method) * 5
        landmark_loss = self.wing_loss(landmark, landmark_gt, landmark_mask, w=2) * 0.1
        loss = hm_loss + reg_loss + landmark_loss

      scaled_loss = self.optimizer_minimize(loss, tape,
                                            self.optimizer,
                                            self.train_model)

      if self.hparams.ema.enable:
        self.ema.update()
      # loss metric
      metrics.loss.update_state(scaled_loss)
      metrics.hm.update_state(hm_loss)
      metrics.reg.update_state(reg_loss)
      metrics.ldmk.update_state(landmark_loss)

    for _ in tf.range(num_steps_to_run):
      self.run_step_fn(step_fn, args=(next(iterator),))

  @tf.function
  def val_step(self, dataset, metrics):
    if self.hparams.ema.enable:
      val_model = self.ema.model
    else:
      val_model = self.val_model

    def step_fn(inputs):
      (images, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask,
       landmark_gt, landmark_mask, num_objs,
       keep_mask) = inputs

      batch_objs = tf.reduce_sum(num_objs)
      hm, tlrb, landmark = val_model(images, training=False)
      hm = tf.nn.sigmoid(hm)
      hm = tf.clip_by_value(hm, 1e-4, 1 - 1e-4)
      tlrb = tf.exp(tlrb)

      heatmap_gt = tf.transpose(heatmap_gt, [0, 2, 3, 1])
      heatmap_posweight = tf.transpose(heatmap_posweight, [0, 2, 3, 1])
      reg_tlrb = tf.transpose(reg_tlrb, [0, 2, 3, 1])
      reg_mask = tf.transpose(reg_mask, [0, 2, 3, 1])
      landmark_gt = tf.transpose(landmark_gt, [0, 2, 3, 1])
      landmark_mask = tf.transpose(landmark_mask, [0, 2, 3, 1])
      keep_mask = tf.transpose(keep_mask, [0, 2, 3, 1])

      hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight,
                                keep_mask=keep_mask) / tf.cast(batch_objs, tf.float32)
      reg_loss = self.iou_loss(tlrb, reg_tlrb, reg_mask, self.hparams.iou_method) * 5
      landmark_loss = self.wing_loss(landmark, landmark_gt, landmark_mask, w=2) * 0.1
      loss = hm_loss + reg_loss + landmark_loss

      metrics.loss.update_state(loss)
      metrics.hm.update_state(hm_loss)
      metrics.reg.update_state(reg_loss)
      metrics.ldmk.update_state(landmark_loss)

    for inputs in dataset:
      self.run_step_fn(step_fn, args=(inputs,))
