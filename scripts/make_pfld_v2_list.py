import numpy as np
from pathlib import Path
import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools.pfld import calculate_pitch_yaw_roll
import cv2
from scripts.make_retinaface_wflw_list import WFLW_98_TO_DLIB_68_IDX_MAPPING


# def plot_landmark(img, landmarks):
#   for landmark in landmarks:
#     cv2.circle(img, tuple(landmark.astype('int32')), 3, [255, 0, 0])
#   plt.imshow(img)
#   plt.show()


# 0-195: landmark 坐标点  196-199: bbox 坐标点;
# 200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
# 201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
# 202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
# 203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
# 204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
# 205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
# 206: 图片名称
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--w300_img_dir', type=str, help='300W images dir',
                      default='/home/zqh/workspace/300w/300W')
  parser.add_argument('--wflw_img_dir', type=str, help='WFLW images dir',
                      default='/home/zqh/workspace/WFLW_images')
  parser.add_argument('--wflw_ann_dir', type=str, help='WFLW annotations dir',
                      default='/home/zqh/workspace/WFLW_annotations')
  parser.add_argument('--output_file', type=str, help='output ann file',
                      default='data/pfld_68_img_ann_list.npy')

  args = parser.parse_args()
  output_file = Path(args.output_file)
  w300_root = Path(args.w300_img_dir)

  wflw_img_dir = Path(args.wflw_img_dir)
  wflw_ann_dir = Path(args.wflw_ann_dir)
  wflw_test_ann_file = wflw_ann_dir / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_test.txt'
  wflw_train_ann_file = wflw_ann_dir / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_train.txt'
  w300_img_dirs = [w300_root / '01_Indoor', w300_root / '02_Outdoor']

  output_file = Path(args.output_file)
  meta = {}

  # 300w dataset
  w300_img_paths = []
  w300_landmarks = []
  w300_attributes = []
  for w300_img_dir in w300_img_dirs:
    ann_files = list(w300_img_dir.glob('*.pts'))
    for ann_file in ann_files:
      img_file = ann_file.as_posix().replace('pts', 'png')
      landmarks = np.loadtxt(ann_file, skiprows=3, max_rows=68)

      w300_img_paths.append(img_file)
      w300_landmarks.append(landmarks)
      w300_attributes.append(np.zeros((6), dtype=np.int32))
  w300_img_paths = np.array(w300_img_paths)
  w300_landmarks = np.array(w300_landmarks)
  w300_attributes = np.array(w300_attributes)

  # wflw train dataset
  wflw_train_anns = np.loadtxt(wflw_train_ann_file.as_posix(), dtype=np.str, delimiter=' ')
  wflw_train_img_paths = np.array([(wflw_img_dir / p).as_posix() for p in wflw_train_anns[:, -1]])
  wflw_train_landmarks = np.reshape(np.asfarray(wflw_train_anns[:, :98 * 2]), (-1, 98, 2))
  wflw_train_landmarks = wflw_train_landmarks[:, WFLW_98_TO_DLIB_68_IDX_MAPPING]
  wflw_train_attributes = wflw_train_anns[:, 200:206].astype(np.int32)

  # wflw test dataset
  wflw_test_anns = np.loadtxt(wflw_test_ann_file.as_posix(), dtype=np.str, delimiter=' ')
  wflw_test_img_paths = np.array([(wflw_img_dir / p).as_posix() for p in wflw_test_anns[:, -1]])
  wflw_test_landmarks = np.reshape(np.asfarray(wflw_test_anns[:, :98 * 2]), (-1, 98, 2))
  wflw_test_landmarks = wflw_test_landmarks[:, WFLW_98_TO_DLIB_68_IDX_MAPPING]
  wflw_test_attributes = wflw_test_anns[:, 200:206].astype(np.int32)

  train_img_paths = np.concatenate([w300_img_paths, wflw_train_img_paths], 0)
  train_img_hws = np.array([cv2.imread(p).shape[:2] for p in train_img_paths]).astype('int32')
  train_landmarks = np.concatenate([w300_landmarks, wflw_train_landmarks], 0).astype('float32')
  train_attributes = np.concatenate([w300_attributes, wflw_train_attributes], 0).astype('float32')

  test_img_paths = wflw_test_img_paths
  test_img_hws = np.array([cv2.imread(p).shape[:2] for p in test_img_paths]).astype('int32')
  test_landmarks = wflw_test_landmarks.astype('float32')
  test_attributes = wflw_test_attributes.astype('float32')

  meta['train_list'] = (train_img_paths, train_img_hws, train_landmarks, train_attributes)
  meta['train_num'] = len(train_img_paths)
  meta['test_list'] = (test_img_paths, test_img_hws, test_landmarks, test_attributes)
  meta['test_num'] = len(test_img_paths)
  np.save(output_file, meta)
