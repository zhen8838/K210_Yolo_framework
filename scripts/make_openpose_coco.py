import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.base import INFO
from pycocotools.coco import COCO
from tqdm import tqdm


def main(coco_root: str, outfile: str):
  root = Path(coco_root)
  meta = {}
  for data_name, meta_name in [('train2017', 'train'), ('val2017', 'val')]:
    coco = COCO(f'{root.as_posix()}/annotations/person_keypoints_{data_name}.json')
    coco_size = len(coco.imgs)

    img_paths = []
    img_kps = []
    print(INFO, f'Load {root.as_posix()}/annotations/person_keypoints_{data_name}.json')
    for id, img in tqdm(coco.imgs.items(), total=len(coco.imgs)):
      annIds = coco.getAnnIds(imgIds=[id])
      anns = coco.loadAnns(annIds)

      img_paths.append(f"{root.as_posix()}/{data_name}/{img['file_name']}")

      joint_list = []
      for ann in anns:
        if ann.get('num_keypoints', 0) == 0:
          continue

        kp = np.array(ann['keypoints'])
        xs = kp[0::3]
        ys = kp[1::3]
        vs = kp[2::3]

        joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

      self_joint_list = []
      transform = list(zip(
          [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
          # https://github.com/ildoonet/tf-pose-estimation/issues/159 没有搞错,是因为注释点不同,有一个点需要根据平均值计算.
          [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
      ))

      for prev_joint in joint_list:
        new_joint = []
        for idx1, idx2 in transform:
          j1 = prev_joint[idx1 - 1]
          j2 = prev_joint[idx2 - 1]
          if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
            new_joint.append((-1000, -1000))
          else:
            new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

        new_joint.append((-1000, -1000))
        self_joint_list.append(new_joint)

      # final keypoint ann will be [n,19,2] or []
      img_kps.append(self_joint_list)

    meta[meta_name + '_list'] = (np.array(img_paths), np.array(img_kps))
    meta[meta_name + '_num'] = len(img_paths)

  print(INFO, f'Save Dataset meta file in {outfile}')
  np.save(outfile, meta, allow_pickle=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_train_path',
      type=str,
      help='coco 2017 dataset path',
      default='/home/zqh/workspace/pysot/training_dataset/coco')
  parser.add_argument(
      '--output_file',
      type=str,
      help='output file path',
      default='data/openpose_coco_img_ann.npy')
  args = parser.parse_args()
  main(args.input_train_path, args.output_file)
