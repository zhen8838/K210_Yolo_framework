import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.getcwd())
from tools.base import BaseHelper, INFO


def make_example(img_str: str):
  """ make example """
  feature = {
      'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str])),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def main(root_path, output_file):

  dataset_root = Path(root_path)
  trainA = dataset_root / 'trainA'
  trainB = dataset_root / 'trainB'
  testA = dataset_root / 'testA'
  testB = dataset_root / 'testB'

  meta = {'train_num': 0, 'test_num': 0}
  for name in ['trainA', 'trainB', 'testA', 'testB']:
    sub_dataset_root = dataset_root / name
    img_paths = [p.as_posix() for p in sub_dataset_root.glob('*.png')]
    tfrecord_path = (dataset_root / (name + '.tfrecords')).as_posix()
    with tf.io.TFRecordWriter(tfrecord_path) as f:
      for img_path in tqdm(img_paths, total=len(img_paths)):
        img_str = tf.io.read_file(img_path).numpy()
        stream = make_example(img_str)
        f.write(stream)
    meta[name] = tfrecord_path
    if 'train' in name:
      meta['train_num'] += len(img_paths)
    elif 'test' in name:
      meta['test_num'] += len(img_paths)

  print(INFO, f'Save Dataset meta file in {output_file}')
  np.save(output_file, meta, allow_pickle=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_path', type=str, help='image dataset root path',
                      default='/home/zqh/workspace/photo2cartoon_resources_20200504/cartoon_data')
  parser.add_argument('--output_file', type=str, help='output file path',
                      default='data/phototransfer_img_ann.npy')
  args = parser.parse_args()
  main(args.root_path, args.output_file)
