import os
import re
import numpy as np
import sys
import argparse
from pathlib import Path
import tensorflow as tf


def make_example(img_name: str, img_str: bytes, img_ann: np.ndarray, img_hw: list):
    stream = tf.train.Example(
        features=tf.train.Features(
            feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList
                                        (value=[img_str])),
                'name': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[img_name.encode()])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(
                    value=img_ann[:, 0])),
                'x1': tf.train.Feature(float_list=tf.train.FloatList(
                    value=img_ann[:, 1])),
                'y1': tf.train.Feature(float_list=tf.train.FloatList(
                    value=img_ann[:, 2])),
                'x2': tf.train.Feature(float_list=tf.train.FloatList(
                    value=img_ann[:, 3])),
                'y2': tf.train.Feature(float_list=tf.train.FloatList(
                    value=img_ann[:, 4])),
                'img_hw': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=img_hw)),
            })).SerializeToString()
    return stream


def main(train_file: str, val_file: str, test_file: str, output_file: str):
    voc_root = Path(train_file).parent / 'VOCdevkit'
    if not voc_root.exists():
        raise ValueError(f'{str(voc_root)} not exists !')
    train_list = np.loadtxt(train_file, dtype=str)
    val_list = np.loadtxt(val_file, dtype=str)
    test_list = np.loadtxt(test_file, dtype=str)

    if not os.path.exists('data'):
        os.makedirs('data')

    save_dict = {}
    for name, image_path_list in [('train', train_list),
                                  ('val', val_list),
                                  ('test', test_list)]:

        ann_list = list(image_path_list)
        ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
        ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]
        with tf.io.TFRecordWriter(str(voc_root / f'{name}.tfrecords')) as writer:
            for i in range(len(ann_list)):
                img_name = Path(image_path_list[i]).stem  # image name
                img_str = tf.io.read_file(image_path_list[i]).numpy()
                img_hw = tf.image.decode_jpeg(img_str, 3).shape.as_list()[: 2]
                img_ann = np.loadtxt(ann_list[i], dtype='float32', ndmin=2)  # image ann [cls,x1,y1,x2,y2]
                stream = make_example(img_name, img_str, img_ann, img_hw)
                writer.write(stream)
        save_dict[name + '_data'] = str(voc_root / f'{name}.tfrecords')
        save_dict[name + '_num'] = len(ann_list)

    np.save(output_file, save_dict, allow_pickle=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='trian.txt file path')
    parser.add_argument('val_file', type=str, help='val.txt file path')
    parser.add_argument('test_file', type=str, help='test.txt file path')
    parser.add_argument('output_file', type=str, help='output file path')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.train_file, args.val_file,
         args.test_file, args.output_file)
