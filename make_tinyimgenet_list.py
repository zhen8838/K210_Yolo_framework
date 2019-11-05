import tensorflow as tf
from pathlib import Path
from tools.base import INFO
from tqdm import trange, tqdm
import numpy as np
import argparse

tf.compat.v1.enable_v2_behavior()


def make_example(img_string: str, label: int):
    """ make example """
    feature = {
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_string])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def main(input_path: str, save_path: str, output_file: str):
    input_path = Path(input_path)
    save_path = Path(save_path)
    output_file = Path(output_file)

    if tf.io.gfile.exists(str(save_path)) is True:
        tf.io.gfile.rmtree(str(save_path))
    tf.io.gfile.makedirs(str(save_path))

    name_path = list((input_path / 'train').iterdir())
    name_path.sort()
    name2label = {}
    [name2label.update({name.stem: i}) for i, name in enumerate(name_path)]

    meta_dict = {'train_list': [], 'val_list': [], 'test_list': []}
    meta_dict['train_num'] = 0

    print(INFO, f'Make Train tfrecords in {str(save_path)} :')
    for subpath in tqdm(name_path, total=len(name_path)):  # type:Path
        record_file = save_path / f'train_{subpath.stem}.tfrecords'
        meta_dict['train_list'].append(str(record_file))
        img_paths = list((subpath / 'images').iterdir())
        meta_dict['train_num'] += len(img_paths)
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for img_path in img_paths:
                im_str = tf.io.read_file(str(img_path)).numpy()
                label = name2label[subpath.stem]
                serialized_example = make_example(im_str, label)
                writer.write(serialized_example)

    print(INFO, f'Make Val tfrecords in {str(save_path)} :')
    val_arr = np.loadtxt(str(input_path / 'val/val_annotations.txt'), dtype=str, usecols=[0, 1])
    record_file = save_path / f'val.tfrecords'
    meta_dict['val_list'].append(str(record_file))
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for img_name, name in tqdm(val_arr, total=len(val_arr)):
            im_str = tf.io.read_file(str(input_path / f'val/images/{img_name}')).numpy()
            label = name2label[name]
            serialized_example = make_example(im_str, label)
            writer.write(serialized_example)

    meta_dict['val_num'] = len(val_arr)

    print(INFO, f'Make Test tfrecords in {str(save_path)} :')
    img_names = tf.io.gfile.listdir(str(input_path / 'test/images'))
    record_file = save_path / f'test.tfrecords'
    meta_dict['test_list'].append(str(record_file))
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for img_name in tqdm(img_names, total=len(img_names)):
            im_str = tf.io.read_file(str(input_path / f'test/images/{img_name}')).numpy()
            label = 0
            serialized_example = make_example(im_str, label)
            writer.write(serialized_example)

    meta_dict['test_num'] = len(img_names)

    print(INFO, f'Save Meta file in {str(output_file)}')
    np.save(str(output_file), meta_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='tiny imagenet dir path',
                        default='/home/zqh/workspace/tiny-imagenet-200')
    parser.add_argument('--save_path', type=str, help='save tfrecords file path',
                        default='/home/zqh/workspace/tiny-imagenet-tfrecord')
    parser.add_argument('--output_file', type=str, help='output file path',
                        default='data/tinyimgnet_img_ann.npy')
    args = parser.parse_args()
    main(args.input_path, args.save_path, args.output_file)
