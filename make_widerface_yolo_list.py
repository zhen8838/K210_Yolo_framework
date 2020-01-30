import os
import re
import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf
from scipy.io import loadmat
from typing import List
from tqdm import tqdm


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
                    value=img_hw))})).SerializeToString()
    return stream


def main(root, output_file):
    root = Path(root)
    meta = {}
    for name in ['train', 'val']:
        img_paths = []
        anns = []
        sub_root = root / name
        with open(sub_root / 'label.txt') as f:
            lines = f.readlines()
            isFirst = True
            labels = []
            for line in lines:
                line = line.rstrip()
                if line.startswith('#'):
                    if isFirst is True:
                        isFirst = False
                    else:
                        labels_copy = np.array(labels.copy())
                        # make annotations
                        annotations = np.zeros((0, 5))
                        if len(labels_copy) == 0:
                            anns.append(annotations)
                        else:
                            for idx, label_copy in enumerate(labels_copy):
                                annotation = np.zeros((1, 5))
                                # bbox
                                annotation[0, 0] = 0.                        # label
                                annotation[0, 1] = label_copy[0]             # x1
                                annotation[0, 2] = label_copy[1]             # y1
                                annotation[0, 3] = label_copy[0] + label_copy[2]  # x2
                                annotation[0, 4] = label_copy[1] + label_copy[3]  # y2

                                annotations = np.append(annotations, annotation, axis=0)

                        anns.append(annotations)
                        labels.clear()
                    path = line[2:]
                    path = sub_root / 'images' / path
                    img_paths.append(str(path))
                else:
                    line = line.split(' ')
                    label = [float(x) for x in line]
                    labels.append(label)

        with tf.io.TFRecordWriter(str(root / f'{name}.tfrecords')) as writer:
            for img_path, ann in tqdm(zip(img_paths, anns), total=len(img_paths)):
                img_str = tf.io.read_file(img_path).numpy()
                img = tf.image.decode_jpeg(img_str, 3)
                img_hw = img.shape.as_list()[: 2]
                writer.write(make_example(Path(img_path).stem, img_str, ann, img_hw))

        meta[name + '_data'] = str(root / f'{name}.tfrecords')
        meta[name + '_num'] = len(img_paths)

    meta['test_data'] = str(root / f'{name}.tfrecords')
    meta['test_num'] = len(img_paths)

    np.save(output_file, meta, allow_pickle=True)


# def main(root: str, output_file: str):
#     root = Path(root)
#     if not root.exists():
#         raise ValueError(f'{str(root)} not exists !')
#     ann_root = root / 'wider_face_split'

#     save_dict = {}
#     for name, ann_file, sub_dir in [
#         ('train', 'wider_face_train.mat', 'WIDER_train'),
#             ('val', 'wider_face_val.mat', 'WIDER_val')]:

#         ann: dict = loadmat(str(ann_root / ann_file), mat_dtype=False, squeeze_me=True)
#         total_num = 0
#         with tf.io.TFRecordWriter(str(root / f'{name}.tfrecords')) as writer:
#             for (blur_label_list, event_list,
#                  expression_label_list, face_bbx_list,
#                  file_list, illumination_label_list,
#                  invalid_label_list, occlusion_label_list,
#                  pose_label_list) in tqdm(zip(ann['blur_label_list'], ann['event_list'],
#                                               ann['expression_label_list'], ann['face_bbx_list'],
#                                               ann['file_list'], ann['illumination_label_list'],
#                                               ann['invalid_label_list'], ann['occlusion_label_list'],
#                                               ann['pose_label_list']),
#                                           total=len(ann['event_list'])):
#                 # get current image sub dir
#                 img_dir = root / sub_dir / 'images' / event_list
#                 total_num += len(file_list)
#                 for (blur_label, expression_label, face_bbx,
#                      img_name, illumination_label,
#                      invalid_label, occlusion_label,
#                      pose_label) in zip(blur_label_list, expression_label_list,
#                                         face_bbx_list, file_list,
#                                         illumination_label_list,
#                                         invalid_label_list, occlusion_label_list,
#                                         pose_label_list):
#                     img_str = tf.io.read_file(str(img_dir / (img_name + '.jpg'))).numpy()
#                     img_hw = tf.image.decode_jpeg(img_str, 3).shape.as_list()[: 2]
#                     face_bbx = np.reshape(face_bbx, (-1, 4))
#                     img_ann = np.hstack((np.zeros((len(face_bbx), 1), 'float32'), face_bbx[:, :2], face_bbx[:, 2:] + face_bbx[:, :2]))
#                     # image ann [cls,x1,y1,x2,y2]
#                     if isinstance(expression_label, int):
#                         blur_label = np.array([blur_label])
#                         expression_label = np.array([expression_label])
#                         illumination_label = np.array([illumination_label])
#                         invalid_label = np.array([invalid_label])
#                         occlusion_label = np.array([occlusion_label])
#                         pose_label = np.array([pose_label])
#                     stream = make_example(img_name, img_str, img_ann, img_hw,
#                                           blur_label, expression_label,
#                                           illumination_label, invalid_label,
#                                           occlusion_label, pose_label)

#                     writer.write(stream)
#         save_dict[name + '_data'] = str(root / f'{name}.tfrecords')
#         save_dict[name + '_num'] = total_num

#     for name, ann_file, sub_dir in [('test', 'wider_face_test.mat', 'WIDER_test')]:

#         ann: dict = loadmat(str(ann_root / ann_file), mat_dtype=False, squeeze_me=True)
#         total_num = 0
#         with tf.io.TFRecordWriter(str(root / f'{name}.tfrecords')) as writer:
#             for (event_list, file_list) in tqdm(zip(ann['event_list'], ann['file_list']),
#                                                 total=len(ann['event_list'])):
#                 # get current image sub dir
#                 img_dir = root / sub_dir / 'images' / event_list
#                 total_num += len(file_list)
#                 for img_name in file_list:
#                     img_str = tf.io.read_file(str(img_dir / (img_name + '.jpg'))).numpy()
#                     img_hw = tf.image.decode_jpeg(img_str, 3).shape.as_list()[: 2]
#                     img_ann = np.zeros((0, 5), 'float32')  # image ann [cls,x1,y1,x2,y2]
#                     stream = make_example(img_name, img_str, img_ann, img_hw,
#                                           np.zeros((0), 'float32'),
#                                           np.zeros((0), 'float32'),
#                                           np.zeros((0), 'float32'),
#                                           np.zeros((0), 'float32'),
#                                           np.zeros((0), 'float32'),
#                                           np.zeros((0), 'float32'))
#                     writer.write(stream)
#         save_dict[name + '_data'] = str(root / f'{name}.tfrecords')
#         save_dict[name + '_num'] = total_num

#     np.save(output_file, save_dict, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='path contain WIDER_train WIDER_test ...')
    parser.add_argument('output_file', type=str, help='output file path')
    args = parser.parse_args()
    main(args.root, args.output_file)
