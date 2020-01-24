import mxnet as mx
from mxnet import recordio
import tensorflow as tf
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm, trange
from tools.base import INFO
import pickle
import argparse
import matplotlib.pyplot as plt


def make_example(im_str_a: str, im_str_b: str, label: int) -> bytes:
    """ make example """
    feature = {
        "img_a": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_str_a])),
        "img_b": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_str_b])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def make_train_example(im_str: str, label: int) -> bytes:
    """ make example """
    feature = {
        "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_str])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def load_bin(path: str, image_size=[112, 112]) -> [list, list]:
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0, 1]:
        data = np.empty((len(issame_list) * 2, image_size[0], image_size[1], 3), np.uint8)
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        if not isinstance(_bin, np.ndarray):
            _bin = np.array(np.frombuffer(_bin, dtype=np.uint8), dtype=np.uint8)
        img = cv2.imdecode(_bin, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != image_size[0]:
            img = cv2.resize(img, tuple(image_size))
        for flip in [0, 1]:
            if flip == 1:
                img = cv2.flip(img, 1)
            data_list[flip][i][:] = img
    return (data_list, issame_list)


def main(data_dir: str, save_dir: str, output_file: str):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)

    if save_dir.exists() is False:
        save_dir.mkdir()
    else:
        shutil.rmtree(str(save_dir))
        save_dir.mkdir()

    imgrec: recordio = recordio.MXIndexedRecordIO(str(data_dir / 'train.idx'),
                                                  str(data_dir / 'train.rec'), 'r')
    s = imgrec.read_idx(0)
    header, _ = recordio.unpack(s)
    if header.flag > 0:
        print('header0 label', header.label)
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = []
        idmap = []
        seq_identity = range(int(header.label[0]), int(header.label[1]))
        for identity in seq_identity:
            s = imgrec.read_idx(identity)
            header, _ = recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            idmap.append(list(range(a, b)))
            imgidx += list(range(a, b))
        print('id个数', len(idmap))
        print('图像个数', len(imgidx))

    save_dict = {}
    train_data = []
    print(INFO, 'Start make train set')
    for i, ids in enumerate(tqdm(idmap, total=len(idmap))):
        fname = str(save_dir / f'train_{i}.tfrecords')
        with tf.io.TFRecordWriter(fname) as writer:
            for idx in ids:
                header, s = recordio.unpack(imgrec.read_idx(idx))
                label = int(header.label)
                serialized_example = make_train_example(s, label)
                writer.write(serialized_example)
        train_data.append(fname)

    save_dict['train_data'] = train_data
    save_dict['train_num'] = len(imgidx)

    print(INFO, 'Start make val and test set')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    target = 'lfw,cfp_fp,agedb_30'
    val_num = 0
    with tf.io.TFRecordWriter(str(save_dir / 'val.tfrecords')) as writer:
        for name in target.split(','):
            path = str(data_dir / (name + ".bin"))
            data_list, issame_list = load_bin(path)
            print(INFO, f'Write {name} data into tfrecords\n')
            val_num += len(issame_list) * 2
            for img_src_list in data_list:
                for i in trange(len(issame_list)):
                    label = issame_list[i]
                    im_str_a = tf.image.encode_jpeg(img_src_list[i * 2], quality=100).numpy()
                    im_str_b = tf.image.encode_jpeg(img_src_list[i * 2 + 1], quality=100).numpy()
                    serialized_example = make_example(im_str_a, im_str_b, label)
                    writer.write(serialized_example)

    save_dict['val_data'] = str(save_dir / 'val.tfrecords')
    save_dict['val_num'] = val_num

    np.save(output_file, save_dict, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='tiny imagenet dir path',
                        default='/media/zqh/Datas/faces_ms1m-refine-v2_112x112/faces_emore')
    parser.add_argument('--save_dir', type=str, help='save tfrecords file path',
                        default='/home/zqh/workspace/faces_ms1m')
    parser.add_argument('--output_file', type=str, help='output file path',
                        default='data/ms1m_img_ann.npy')
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.output_file)
