import numpy as np
import tensorflow as tf
from pathlib import Path
from tools.base import BaseHelper, INFO
import argparse


def make_example(img_string: str, label: int, bbox_string: str):
    """ make example """
    feature = {
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_string])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_string])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def main(pkl_path, save_path, output_file):
    data = np.load(open(pkl_path, 'rb'), allow_pickle=True)  # type:dict
    pos_arr = []
    neg_arr = []
    for k, v in data.items():
        if v[1] == 0:  # negative
            neg_arr.append(k)
        else:  # positive
            pos_arr.append(k)
    pos_arr = np.array(pos_arr)
    neg_arr = np.array(neg_arr)

    train_pos_arr, val_pos_arr = np.split(pos_arr, [int(0.8 * len(pos_arr))])
    train_neg_arr, val_neg_arr = np.split(neg_arr, [int(0.8 * len(neg_arr))])

    meta = {'data_path': str(Path(pkl_path).absolute()),
            'train_pos': train_pos_arr,
            'train_neg': train_neg_arr,
            'val_pos': val_pos_arr,
            'val_neg': val_neg_arr,
            'train_num': len(train_pos_arr),
            'val_num': len(val_pos_arr)}

    print(INFO, f'Save Dataset meta file in {output_file}')
    np.save(output_file, meta, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, help='widerface_train_data_gt_8.pkl file path',
                        default='/home/zqh/workspace/widerface_train_data_gt_8.pkl')
    parser.add_argument('--save_path', type=str, help='save image file path',
                        default='/home/zqh/workspace/widerface_train')
    parser.add_argument('--output_file', type=str, help='output file path',
                        default='data/lffd_img_ann.npy')
    args = parser.parse_args()
    main(args.pkl_path, args.save_path, args.output_file)
