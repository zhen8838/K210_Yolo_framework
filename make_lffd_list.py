import numpy as np
import tensorflow as tf
from pathlib import Path
from tools.base import BaseHelper, INFO
import argparse
from tqdm import tqdm
import shutil

tf.compat.v1.enable_v2_behavior()


def main(pkl_path, save_path, output_file):
    data = np.load(open(pkl_path, 'rb'), allow_pickle=True)  # type:dict
    _positive_index = []
    _negative_index = []
    for k, v in data.items():
        if v[1] == 0:  # negative
            _negative_index.append(k)
        else:  # positive
            _positive_index.append(k)
    _positive_index = np.array(_positive_index)
    _negative_index = np.array(_negative_index)

    save_path = Path(save_path)

    if save_path.exists() is True:
        shutil.rmtree(str(save_path))

    save_path.mkdir(parents=True)

    pos_name_list = []
    print(INFO, 'Make Positive List')
    for idx in tqdm(_positive_index, total=len(_positive_index)):
        im_buf, _, bboxes = data[idx]
        im = tf.image.decode_jpeg(im_buf.tostring(), channels=3).numpy()
        name = str(save_path / f'{str(idx)}.npy')
        np.save(name, np.array([im, bboxes]))
        pos_name_list.append(name)

    neg_name_list = []
    print(INFO, 'Make Negative List')
    for idx in tqdm(_negative_index, total=len(_negative_index)):
        im_buf, _, _ = data[idx]
        im = tf.image.decode_jpeg(im_buf.tostring(), channels=3).numpy()
        name = str(save_path / f'{str(idx)}.npy')
        np.save(name, np.array([im, None]))
        neg_name_list.append(name)

    np.save(output_file, np.array([np.array(pos_name_list), np.array(neg_name_list)]))


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
