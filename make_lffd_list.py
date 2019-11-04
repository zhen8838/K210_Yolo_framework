import numpy as np
import tensorflow as tf
from pathlib import Path
from tools.base import BaseHelper, INFO
import argparse
from tqdm import tqdm


tf.compat.v1.enable_v2_behavior()


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
        tf.io.gfile.rmtree(str(save_path))

    save_path.mkdir(parents=True)

    validation_split = 0.2
    train_pos_list, val_pos_list = np.split(_positive_index,
                                            [int((1 - validation_split) * len(_positive_index))])
    train_neg_list, val_neg_list = np.split(_negative_index,
                                            [int((1 - validation_split) * len(_negative_index))])
    meta_dict = {}

    for idx_list, name in [(train_pos_list, 'train_pos.tfrecords'), (val_pos_list, 'val_pos.tfrecords')]:
        record_file = save_path / name
        meta_dict[record_file.stem] = str(record_file)
        meta_dict[record_file.stem + '_num'] = len(idx_list)
        print(INFO, f'Make List : {str(record_file)}')
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for idx in tqdm(idx_list, total=len(idx_list)):
                im_buf, label, bboxes = data[idx]
                bboxes = tf.io.serialize_tensor(bboxes).numpy()
                serialized_example = make_example(im_buf.tostring(), label, bboxes)
                writer.write(serialized_example)

    for idx_list, name in [(train_neg_list, 'train_neg.tfrecords'), (val_neg_list, 'val_neg.tfrecords')]:
        record_file = save_path / name
        meta_dict[record_file.stem] = str(record_file)
        meta_dict[record_file.stem + '_num'] = len(idx_list)
        print(INFO, f'Make List : {record_file}')
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for idx in tqdm(idx_list, total=len(idx_list)):
                im_buf, label, _ = data[idx]
                bboxes = tf.io.serialize_tensor(np.array(0., dtype=np.float32)).numpy()
                serialized_example = make_example(im_buf.tostring(), label, bboxes)
                writer.write(serialized_example)

    print(INFO, f'Save Dataset meta file in {output_file}')
    np.save(output_file, meta_dict, allow_pickle=True)


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
