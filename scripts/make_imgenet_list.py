import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from tools.base import INFO


def main(train_root: str, val_root: str, outfile: str):
    train_root = Path(train_root)
    val_root = Path(val_root)
    np.random.seed(10101)

    clas_name = tf.io.gfile.listdir(str(train_root))
    clas_dict = dict(zip(clas_name, range(len(clas_name))))
    train_name = []
    for clas in clas_name:
        train_name.extend([str((train_root / clas / name).absolute()) for name in tf.io.gfile.listdir(str(train_root / clas))])

    np.random.shuffle(train_name)

    train_label = list(map(lambda s: clas_dict[s.split('/')[-2]], train_name))

    val_name = tf.io.gfile.listdir(str(val_root))
    val_name.remove('imagenet_2012_validation_synset_labels.txt')
    val_name.sort()
    val_name = [str((val_root / name).absolute()) for name in val_name]
    val_label = list(map(lambda s: clas_dict[s], np.loadtxt(str(val_root / 'imagenet_2012_validation_synset_labels.txt'), np.str)))

    meta = {
        'train_list': [train_name, train_label],
        'train_num': len(train_name),
        'val_list': [val_name, val_label],
        'val_num': len(val_name),
        'clas_dict': clas_dict}

    print(INFO, f'Save Dataset meta file in {outfile}')
    np.save(outfile, meta, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_path', type=str, help='tiny imagenet dir path',
                        default='../imagenet')
    parser.add_argument('--input_val_path', type=str, help='tiny imagenet dir path',
                        default='../imagenetval')
    parser.add_argument('--output_file', type=str, help='output file path',
                        default='data/imgnet_img_ann.npy')
    args = parser.parse_args()
    main(args.input_train_path, args.input_val_path, args.output_file)
