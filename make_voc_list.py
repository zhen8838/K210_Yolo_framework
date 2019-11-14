import os
import re
import numpy as np
import sys
import argparse
from matplotlib.pyplot import imread


def main(train_file: str, val_file: str, test_file: str, output_file: str):
    train_list = np.loadtxt(train_file, dtype=str)
    val_list = np.loadtxt(val_file, dtype=str)
    test_list = np.loadtxt(test_file, dtype=str)

    if not os.path.exists('data'):
        os.makedirs('data')

    save_dict = {}
    for name, image_path_list in [('train_data', train_list),
                                  ('val_data', val_list),
                                  ('test_data', test_list)]:

        ann_list = list(image_path_list)
        ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
        ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]

        save_dict[name] = np.array([
            np.array([
                image_path_list[i],  # image path
                np.loadtxt(ann_list[i], dtype=float, ndmin=2),  # image ann [cls,x,y,w,h]
                np.array(imread(image_path_list[i]).shape[0:2])]  # image [h w]
            ) for i in range(len(ann_list))])

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
