import numpy as np
from pathlib import Path
import sys
import argparse
from skimage.io import imread
from tools.utils import INFO, ERROR, NOTE
from tqdm import tqdm


def main(args: dict):
    celeb_root = Path(args.celeb_root)
    bbox_ann_file = Path(args.bbox_ann_file)
    landmark_ann_file = Path(args.landmark_ann_file)
    # celeb_root = Path('/home/zqh/workspace/img_celeba')
    # bbox_ann_file = Path('/home/zqh/workspace/list_bbox_celeba.txt')
    # landmark_ann_file = Path('/home/zqh/workspace/list_landmarks_celeba.txt')

    img_paths = np.loadtxt(str(bbox_ann_file), dtype=str, skiprows=2, usecols=0)
    img_paths = np.array([str(celeb_root / p) for p in img_paths])
    bbox_ann = np.loadtxt(str(bbox_ann_file), dtype=float, skiprows=2, usecols=[1, 2, 3, 4])
    bbox_upper_left = bbox_ann[:, 0:2].copy()  # record upper left coordinate
    bbox_wh = bbox_ann[:, 2:4].copy()  # record bbox width and height
    bbox_ann[:, 0:2] += bbox_ann[:, 2:4] / 2  # NOTE correct xywh(xy is upper left) to xywh(xy is center)

    landmark_ann = np.loadtxt(str(landmark_ann_file), skiprows=2, usecols=list(range(1, 11)))

    lines = []
    print(INFO, 'Start Make Lists:')
    for i in tqdm(range(len(img_paths))):
        p = img_paths[i]
        bbox = bbox_ann[i].reshape((-1, 4))
        hw = np.array(imread(img_paths[i]).shape[0:2])
        bbox[:, [0, 2]] /= hw[1]  # w
        bbox[:, [1, 3]] /= hw[0]  # h
        # landmark is all image scale [0-1]
        landmark = landmark_ann[i].reshape((-1, args.landmark_num, 2)) / hw[::-1]  # type : np.ndarray

        # make box
        true_box = np.hstack((np.zeros((bbox.shape[0], 1)), bbox, landmark.reshape((-1, args.landmark_num * 2))))

        lines.append(np.array([p, true_box, hw]))

    print(INFO, f'Save Lists as {args.output_file}')
    np.save(args.output_file, np.array(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('celeb_root', type=str, help='celeba dataset image root path')
    parser.add_argument('bbox_ann_file', type=str, help='celeba dataset list_bbox_celeba.txt path')
    parser.add_argument('landmark_ann_file', type=str, help='celeba dataset list_landmarks_celeba.txt')
    parser.add_argument('output_file', type=str, help='output file path')
    parser.add_argument('--landmark_num', type=int, help='landmark numbers', default=5)
    args = parser.parse_args(sys.argv[1:])
    main(args)
