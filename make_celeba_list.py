import numpy as np
from pathlib import Path
import sys
import argparse
from skimage.io import imread, imshow, show, imsave
from skimage.util import crop
from tools.utils import INFO, ERROR, NOTE
from tqdm import tqdm
import shutil


def main(args: dict):
    celeb_root = Path(args.celeb_root)
    bbox_ann_file = Path(args.bbox_ann_file)
    landmark_ann_file = Path(args.landmark_ann_file)
    croped_hw = np.array(args.croped_hw)
    # celeb_root = Path('/home/zqh/workspace/img_celeba')
    # bbox_ann_file = Path('tmp/list_bbox_celeba.txt')
    # landmark_ann_file = Path('/home/zqh/workspace/list_landmarks_celeba.txt')
    # croped_hw=np.array([224,320])
    img_paths = np.loadtxt(str(bbox_ann_file), dtype=str, skiprows=2, usecols=0)
    img_paths = np.array([str(celeb_root / p) for p in img_paths])
    bbox_ann = np.loadtxt(str(bbox_ann_file), dtype=float, skiprows=2, usecols=[1, 2, 3, 4])
    bbox_upper_left = bbox_ann[:, 0:2].copy()  # record upper left coordinate
    bbox_wh = bbox_ann[:, 2:4].copy()  # record bbox width and height
    bbox_ann[:, 0:2] += bbox_ann[:, 2:4] / 2  # NOTE correct xywh(xy is upper left) to xywh(xy is center)

    landmark_ann = np.loadtxt(str(landmark_ann_file), skiprows=2, usecols=list(range(1, 11)))
    landmark_ann = np.reshape(landmark_ann, (-1, args.landmark_num, 2))
    croped_img_root = celeb_root.parent / 'loose_croped_img_celeba'  # type:Path
    croped_img_paths = np.array([str(croped_img_root / p) for p in np.loadtxt(str(bbox_ann_file), dtype=str, skiprows=2, usecols=0)])

    if args.just_mklist == False:
        if croped_img_root.exists() == True:
            shutil.rmtree(str(croped_img_root))
        croped_img_root.mkdir()

        valid_croped_img_paths = []
        valid_croped_bbox = []
        valid_croped_landmark = []
        valid_croped_hw = []
        print(INFO, 'Start Extract Images:')
        for i in tqdm(range(len(img_paths)), unit=' images'):
            # crop all face [h,w] < [224,320]
            if (bbox_ann[i][2:4] < croped_hw[::-1]).all() == True:
                img_src = imread(img_paths[i])
                hw = np.array(img_src.shape[0:2])
                yx = np.array(bbox_ann[i][0:2][::-1])

                left_hw = np.maximum(yx - (croped_hw / 2), 0)
                right_hw = np.maximum(hw - (yx + croped_hw / 2), 0)
                croped_src = crop(img_src, ((left_hw[0], right_hw[0]), (left_hw[1], right_hw[1]), (0, 0)))
                # save img
                imsave(croped_img_paths[i], croped_src)
                # correct landmark and bbox
                cropped_bbox = np.hstack((bbox_ann[i][0:2] - left_hw[::-1], bbox_ann[i][2:4]))
                cropped_landmark = landmark_ann[i] - left_hw[::-1]
                # store in list
                valid_croped_img_paths.append(croped_img_paths[i])
                valid_croped_bbox.append(cropped_bbox)
                valid_croped_landmark.append(cropped_landmark)
                valid_croped_hw.append(croped_src.shape[0:2])

        valid_croped_img_paths = np.array(valid_croped_img_paths)
        valid_croped_bbox = np.array(valid_croped_bbox)
        valid_croped_landmark = np.array(valid_croped_landmark)
        valid_croped_hw = np.array(valid_croped_hw)
        np.savez(str(croped_img_root / 'tmp.npz'),
                 valid_croped_img_paths=valid_croped_img_paths,
                 valid_croped_bbox=valid_croped_bbox,
                 valid_croped_landmark=valid_croped_landmark,
                 valid_croped_hw=valid_croped_hw)

    tmp = np.load(str(croped_img_root / 'tmp.npz'))
    valid_croped_img_paths = tmp['valid_croped_img_paths']
    valid_croped_bbox = tmp['valid_croped_bbox']
    valid_croped_landmark = tmp['valid_croped_landmark']
    valid_croped_hw = tmp['valid_croped_hw']
    lines = []
    print(INFO, 'Start Make Lists:')
    for i in tqdm(range(len(valid_croped_img_paths))):
        p = valid_croped_img_paths[i]
        bbox = valid_croped_bbox[i].reshape((-1, 4))
        hw = valid_croped_hw[i]
        # rescale bbox
        bbox[:, [0, 2]] /= hw[1]  # w
        bbox[:, [1, 3]] /= hw[0]  # h
        # landmark is all image scale [0-1]
        landmark = valid_croped_landmark[i].reshape((-1, args.landmark_num, 2)) / hw[::-1]  # type : np.ndarray

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
    parser.add_argument('--croped_hw', type=int, help='croped image height width', default=(224, 320), nargs='+')
    parser.add_argument('--just_mklist', dest='just_mklist', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    main(args)
