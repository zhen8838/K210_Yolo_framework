from pathlib import Path
import numpy as np
from tools.aligndlib import AlignDlib
from tools.base import INFO, ERROR, NOTE
from skimage.io import imread, imshow, imsave
from yaml import safe_load, safe_dump
import sys
import argparse
from tqdm import tqdm


def main(input_shape: list, align_data: str, org_root: str, new_root: str, identity_file: str, partition_file: str, ann_file: str, is_crop: bool, is_save: bool):
    aligner = AlignDlib(align_data)
    org_root = Path(org_root)
    new_root = Path(new_root)
    """ create new root dir """
    if new_root.exists() == False:
        new_root.mkdir()
    """ get image path lists """
    org_img_paths = np.array([(org_root / number) for number in np.loadtxt(identity_file, dtype=str, usecols=0)])
    img_id = np.array([ids for ids in np.loadtxt(identity_file, dtype=int, usecols=1)])
    mask = np.array([ids for ids in np.loadtxt(partition_file, dtype=int, usecols=1)])
    if is_crop == True:
        print(INFO, 'Start Crop')
        for i in tqdm(range(len(org_img_paths))):
            img_arr = imread(str(org_img_paths[i]))
            # find face
            bb = aligner.getLargestFaceBoundingBox(img_arr)
            if bb != None:
                # croped face to image size
                jc_aligned = aligner.align(input_shape, img_arr, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
                new_img_path = str(new_root / org_img_paths[i].name)
                imsave(new_img_path, jc_aligned)
        print(INFO, 'Finsh Crop')

    if is_save == True:
        """ filter no face image, get new img path and id  """
        new_idx = [int(i.stem) for i in list(new_root.iterdir())]
        new_idx.sort()
        new_idx = np.array(new_idx) - 1
        new_img_paths = np.array([str(new_root / p.name) for p in org_img_paths[new_idx]])
        new_img_id = img_id[new_idx]
        new_mask = mask[new_idx]
        assert len(new_img_paths) == len(new_img_id)

        """ filter only one identity """
        print(INFO, 'Start Find identity')
        mult_id_idx = np.where(np.array([np.count_nonzero(new_img_id == new_img_id[i]) for i in tqdm(range(len(new_img_id)))]) != 1)
        new_img_id = new_img_id[mult_id_idx]
        new_mask = new_mask[mult_id_idx]
        new_img_paths = new_img_paths[mult_id_idx]

        """ filter no same identity image """
        new_identity = []
        for i in tqdm(range(len(new_img_paths))):
            same_idx = np.flatnonzero(new_img_id == new_img_id[i])
            same_idx = np.delete(same_idx, np.where(same_idx == i))
            new_identity.append(same_idx)

        assert len(new_img_paths) == len(new_mask)
        assert len(new_img_paths) == len(new_identity)
        print(INFO, 'Finish Find identity')

        metadata = np.array([new_img_paths,
                             new_identity,
                             new_mask])
        np.save(ann_file, metadata)
        print(INFO, f"Save Metadata in {ann_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', type=int, help='crop face image size', default=96)
    parser.add_argument('--align_data', type=str, help='dlib align data file', default='data/landmarks.dat')
    parser.add_argument('--org_root', type=str, help='celeba root dataset dir', default='/home/zqh/workspace/img_align_celeba')
    parser.add_argument('--new_root', type=str, help='celeba croped dataset dir', default='/home/zqh/workspace/img_cropped_celeba')
    parser.add_argument('--identity_file', type=str, help='celeba dataset identity file', default='data/identity_CelebA.txt')
    parser.add_argument('--partition_file', type=str, help='celeba dataset partition file', default='data/list_eval_partition.txt')
    parser.add_argument('--ann_file', type=str, help='celeba dataset partition file', default='data/celeba_facerec_img_ann.npy')
    parser.add_argument('--is_crop', type=str, help='whether crop the image', default='True', choices=['True', 'False'])
    parser.add_argument('--is_save', type=str, help='whether save the metadata', default='True', choices=['True', 'False'])
    args = parser.parse_args(sys.argv[1:])

    main(args.input_shape, args.align_data, args.org_root, args.new_root, args.identity_file, args.partition_file, args.ann_file,
         True if args.is_crop == 'True' else False, True if args.is_save == 'True' else False)
