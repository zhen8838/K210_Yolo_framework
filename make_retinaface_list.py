import numpy as np
from pathlib import Path
import argparse


def main(root, output_file):
    root = Path(root)
    meta = {}
    for name in ['train', 'val', 'test']:
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
                        annotations = np.zeros((0, 15))
                        if len(labels_copy) == 0:
                            anns.append(annotations)
                        else:
                            for idx, label_copy in enumerate(labels_copy):
                                annotation = np.zeros((1, 15))
                                # bbox
                                annotation[0, 0] = label_copy[0]             # x1
                                annotation[0, 1] = label_copy[1]             # y1
                                annotation[0, 2] = label_copy[0] + label_copy[2]  # x2
                                annotation[0, 3] = label_copy[1] + label_copy[3]  # y2
                                if name == 'train':
                                    # landmarks
                                    annotation[0, 4] = label_copy[4]    # l0_x
                                    annotation[0, 5] = label_copy[5]    # l0_y
                                    annotation[0, 6] = label_copy[7]    # l1_x
                                    annotation[0, 7] = label_copy[8]    # l1_y
                                    annotation[0, 8] = label_copy[10]   # l2_x
                                    annotation[0, 9] = label_copy[11]   # l2_y
                                    annotation[0, 10] = label_copy[13]  # l3_x
                                    annotation[0, 11] = label_copy[14]  # l3_y
                                    annotation[0, 12] = label_copy[16]  # l4_x
                                    annotation[0, 13] = label_copy[17]  # l4_y
                                    if (annotation[0, 4] < 0):
                                        annotation[0, 14] = -1
                                    else:
                                        annotation[0, 14] = 1
                                elif name == 'val':
                                    annotation[0, 14] = -1

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

        meta[name] = np.array([(a, b.astype('float32')) for a, b in zip(img_paths, anns)])

    np.save(output_file, meta, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='retina face dataset root path',
                        default='/home/zqh/workspace/retina_dataset')
    parser.add_argument('--output_file', type=str, help='output file path',
                        default='data/retinaface_img_ann.npy')
    args = parser.parse_args()
    main(args.root, args.output_file)
