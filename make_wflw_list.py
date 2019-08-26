# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
import shutil
from tools.landmarkutils import calculate_pitch_yaw_roll
import sys
import argparse
from tqdm import tqdm


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        imgDir = Path(imgDir)
        self.image_size = image_size
        line = line.strip().split()
        # 0-195: landmark 坐标点  196-199: bbox 坐标点;
        # 200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        # 201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        # 202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        # 203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        # 204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        # 205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        # 206: 图片名称
        assert(len(line) == 207)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32)
        flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.path = imgDir / line[206]  # type:Path
        self.img = None

        self.imgs = []
        self.landmarks = []
        self.boxes = []
        self.mirror_idx = np.array([
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
            14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42,
            50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 59,
            58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67,
            66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89,
            88, 95, 94, 93, 97, 96])

    def load_data(self, is_train, repeat):

        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh / 2).astype(np.int32)
        img = cv2.imread(str(self.path))
        boxsize = int(np.max(wh) * 1.2)
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        if is_train:
            imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy) / boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-20, 20)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx, cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size // 2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if self.mirror_idx is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[self.mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        path = Path(path)
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = path / (prefix + '_' + str(i) + '.png')  # type:Path
            assert not save_path.exists(), save_path
            cv2.imwrite(str(save_path), img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append([lanmark[index][0]*img.shape[0],lanmark[index][1]*img.shape[1]])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str, lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(str(save_path), landmark_str, attributes_str, euler_angles_str)
            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        outDir = Path(outDir)
        save_img = outDir / 'imgs'  # type:Path

        if not save_img.exists():
            save_img.mkdir()

        for i, line in tqdm(enumerate(lines), total=len(lines), unit=' image'):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 10)
            filename = img_name.stem
            label_txt = Img.save_data(str(save_img), str(i) + '_' + filename)
            labels.append(label_txt)

    with (outDir / 'list.txt').open('w') as f:
        for label in labels:
            f.writelines(label)


def gen_data(file_list):
    with open(file_list, 'r') as f:
        lines = f.readlines()
    filenames, landmarks, attributes, euler_angles = [], [], [], []
    for line in lines:
        line = line.strip().split()
        path = line[0]
        landmark = line[1:197]
        attribute = line[197:203]
        euler_angle = line[203:206]

        landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        euler_angle = np.asarray(euler_angle, dtype=np.float32)
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)
        euler_angles.append(euler_angle)

    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    euler_angles = np.asarray(euler_angles, dtype=np.float32)
    return (filenames, landmarks, attributes, euler_angles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wflw_ann_dir', type=str, help='WFLW annotations dir')
    parser.add_argument('wflw_img_dir', type=str, help='WFLW images dir')
    parser.add_argument('--just_mklist', dest='just_mklist', action='store_true')
    parser.add_argument('out_img_dir', type=str, help='croped images dir')

    args = parser.parse_args(sys.argv[1:])
    wflw_ann_dir = Path(args.wflw_ann_dir)
    imageDirs = Path(args.wflw_img_dir)
    out_img_dir = Path(args.out_img_dir)
    output_file = Path('data/wflw_img_ann.npy')
    landmarkDirs = [
        wflw_ann_dir / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_test.txt',
        wflw_ann_dir / 'list_98pt_rect_attr_train_test' / 'list_98pt_rect_attr_train.txt']

    outDirs = [out_img_dir / 'test_data', out_img_dir / 'train_data']

    if not args.just_mklist:
        for landmarkDir, outDir in zip(landmarkDirs, outDirs):
            print(outDir)

            if outDir.exists():
                shutil.rmtree(str(outDir))

            outDir.mkdir(parents=True)

            if 'list_98pt_rect_attr_test.txt' in str(landmarkDir):
                is_train = False
            else:
                is_train = True
            imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)

    img_ann_dict = {}
    print("Make Image Annotation List")
    for outDir in outDirs:
        file_list = outDir / 'list.txt'
        filenames, landmarks, attributes, euler_angles = gen_data(str(file_list))
        # NOTE concat landmarks [n,196]  attributes [n,6]  euler_angles [n,3]
        boxes = np.hstack((landmarks, attributes, euler_angles))

        img_ann_dict[outDir.stem] = np.array([
            np.array([
                filenames[i],  # image path
                boxes[i],  # boxes contain [landmarks, attributes, euler_angles]
                np.array(cv2.imread(filenames[i]).shape[0:2])]  # image [h w]
            ) for i in tqdm(range(len(filenames)))])

    np.save(output_file, img_ann_dict)
