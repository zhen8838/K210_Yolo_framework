import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug import KeypointsOnImage
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.contrib.data import assert_element_shape
from tools.base import BaseHelper, INFO, ERROR, NOTE
import matplotlib.pyplot as plt
from pathlib import Path


class PFLDHelper(BaseHelper):
    def __init__(self, image_ann: str, in_hw: tuple, landmark_num: int, attribute_num: int, validation_split=0.1):
        self.in_hw = np.array(in_hw)
        assert self.in_hw.ndim == 1
        self.landmark_num = landmark_num
        self.attribute_num = attribute_num
        self.validation_split = validation_split  # type:float
        if image_ann == None:
            self.train_list = None
            self.test_list = None
        else:
            img_ann_list = np.load(image_ann, allow_pickle=True)

            if isinstance(img_ann_list[()], dict):
                # NOTE can use dict set trian and test dataset
                self.train_list = img_ann_list[()]['train_data']  # type:np.ndarray
                self.test_list = img_ann_list[()]['test_data']  # type:np.ndarray
            elif isinstance(img_ann_list[()], np.ndarray):
                num = int(len(img_ann_list) * self.validation_split)
                self.train_list = img_ann_list[num:]  # type:np.ndarray
                self.test_list = img_ann_list[:num]  # type:np.ndarray
            else:
                raise ValueError(f'{image_ann} data format error!')
            self.train_total_data = len(self.train_list)  # type:int
            self.test_total_data = len(self.test_list)  # type:int

        self.iaaseq = iaa.OneOf([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Affine(rotate=(-10, 10)),  # 随机旋转
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 随机平移
        ])  # type: iaa.meta.Augmenter

    def resize_img(self, raw_img: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(raw_img, self.in_hw, method=0)

    def _compute_dataset_shape(self) -> tuple:
        """ compute dataset shape to avoid keras check shape error

        Returns
        -------
        tuple
            dataset shape lists
        """
        img_shapes = tf.TensorShape([None] + list(self.in_hw) + [3])
        # landmark_shape = tf.TensorShape([None, self.landmark_num * 2])
        # attribute_w_shape = tf.TensorShape([None, 1])
        # eular_shape = tf.TensorShape([None, 3])
        label_shape = tf.TensorShape([None, self.landmark_num * 2 + 1 + 3])
        dataset_shapes = (img_shapes, label_shape)
        return dataset_shapes

    def data_augmenter(self, img, ann):
        """ No implement """
        return img, ann

    def build_datapipe(self, image_ann_list: np.ndarray, batch_size: int,
                       rand_seed: int, is_augment: bool,
                       is_normlize: bool, is_training: bool) -> tf.data.Dataset:
        print(INFO, 'data augment is ', str(is_augment))

        def _parser_wrapper(i: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
            img_path, label = tf.numpy_function(
                lambda idx: (image_ann_list[idx][0].copy(), image_ann_list[idx][1].copy().astype('float32')),
                [i], [tf.dtypes.string, tf.float32])

            raw_img = self.read_img(img_path)

            if is_augment == False:
                raw_img = self.resize_img(raw_img)

            # NOTE standardized image
            img = self.normlize_img(raw_img)

            return img, label

        dataset_shapes = self._compute_dataset_shape()

        def _batch_parser(img: tf.Tensor, label: tf.Tensor) -> [
                tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """ 
                process ann , calc the attribute weights 

                return : img , landmarks , attribute_weight , euluar
            """
            attr = label[:, self.landmark_num * 2:self.landmark_num * 2 + self.attribute_num]
            mat_ratio = tf.reduce_mean(attr, axis=0, keepdims=True)
            mat_ratio = tf.where(mat_ratio > 0, 1. / mat_ratio,
                                 tf.ones([1, self.attribute_num]) * batch_size)
            attribute_weight = tf.matmul(attr, mat_ratio, transpose_b=True)  # [n,1]

            return img, tf.concat((label[:, 0:self.landmark_num * 2], attribute_weight,
                                   label[:, self.landmark_num * 2 + self.attribute_num:]), 1)
        if is_training:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .shuffle(batch_size * 500, rand_seed)
                  .repeat()
                  .map(_parser_wrapper, -1)
                  .batch(batch_size, True)
                  .map(_batch_parser, -1)
                  .prefetch(-1)
                  .apply(assert_element_shape(dataset_shapes)))
        else:
            ds = (tf.data.Dataset.from_tensor_slices(tf.range(len(image_ann_list)))
                  .map(_parser_wrapper, -1)
                  .batch(batch_size, True)
                  .map(_batch_parser, -1)
                  .prefetch(-1)
                  .apply(assert_element_shape(dataset_shapes)))

        return ds

    def draw_image(self, img: np.ndarray, ann: np.ndarray, is_show=True):
        """ draw img and show bbox , set ann = None will not show bbox

        Parameters
        ----------
        img : np.ndarray

        ann : np.ndarray

           shape : [p,x,y,w,h]

        is_show : bool

            show image
        """

        landmark, attribute, euler = np.split(
            ann, [self.landmark_num * 2,
                  self.landmark_num * 2 + self.attribute_num])

        landmark = landmark.reshape(-1, 2) * img.shape[0:2]
        for (x, y) in landmark.astype('uint8'):
            cv2.circle(img, (x, y), 0, (255, 0, 0), 1)
        plt.imshow(img)
        plt.show()


def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256, radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x

    # Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # The dlib shape predictor returns 68 points, we are interested only in a few of those
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    LEFT_EYEBROW_LEFT = [6.825897, 6.760612, 4.402142]
    LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_RIGHT = [-6.825897, 6.760612, 4.402142]
    LEFT_EYE_LEFT = [5.311432, 5.485328, 3.987654]
    LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    RIGHT_EYE_RIGHT = [-5.311432, 5.485328, 3.987654]
    NOSE_LEFT = [2.005628, 1.409845, 6.165652]
    NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    MOUTH_RIGHT = [-2.774015, -2.080775, 5.048531]
    LOWER_LIP = [0.000000, -3.116408, 6.097667]
    CHIN = [0.000000, -7.415691, 4.070434]

    landmarks_3D = np.float32([LEFT_EYEBROW_LEFT,
                               LEFT_EYEBROW_RIGHT,
                               RIGHT_EYEBROW_LEFT,
                               RIGHT_EYEBROW_RIGHT,
                               LEFT_EYE_LEFT,
                               LEFT_EYE_RIGHT,
                               RIGHT_EYE_LEFT,
                               RIGHT_EYE_RIGHT,
                               NOSE_LEFT,
                               NOSE_RIGHT,
                               MOUTH_LEFT,
                               MOUTH_RIGHT,
                               LOWER_LIP,
                               CHIN])

    # Return the 2D position of our landmarks
    assert landmarks_2D is not None, 'landmarks_2D is None'
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    # Get as input the rotational vector
    # Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))

    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = map(lambda temp: temp[0], euler_angles)
    return pitch, yaw, roll


def rotationMatrixToEulerAngles(R):
    # assert(isRotationMatrix(R))
    # To prevent the Gimbal Lock it is possible to use
    # a threshold of 1e-6 for discrimination
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6
    if not singular:
        x = np.atan2(R[2, 1], R[2, 2])
        y = np.atan2(-R[2, 0], sy)
        z = np.atan2(R[1, 0], R[0, 0])
    else:
        x = np.atan2(-R[1, 2], R[1, 1])
        y = np.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class PFLD_Loss(Loss):
    def __init__(self, h: PFLDHelper, landmark_weight=1., eular_weight=1.,
                 reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        """ landmark loss

        Parameters
        ----------
        h : PFLDHelper

        landmark_weight : [float], optional
            by default 1.
        eular_weight : [float], optional
            by default 1.

        """
        super().__init__(reduction=reduction, name=name)
        self.h = h
        self.landmark_weight = landmark_weight
        self.eular_weight = eular_weight

    def call(self, y_true, y_pred):
        true_landmark, true_a_w, true_eular = tf.split(y_true, [self.h.landmark_num * 2, 1, 3], 1)

        pred_landmark, pred_eular = tf.split(y_pred, [self.h.landmark_num * 2, 3], 1)

        eular_loss = self.eular_weight * tf.reduce_sum(
            1. - tf.cos(tf.abs(true_eular - pred_eular)), axis=1)
        landmark_loss = self.landmark_weight * tf.reduce_sum(
            tf.square(true_landmark - tf.sigmoid(pred_landmark)), 1)

        loss = tf.reduce_mean(true_a_w * landmark_loss * eular_loss)
        return loss


def pfld_infer(img_path: Path, infer_model: tf.keras.Model,
               result_path: Path, h: PFLDHelper, radius: int = 1):
    """
        pfld load image and get result~
    Parameters
    ----------
    img_path : Path
        img path or img folder path
    infer_model : k.Model

    result_path : Path
        by default None

    h : PFLDHelper

    """
    print(INFO, f'Load Images from {str(img_path)}')
    if img_path.is_dir():
        img_path_list = []
        for suffix in ['bmp', 'jpg', 'jpeg', 'png']:
            img_path_list += list(img_path.glob(f'*.{suffix}'))
        raw_img = np.array([h.read_img(str(p)).numpy() for p in img_path_list])
    elif img_path.is_file():
        raw_img = tf.expand_dims(h.read_img(str(img_path)), 0).numpy()
    else:
        ValueError(f'{ERROR} img_path `{str(img_path)}` is invalid')

    raw_img_hw = np.array([src.shape[0:2] for src in raw_img])

    if result_path is not None:
        print(INFO, f'Load Nncase Results from {str(result_path)}')
        if result_path.is_dir():
            ncc_result = np.array([np.fromfile(
                str(result_path / (img_path_list[i].stem + '.bin')),
                dtype='float32') for i in range(len(img_path_list))])  # type:np.ndarray
        elif result_path.is_file():
            ncc_result = np.expand_dims(np.fromfile(str(result_path),
                                                    dtype='float32'), 0)  # type:np.ndarray
        else:
            ValueError(f'{ERROR} result_path `{str(result_path)}` is invalid')
    else:
        ncc_result = None
    print(INFO, f'Infer Results')
    resize_img = tf.stack([h.resize_img(src) for src in raw_img])

    img = h.normlize_img(resize_img)
    """ get output """
    result = infer_model.predict(img)  # type:np.ndarray
    if result.ndim == 4:
        result = np.reshape(np.transpose(result, (0, 3, 1, 2)), (-1, h.landmark_num * 2))

    """ show pfld_draw_img_result """
    if ncc_result is None:
        """ parser output """
        pred_landmark = tf.sigmoid(result).numpy()
        """ draw """

        for i in range(len(raw_img)):
            landmark = np.reshape(pred_landmark[i], (h.landmark_num, 2)) * raw_img_hw[i][::-1]

            for j in range(h.landmark_num):
                # NOTE circle( y, x, radius )
                rr, cc = circle(landmark[j][1], landmark[j][0], 1)
                raw_img[i][rr, cc] = (200, 0, 0)

            plt.imshow(raw_img[i])
            plt.show()
    else:
        """ parser output """
        pred_landmark = tf.sigmoid(result).numpy()
        ncc_pred_landmark = tf.sigmoid(ncc_result).numpy()
        """ draw """
        for i in range(len(raw_img)):
            fig = plt.figure()
            ax1 = plt.subplot(121)  # type:plt.Axes
            ax2 = plt.subplot(122)  # type:plt.Axes
            ax1_img = raw_img[i].copy()
            ax2_img = raw_img[i].copy()
            """ plot ax1 """
            landmark = np.minimum(np.reshape(pred_landmark[i], (h.landmark_num, 2)) * raw_img_hw[i][::-1],
                                  raw_img_hw[i][0] - 1)
            for j in range(h.landmark_num):
                # NOTE circle( y, x, radius )
                cv2.circle(ax1_img, tuple(landmark[j]), 1, (200, 0, 0), 1)

            """ plot ax2 """
            landmark = np.minimum(np.reshape(ncc_pred_landmark[i], (h.landmark_num, 2)) * raw_img_hw[i][::-1],
                                  raw_img_hw[i][0] - 1)
            for j in range(h.landmark_num):
                # NOTE circle( y, x, radius )
                cv2.circle(ax2_img, tuple(landmark[j]), radius, (200, 0, 0), 1)

            ax1.imshow(ax1_img)
            ax2.imshow(ax2_img)
            ax1.set_title('GPU Infer')
            ax2.set_title('Ncc Infer')
            plt.show()
