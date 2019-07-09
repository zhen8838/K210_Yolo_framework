import tensorflow as tf
from tensorflow.python import keras
import os
import sys
import argparse
from pathlib import Path
from termcolor import colored
from tools.utils import INFO, ERROR, NOTE
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def main(pre_ckpt):
    pre_ckpt = Path(pre_ckpt)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(pre_ckpt)
    tflite_model = converter.convert()
    (pre_ckpt.parent / 'yolo_model.tflite').open('wb').write(tflite_model)

    yolo_model = keras.models.load_model(str(pre_ckpt))  # type: keras.Model
    print(NOTE, ' Model Inputs Node: ', yolo_model.inputs)
    print(NOTE, ' Model Outputs Node: ', yolo_model.outputs)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_ckpt', type=str, help='pre-train model file(.h5) path')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.pre_ckpt)
