import tensorflow.python as tf
from tensorflow.contrib.data import assert_element_shape
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler
from tools.utils import Helper, create_loss_fn, INFO, ERROR, NOTE
from tools.custom import Yolo_Precision, Yolo_Recall
from models.yolonet import *
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import sys
import argparse
from termcolor import colored
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))


def main(args, train_set, class_num, pre_ckpt, model_def,
         depth_multiplier, is_augmenter, image_size, output_size,
         batch_size, rand_seed, max_nrof_epochs, init_learning_rate,
         learning_rate_decay_factor, obj_weight, noobj_weight,
         wh_weight, obj_thresh, iou_thresh, vaildation_split, log_dir,
         is_prune, initial_sparsity, final_sparsity, end_epoch, frequency):
    # Build path
    log_dir = (Path(log_dir) / datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))  # type: Path
    ckpt_weights = log_dir / 'yolo_weights.h5'
    ckpt = log_dir / 'yolo_model.h5'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    write_arguments_to_file(args, str(log_dir / 'args.txt'))

    # Build utils
    h = Helper(f'data/{train_set}_img_ann.npy', class_num, f'data/{train_set}_anchor.npy',
               np.reshape(np.array(image_size), (-1, 2)), np.reshape(np.array(output_size), (-1, 2)), vaildation_split)
    h.set_dataset(batch_size, rand_seed, is_training=(is_augmenter == 'True'))

    # Build network
    network = eval(model_def)  # type :yolo_mobilev2
    yolo_model, yolo_model_warpper = network([image_size[0], image_size[1], 3], len(h.anchors[0]), class_num, alpha=depth_multiplier)

    if pre_ckpt != None and pre_ckpt != 'None' and pre_ckpt != '':
        if 'h5' in pre_ckpt:
            yolo_model_warpper.load_weights(str(pre_ckpt))
            print(INFO, f' Load CKPT {str(pre_ckpt)}')
        else:
            print(ERROR, ' Pre CKPT path is unvalid')

    # prune model
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                     final_sparsity=final_sparsity,
                                                     begin_step=0,
                                                     end_step=h.train_epoch_step * end_epoch,
                                                     frequency=frequency)
    }

    if is_prune == 'True':
        train_model = sparsity.prune_low_magnitude(yolo_model_warpper, **pruning_params)
    else:
        train_model = yolo_model_warpper

    train_model.compile(
        keras.optimizers.Adam(
            lr=init_learning_rate,
            decay=learning_rate_decay_factor),
        loss=[create_loss_fn(h, obj_thresh, iou_thresh, obj_weight, noobj_weight, wh_weight, layer)
              for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)],
        metrics=[Yolo_Precision(obj_thresh, name='p'), Yolo_Recall(obj_thresh, name='r')])

    """ NOTE fix the dataset output shape """
    shapes = (train_model.input.shape, tuple(h.output_shapes))
    h.train_dataset = h.train_dataset.apply(assert_element_shape(shapes))
    h.test_dataset = h.test_dataset.apply(assert_element_shape(shapes))

    """ Callbacks """
    if is_prune == 'True':
        cbs = [
            sparsity.UpdatePruningStep(),
            sparsity.PruningSummaries(log_dir=str(log_dir), profile_batch=0)]
    else:
        cbs = [TensorBoard(str(log_dir), update_freq='batch', profile_batch=3)]

    # Training
    try:
        train_model.fit(h.train_dataset, epochs=max_nrof_epochs,
                        steps_per_epoch=h.train_epoch_step, callbacks=cbs,
                        validation_data=h.test_dataset, validation_steps=int(h.test_epoch_step * h.validation_split))
    except KeyboardInterrupt as e:
        pass

    if is_prune == 'True':
        final_model = sparsity.strip_pruning(train_model)
        prune_ckpt = log_dir / 'yolo_prune_model.h5'
        keras.models.save_model(yolo_model, str(prune_ckpt), include_optimizer=False)
        print()
        print(INFO, f' Save Pruned Model as {str(prune_ckpt)}')
    else:
        keras.models.save_model(yolo_model, str(ckpt))
        print()
        print(INFO, f' Save Model as {str(ckpt)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, help='trian file lists', default='voc')
    parser.add_argument('--class_num', type=int, help='trian class num', default=20)
    parser.add_argument('--pre_ckpt', type=str, help='pre-train model weights', default='None')
    parser.add_argument('--model_def', type=str, help='Model definition.', default='yolo_mobilev2')
    parser.add_argument('--depth_multiplier', type=float, help='mobilenet depth_multiplier', choices=[0.5, 0.75, 1.0], default=1.0)
    parser.add_argument('--augmenter', type=str, help='use image augmenter', choices=['True', 'False'], default='True')
    parser.add_argument('--image_size', type=int, help='net work input image size', default=(224, 320), nargs='+')
    parser.add_argument('--output_size', type=int, help='net work output image size', default=(7, 10, 14, 20), nargs='+')
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--rand_seed', type=int, help='random seed', default=6)
    parser.add_argument('--max_nrof_epochs', type=int, help='epoch num', default=10)
    parser.add_argument('--init_learning_rate', type=float, help='init learning rate', default=0.001)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='learning rate decay factor', default=0)
    parser.add_argument('--obj_weight', type=float, help='obj loss weight', default=5.0)
    parser.add_argument('--noobj_weight', type=float, help='noobj loss weight', default=0.5)
    parser.add_argument('--wh_weight', type=float, help='wh loss weight', default=0.5)
    parser.add_argument('--obj_thresh', type=float, help='obj mask thresh', default=0.7)
    parser.add_argument('--iou_thresh', type=float, help='iou mask thresh', default=0.3)
    parser.add_argument('--vaildation_split', type=float, help='vaildation split factor', default=0.1)
    parser.add_argument('--log_dir', type=str, help='log dir', default='log')
    parser.add_argument('--is_prune', type=str, help='whether to prune model ', choices=['True', 'False'], default='False')
    parser.add_argument('--prune_initial_sparsity', type=float, help='prune initial sparsity range = [0 ~ 1]', default=0.5)
    parser.add_argument('--prune_final_sparsity', type=float, help='prune final sparsity range = [0 ~ 1]', default=0.9)
    parser.add_argument('--prune_end_epoch', type=int, help='prune epochs NOTE: must < train epochs', default=5)
    parser.add_argument('--prune_frequency', type=int, help='how many steps for prune once', default=100)

    args = parser.parse_args(sys.argv[1:])

    main(args, args.train_set, args.class_num, args.pre_ckpt,
         args.model_def, args.depth_multiplier, args.augmenter,
         args.image_size, args.output_size, args.batch_size, args.rand_seed, args.max_nrof_epochs,
         args.init_learning_rate, args.learning_rate_decay_factor,
         args.obj_weight, args.noobj_weight, args.wh_weight, args.obj_thresh, args.iou_thresh, args.vaildation_split,
         args.log_dir,
         args.is_prune,
         args.prune_initial_sparsity,
         args.prune_final_sparsity,
         args.prune_end_epoch,
         args.prune_frequency)
