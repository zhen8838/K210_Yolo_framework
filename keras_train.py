import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler
from tools.utils import Helper, create_yolo_loss, INFO, ERROR, NOTE
from tools.alignutils import YOLOAlignHelper, create_yoloalign_loss
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
from register import dict2obj, network_register, optimizer_register
from yaml import safe_dump, safe_load

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


def main(config_file, new_cfg, mode, model, train, prune):
    """ Set Golbal Paramter """
    tf.set_random_seed(train.rand_seed)
    np.random.seed(train.rand_seed)

    log_dir = (Path(train.log_dir) / datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S'))  # type: Path
    ckpt = log_dir / 'saved_model.h5'

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    with (log_dir / Path(config_file).name).open('w') as f:
        safe_dump(new_cfg, f, sort_keys=False)  # save config file name

    """ Build Data Input PipeLine """
    if model.name == 'yolo':
        h = Helper(f'data/{train.dataset}_img_ann.npy', model.class_num, f'data/{train.dataset}_anchor.npy',
                   model.input_hw, np.reshape(np.array(model.output_hw), (-1, 2)), train.vail_split)
        h.set_dataset(train.batch_size, train.rand_seed, is_training=train.augmenter)
    elif model.name == 'yoloalgin':
        h = YOLOAlignHelper(f'data/{train.dataset}_img_ann.npy', model.class_num, f'data/{train.dataset}_anchor.npy',
                            model.landmark_num, model.input_hw, np.reshape(np.array(model.output_hw), (-1, 2)),
                            train.vail_split)
        h.set_dataset(train.batch_size, train.rand_seed, is_training=train.augmenter)

    train_ds = h.train_dataset
    validation_ds = h.test_dataset
    train_epoch_step = h.train_epoch_step
    vali_epoch_step = int(h.train_epoch_step * h.validation_split)

    """ Build Network """
    network = network_register[model.network]  # type :yolo_mobilev2
    if model.name == 'yolo':
        saved_model, trainable_model = network([model.input_hw[0], model.input_hw[1], 3],
                                               len(h.anchors[0]), model.class_num,
                                               alpha=model.depth_multiplier)
    elif model.name == 'yoloalgin':
        saved_model, trainable_model = network([model.input_hw[0], model.input_hw[1], 3],
                                               len(h.anchors[0]), model.class_num, model.landmark_num,
                                               alpha=model.depth_multiplier)

    """ Load Pre-Train Model """
    if train.pre_ckpt != None and train.pre_ckpt != 'None' and train.pre_ckpt != '':
        if 'h5' in train.pre_ckpt:
            trainable_model.load_weights(str(train.pre_ckpt))
            print(INFO, f' Load CKPT {str(train.pre_ckpt)}')
        else:
            print(ERROR, ' Pre CKPT path is unvalid')

    """  Config Prune Model Paramter """
    if prune.is_prune == 'True':
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=prune.init_sparsity,
                                                         final_sparsity=prune.final_sparsity,
                                                         begin_step=0,
                                                         end_step=train_epoch_step * prune.end_epoch,
                                                         frequency=prune.frequency)
        }
        train_model = sparsity.prune_low_magnitude(trainable_model, **pruning_params)
    else:
        train_model = trainable_model

    """ Comile Model """
    optimizer = optimizer_register[train.optimizer](**train.optimizer_kwarg)
    if model.name == 'yolo':
        losses = [create_yolo_loss(h, model.obj_thresh, model.iou_thresh, model.obj_weight, model.noobj_weight, model.wh_weight, layer)
                  for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)]
        metrics = [Yolo_Precision(model.obj_thresh, name='p'), Yolo_Recall(model.obj_thresh, name='r')]
    elif model.name == 'yoloalgin':
        losses = [create_yoloalign_loss(h, model.obj_thresh, model.iou_thresh, model.obj_weight, model.noobj_weight, model.wh_weight, layer)
                  for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)]
        metrics = [Yolo_Precision(model.obj_thresh, name='p'), Yolo_Recall(model.obj_thresh, name='r')]

    train_model.compile(optimizer, loss=losses, metrics=metrics)

    """ Callbacks """
    if prune.is_prune == True:
        cbs = [
            sparsity.UpdatePruningStep(),
            sparsity.PruningSummaries(log_dir=str(log_dir), profile_batch=0)]
    else:
        cbs = [TensorBoard(str(log_dir), update_freq='batch', profile_batch=3)]

    """ Start Training """
    try:
        train_model.fit(train_ds, epochs=train.epochs, steps_per_epoch=train_epoch_step, callbacks=cbs,
                        validation_data=validation_ds, validation_steps=vali_epoch_step)
    except KeyboardInterrupt as e:
        pass

    """ Finish Training """
    if prune.is_prune == True:
        final_model = sparsity.strip_pruning(train_model)
        prune_ckpt = log_dir / 'yolo_prune_model.h5'
        keras.models.save_model(saved_model, str(prune_ckpt), include_optimizer=False)
        print()
        print(INFO, f' Save Pruned Model as {str(prune_ckpt)}')
    else:
        keras.models.save_model(saved_model, str(ckpt))
        print()
        print(INFO, f' Save Model as {str(ckpt)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path', default='config/default.yml')
    args = parser.parse_args(sys.argv[1:])

    with open(args.config_file, 'r') as f:
        new_cfg = safe_load(f)

    ArgMap = dict2obj(new_cfg)
    main(args.config_file, new_cfg, ArgMap.mode, ArgMap.model, ArgMap.train, ArgMap.prune)
