import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
from tools.utils import Helper, YOLO_Loss, INFO, ERROR, NOTE
from tools.alignutils import YOLOAlignHelper, YOLOAlign_Loss
from tools.landmarkutils import LandmarkHelper, LandMark_Loss
from tools.custom import Yolo_P_R, Lookahead, PFLDMetric, YOLO_LE
from models.yolonet import yolo_mobilev2, pfld
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import sys
import argparse
from termcolor import colored
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
from register import dict2obj, network_register, optimizer_register, helper_register, loss_register
from yaml import safe_dump, safe_load
from tensorflow.python import debug as tfdebug


def main(config_file, new_cfg, mode, model, train, prune):
    """ config tensorflow backend """
    tf.reset_default_graph()
    tfcfg = tf.ConfigProto()
    tfcfg.gpu_options.allow_growth = True
    sess = tf.Session(config=tfcfg)

    if train.debug == True:
        sess = tfdebug.LocalCLIDebugWrapperSession(sess)

    keras.backend.set_session(sess)

    """ Set Golbal Paramter """
    tf.set_random_seed(train.rand_seed)
    np.random.seed(train.rand_seed)
    initial_epoch = 0
    log_dir = (Path(train.log_dir) / datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S'))  # type: Path

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    with (log_dir / Path(config_file).name).open('w') as f:
        safe_dump(new_cfg, f, sort_keys=False)  # save config file name

    """ Build Data Input PipeLine """

    h = helper_register[model.helper](**model.helper_kwarg)  # type: Helper
    h.set_dataset(train.batch_size, train.rand_seed, is_training=train.augmenter)

    train_ds = h.train_dataset
    validation_ds = h.test_dataset
    train_epoch_step = h.train_epoch_step
    vali_epoch_step = h.test_epoch_step * train.vali_step_factor

    """ Build Network """

    if 'yolo' in model.name:
        network = network_register[model.network]  # type:yolo_mobilev2
        saved_model, trainable_model = network(model.helper_kwarg['in_hw'] + [3],
                                               len(h.anchors[0]), model.helper_kwarg['class_num'],
                                               **model.network_kwarg)
    elif model.name == 'pfld':
        network = network_register[model.network]  # type:pfld
        pflp_infer_model, auxiliary_model, trainable_model = network(
            model.helper_kwarg['in_hw'] + [3], **model.network_kwarg)
        saved_model = trainable_model

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
    if 'yolo' in model.name:
        loss_obj = loss_register[model.loss]  # type:YOLOAlign_Loss
        losses = [loss_obj(h=h, layer=layer, name='loss', **model.loss_kwarg)
                  for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)]  # type:[YOLOAlign_Loss]

        metrics = []
        for i in range(len(losses)):
            precision = Yolo_P_R(0, model.loss_kwarg['obj_thresh'], name='p', dtype=tf.float32)
            recall = Yolo_P_R(1, model.loss_kwarg['obj_thresh'], name='r', dtype=tf.float32)
            recall.tp, recall.fn = precision.tp, precision.fn  # share the variable avoid more repeated calculation
            metrics.append([precision, recall])

        if model.name == 'yoloalign':
            for i, m in enumerate(metrics):
                m.append(YOLO_LE(losses[i].landmark_error))

    elif model.name == 'pfld':
        loss_obj = loss_register[model.loss]  # type:LandMark_Loss
        losses = [loss_obj(h=h, **model.loss_kwarg)]
        # NOTE share the variable avoid more repeated calculation
        le = PFLDMetric(False, model.helper_kwarg['landmark_num'], train.batch_size, name='LE', dtype=tf.float32)
        fr = PFLDMetric(True, model.helper_kwarg['landmark_num'], train.batch_size, name='FR', dtype=tf.float32)
        fr.failure_num, fr.total = le.failure_num, le.total
        metrics = [le, fr]

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    train_model.compile(optimizer, loss=losses, metrics=metrics)

    """ Load Pre-Train Model Weights """
    if train.pre_ckpt != None and train.pre_ckpt != 'None' and train.pre_ckpt != '':
        if 'h5' in train.pre_ckpt:
            initial_epoch = int(Path(train.pre_ckpt).stem.split('_')[-1]) + 1
            trainable_model.load_weights(str(train.pre_ckpt))
            print(INFO, f' Load CKPT {str(train.pre_ckpt)}')
        else:
            print(ERROR, ' Pre CKPT path is unvalid')

    if train.Lookahead == True:
        lookahead = Lookahead(**train.Lookahead_kwarg)  # init Lookahead
        lookahead.inject(train_model)  # inject to model

    """ Callbacks """
    cbs = []
    if prune.is_prune == True:
        cbs += [sparsity.UpdatePruningStep(),
                sparsity.PruningSummaries(log_dir=str(log_dir), profile_batch=0)]

    cbs.append(TensorBoard(str(log_dir), update_freq='batch', profile_batch=3))
    if train.earlystop == True:
        cbs.append(EarlyStopping(**train.earlystop_kwarg))

    file_writer = tf.summary.FileWriter(str(log_dir), sess.graph)  # NOTE avoid can't write graph, I don't now why..

    """ Start Training """
    try:
        train_model.fit(train_ds, epochs=train.epochs, steps_per_epoch=train_epoch_step, callbacks=cbs,
                        validation_data=validation_ds, validation_steps=vali_epoch_step,
                        verbose=train.verbose,
                        initial_epoch=initial_epoch)
    except KeyboardInterrupt as e:
        pass

    """ Finish Training """
    model_name = f'saved_model_{initial_epoch+int(train_model.optimizer.iterations.eval(sess) / train_epoch_step)}.h5'
    ckpt = log_dir / model_name

    if prune.is_prune == True:
        final_model = sparsity.strip_pruning(train_model)
        prune_ckpt = log_dir / 'prune' + model_name
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
