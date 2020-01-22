import tensorflow as tf
k = tf.keras
kcall = tf.keras.callbacks
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy
from tools.base import INFO, ERROR, NOTE
from tools.custom import Lookahead, PFLDMetric, DummyMetric, DummyOnceMetric, SignalStopping
from tools.base import BaseHelper
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
from register import dict2obj, network_register, optimizer_register,\
    helper_register, loss_register, callback_register
from yaml import safe_dump, safe_load
from tensorflow.python import debug as tfdebug
from typing import List


def main(config_file, new_cfg, mode, model, train, prune):
    """ config tensorflow backend """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if train.graph_optimizer is True:
        tf.config.optimizer.set_experimental_options(train.graph_optimizer_kwarg)

    if train.debug == True:
        # sess = tfdebug.LocalCLIDebugWrapperSession(sess)
        pass

    """ Set Golbal Paramter """
    tf.random.set_seed(train.rand_seed)
    np.random.seed(train.rand_seed)
    initial_epoch = 0
    log_dir = (Path(train.log_dir) / (datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S')
                                      if train.sub_log_dir is None else train.sub_log_dir))  # type: Path

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    with (log_dir / Path(config_file).name).open('w') as f:
        safe_dump(new_cfg, f, sort_keys=False)  # save config file name

    """ Build Data Input PipeLine """

    h = helper_register[model.helper](**model.helper_kwarg)  # type:BaseHelper
    h.set_dataset(train.batch_size, train.augmenter)

    train_ds = h.train_dataset
    validation_ds = h.val_dataset
    train_epoch_step = h.train_epoch_step
    vali_epoch_step = int(h.val_epoch_step * train.vali_step_factor)

    """ Build Network """
    if model.name == 'feacrec':
        network = network_register[model.network]
        infer_model, val_model, train_model = network(**model.network_kwarg)
    else:
        network = network_register[model.network]
        infer_model, train_model = network(**model.network_kwarg)

    """  Config Prune Model Paramter """
    if prune.is_prune == 'True':
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=prune.init_sparsity,
                                                         final_sparsity=prune.final_sparsity,
                                                         begin_step=0,
                                                         end_step=train_epoch_step * prune.end_epoch,
                                                         frequency=prune.frequency)
        }
        train_model = sparsity.prune_low_magnitude(train_model, **pruning_params)  # type:k.Model

    """ Comile Model, Set loss and metric """
    optimizer = optimizer_register[train.optimizer](**train.optimizer_kwarg)
    metrics: List[k.metrics.Metric] = []
    if 'yolo' in model.name:
        loss_fn = loss_register[model.loss]  # type:YOLOAlignLoss
        losses = [loss_fn(h=h, layer=layer, name='loss', **model.loss_kwarg)
                  for layer in range(len(train_model.output) if isinstance(train_model.output, list) else 1)]  # type: List[YOLOAlignLoss]

        for loss_obj in losses:
            l = [DummyMetric(var, name) for (var, name) in loss_obj.lookups]
            metrics.append(l)
    elif model.name == 'retinaface':
        loss_obj = loss_register[model.loss](h=h, **model.loss_kwarg)
        losses = [loss_obj]
        l = [DummyMetric(var, name) for (var, name) in loss_obj.lookups]
        metrics.append(l)

    elif model.name == 'pfld':
        loss_fn = loss_register[model.loss]  # type:PFLDLoss
        losses = [loss_fn(h=h, **model.loss_kwarg)]
        # NOTE share the variable avoid more repeated calculation
        le = PFLDMetric(False, model.helper_kwarg['landmark_num'],
                        train.batch_size, name='LE', dtype=tf.float32)
        fr = PFLDMetric(True, model.helper_kwarg['landmark_num'],
                        train.batch_size, name='FR', dtype=tf.float32)
        fr.failure_num, fr.total = le.failure_num, le.total
        metrics = [le, fr]
    elif model.name == 'feacrec':
        loss_obj = loss_register[model.loss](**model.loss_kwarg)
        losses = [loss_obj]
        if model.helper_kwarg['use_softmax'] == True:
            metrics = [SparseCategoricalAccuracy(name='acc')]
        else:
            metrics = [DummyMetric(loss_obj.triplet_acc, name='acc')]
        vali_acc_var = tf.Variable(0., False, dtype=tf.float32)
        metrics.append(DummyOnceMetric(vali_acc_var, 'val_acc'))
    elif model.name == 'lffd':
        loss_fn = loss_register[model.loss]
        losses = [loss_fn(h=h, **model.loss_kwarg) for i in range(h.scale_num)]
    elif model.name in 'imagenet':
        loss_fn = loss_register[model.loss]
        losses = [loss_fn(**model.loss_kwarg)]
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        loss_obj = loss_register[model.loss](h=h, **model.loss_kwarg)
        losses = [loss_obj]

    train_model.compile(optimizer, loss=losses, metrics=metrics)

    """ Load Pre-Train Model Weights """
    if train.pre_ckpt != None and train.pre_ckpt != 'None' and train.pre_ckpt != '':
        if 'h5' in train.pre_ckpt:
            initial_epoch = int(Path(train.pre_ckpt).stem.split('_')[-1]) + 1
            train_model.load_weights(str(train.pre_ckpt))
            print(INFO, f' Load CKPT {str(train.pre_ckpt)}')
        else:
            print(ERROR, ' Pre CKPT path is unvalid')

    if train.Lookahead == True:
        lookahead = Lookahead(**train.Lookahead_kwarg)  # init Lookahead
        lookahead.inject(train_model)  # inject to model

    """ Callbacks """
    cbs = [SignalStopping(),
           kcall.TensorBoard(str(log_dir), update_freq='batch', profile_batch=3),
           kcall.CSVLogger(str(log_dir / 'training.csv'), '\t', True)]

    if prune.is_prune == True:
        cbs += [sparsity.UpdatePruningStep(),
                sparsity.PruningSummaries(log_dir=str(log_dir), profile_batch=0)]

    for cbkparam in train.callbacks:
        cbk_fn = callback_register[cbkparam.name]
        if cbkparam.name == 'ModelCheckpoint':
            cbs.append(cbk_fn(str(log_dir / 'auto_train_{epoch:d}.h5'), **cbkparam.kwarg))
        elif cbkparam.name == 'MultiScaleTrain':
            cbs.append(cbk_fn(h, **cbkparam.kwarg))
        elif cbkparam.name == 'TerminateOnNaN':
            cbs.append(cbk_fn())
        elif cbkparam.name == 'FacerecValidation':
            cbs.append(cbk_fn(val_model, validation_ds, vali_epoch_step, train_epoch_step, vali_acc_var, **cbkparam.kwarg))
        else:
            cbs.append(cbk_fn(**cbkparam.kwarg))

    """ Start Training """
    if model.name == 'feacrec':
        # NOTE when use facerec , need manual validation
        train_model.fit(train_ds, epochs=initial_epoch + train.epochs,
                        steps_per_epoch=train_epoch_step, callbacks=cbs,
                        verbose=train.verbose,
                        initial_epoch=initial_epoch)
    else:
        train_model.fit(train_ds, epochs=initial_epoch + train.epochs,
                        steps_per_epoch=train_epoch_step, callbacks=cbs,
                        validation_data=validation_ds, validation_steps=vali_epoch_step,
                        verbose=train.verbose,
                        initial_epoch=initial_epoch)

    """ Finish Training """
    model_name = f'train_model_{initial_epoch+int(train_model.optimizer.iterations / train_epoch_step)}.h5'
    ckpt = log_dir / model_name

    if prune.is_prune == True:
        final_model = sparsity.strip_pruning(train_model)
        prune_ckpt = log_dir / 'prune' + model_name
        k.models.save_model(train_model, str(prune_ckpt), include_optimizer=False)
        print()
        print(INFO, f' Save Pruned Model as {str(prune_ckpt)}')
    else:
        k.models.save_model(train_model, str(ckpt))
        print()
        print(INFO, f' Save Train Model as {str(ckpt)}')

        infer_model_name = f'infer_model_{initial_epoch+int(train_model.optimizer.iterations / train_epoch_step)}.h5'
        infer_ckpt = log_dir / infer_model_name
        k.models.save_model(infer_model, str(infer_ckpt))
        print(INFO, f' Save Infer Model as {str(infer_ckpt)}')

        # find best auto saved model, and save best infer model
        auto_saved_list = list(log_dir.glob('auto_train_*.h5'))  # type:List[Path]
        # use `int value`  for sort ~
        auto_saved_list = list(zip(auto_saved_list, [int(p.stem.split('_')[-1]) for p in auto_saved_list]))
        if len(auto_saved_list) > 0:
            auto_saved_list.sort(key=lambda x: x[1])
            train_model.load_weights(str(auto_saved_list[-1][0]), by_name=True)
            infer_ckpt = str(auto_saved_list[-1][0]).replace('train', 'infer')
            k.models.save_model(infer_model, infer_ckpt)
        print(INFO, f' Save Best Infer Model as {infer_ckpt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path', default='config/default.yml')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        new_cfg = safe_load(f)

    ArgMap = dict2obj(new_cfg)
    main(args.config_file, new_cfg, ArgMap.mode, ArgMap.model, ArgMap.train, ArgMap.prune)
