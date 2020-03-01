import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from pathlib import Path
from register import dict2obj, network_register, optimizer_register,\
    helper_register, callback_register, trainloop_register
from tools.training_engine import BaseTrainingLoop
import numpy as np
from datetime import datetime
from yaml import safe_dump, safe_load
from tools.custom import SignalStopping
from tools.base import ERROR, INFO, NOTE
import argparse


def main(config_file, new_cfg, mode, model, train):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  if train.graph_optimizer is True:
    tf.config.optimizer.set_experimental_options(train.graph_optimizer_kwarg)
  """ Set Golbal Paramter """
  tf.random.set_seed(train.rand_seed)
  np.random.seed(train.rand_seed)
  log_dir = (Path(train.log_dir) /
             (datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S')
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
  train_epoch_step = int(h.train_epoch_step)
  vali_epoch_step = int(h.val_epoch_step)

  network = network_register[model.network]
  infer_model, val_model, train_model = network(**model.network_kwarg)

  optimizer = optimizer_register[train.optimizer](**train.optimizer_kwarg)
  train_model.compile(optimizer, loss=lambda x, y: 0.)

  loop: BaseTrainingLoop = trainloop_register[train.trainloop](
      train_model, val_model, **train.trainloop_kwarg)
  loop.set_dataset(train_ds, validation_ds, train_epoch_step, vali_epoch_step)

  cbs = [
      SignalStopping(),
      tf.keras.callbacks.CSVLogger(str(log_dir / 'training.csv'), '\t', True)
  ]

  for cbkparam in train.callbacks:
    cbk_fn = callback_register[cbkparam.name]
    if cbkparam.name == 'ModelCheckpoint':
      cbs.append(
          cbk_fn(str(log_dir / 'auto_train_{epoch:d}.h5'), **cbkparam.kwarg))
    elif cbkparam.name == 'AugmenterStateSync':
      cbs.append(cbk_fn(h.augmenter, **cbkparam.kwarg))
      loop.set_augmenter(h.augmenter)
    else:
      cbs.append(cbk_fn(**cbkparam.kwarg))

  loop.set_callbacks(cbs)
  loop.set_summary_writer(
      tf.summary.create_file_writer(
          str(log_dir / datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S'))))
  """ Load Pre-Train Model Weights """
  initial_epoch = 0
  if train.pre_ckpt != None and train.pre_ckpt != 'None' and train.pre_ckpt != '':
    if 'h5' in train.pre_ckpt:
      initial_epoch = int(Path(train.pre_ckpt).stem.split('_')[-1]) + 1
      train_model.load_weights(
          str(train.pre_ckpt), by_name=True, skip_mismatch=True)
      print(INFO, f' Load CKPT {str(train.pre_ckpt)}')
    else:
      print(ERROR, ' Pre CKPT path is unvalid')

  loop.train_and_eval(
      epochs=train.epochs + initial_epoch,
      initial_epoch=initial_epoch,
      steps_per_run=train.steps_per_run)
  """ Finish Training """
  model_name = f'train_model_{initial_epoch+int(train_model.optimizer.iterations / train_epoch_step)}.h5'
  ckpt = log_dir / model_name

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
  auto_saved_list = list(
      zip(auto_saved_list, [int(p.stem.split('_')[-1]) for p in auto_saved_list]))
  if len(auto_saved_list) > 0:
    auto_saved_list.sort(key=lambda x: x[1])
    train_model.load_weights(str(auto_saved_list[-1][0]), by_name=True)
    infer_ckpt = str(auto_saved_list[-1][0]).replace('train', 'infer')
    k.models.save_model(infer_model, infer_ckpt)
  print(INFO, f' Save Best Infer Model as {infer_ckpt}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config_file',
      type=str,
      help='config file path',
      default='config/default.yml')
  args = parser.parse_args()

  with open(args.config_file, 'r') as f:
    new_cfg = safe_load(f)

  ArgMap = dict2obj(new_cfg)
  main(args.config_file, new_cfg, ArgMap.mode, ArgMap.model, ArgMap.train)
