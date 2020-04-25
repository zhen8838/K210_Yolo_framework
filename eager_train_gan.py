import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from pathlib import Path
from register import dict2obj, network_register, optimizer_register,\
    helper_register, callback_register, trainloop_register
from tools.training_engine import BaseTrainingLoop, DistributionStrategyHelper
from tools.custom import VariableCheckpoint, LRCallback
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
  datetime_dir = datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S')
  if not log_dir.exists():
    log_dir.mkdir(parents=True)

  with (log_dir / Path(config_file).name).open('w') as f:
    safe_dump(new_cfg, f, sort_keys=False)  # save config file name
  """ Build Data Input PipeLine """
  distribution = DistributionStrategyHelper(**train.distributionstrategy_kwarg)
  strategy_scope = distribution.get_strategy_scope()
  with strategy_scope:
    h = helper_register[model.helper](**model.helper_kwarg)  # type:BaseHelper
    h.set_dataset(train.batch_size, train.augmenter)

    train_ds, validation_ds = distribution.get_strategy_dataset(
        h.train_dataset, h.val_dataset)

    train_epoch_step = train.train_epoch_step if train.train_epoch_step else int(
        h.train_epoch_step)

    vali_epoch_step = train.vali_epoch_step if train.vali_epoch_step else int(
        h.val_epoch_step)
    """ mixed precision policy
    Args:
      mixed_precision : {
        enable (bool): true or false
        dtype (str): `mixed_float16` or `mixed_bfloat16` policy can be used
      } 
    """
    if train.mixed_precision.enable:
      policy = tf.keras.mixed_precision.experimental.Policy(
          train.mixed_precision.dtype)
      tf.keras.mixed_precision.experimental.set_policy(policy)

    network = network_register[model.network]
    generator_model, discriminator_model, val_model = network(
        **model.network_kwarg)

    generator_optimizer: tf.keras.optimizers.Optimizer = optimizer_register[
        train.generator_optimizer](**train.generator_optimizer_kwarg)
    discriminator_optimizer: tf.keras.optimizers.Optimizer = optimizer_register[
        train.discriminator_optimizer](**train.discriminator_optimizer_kwarg)
    if train.mixed_precision.enable:
      generator_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          generator_optimizer, policy.loss_scale)
      discriminator_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          discriminator_optimizer, policy.loss_scale)

    loop: BaseTrainingLoop = trainloop_register[train.trainloop](
        generator_model, discriminator_model, val_model, generator_optimizer,
        discriminator_optimizer, distribution.strategy, **train.trainloop_kwarg)
    loop.set_dataset(train_ds, validation_ds, train_epoch_step, vali_epoch_step)

    cbs = [
        SignalStopping(),
        tf.keras.callbacks.CSVLogger(str(log_dir / 'training.csv'), '\t', True)
    ]
    if train.callbacks:
      for cbkparam in train.callbacks:
        cbk_fn = callback_register[cbkparam.name]
        if cbkparam.name == 'AugmenterStateSync':
          cbs.append(cbk_fn(h.augmenter, **cbkparam.kwarg))
          loop.set_augmenter(h.augmenter)
          need_saved_variable_dict = h.augmenter.get_state()
        else:
          cbk_obj = cbk_fn(**cbkparam.kwarg)
          if isinstance(cbk_obj, LRCallback):
            cbk_obj.set_optimizer(eval(cbk_obj.outside_optimizer))
          cbs.append(cbk_obj)
    """ Load Pre-Train Model Weights """
    # NOTE use eval captrue local variables
    variable_str_dict = train.variablecheckpoint_kwarg.pop('variable_dict')
    variable_dict = {}
    for (key, v) in variable_str_dict.items():
      variable_dict.setdefault(key, eval(v))
    # NOTE get other variables which need to save
    if 'need_saved_variable_dict' in vars().keys():
      variable_dict.update(need_saved_variable_dict)
    print(INFO, 'VariableCheckpoint will save or load: \n',
          ' '.join(variable_dict.keys()))
    variablecheckpoint = VariableCheckpoint(log_dir, variable_dict,
                                            **train.variablecheckpoint_kwarg)
    if train.pre_ckpt and '.h5' in train.pre_ckpt:
      generator_model.load_weights(
          str(train.pre_ckpt), by_name=True, skip_mismatch=True)
      print(INFO, f' Load CKPT {str(train.pre_ckpt)}')
    else:
      variablecheckpoint.load_checkpoint(train.pre_ckpt)
    cbs.append(variablecheckpoint)

    loop.set_callbacks(cbs)
    loop.set_summary_writer(
        str(log_dir), datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S'))
    initial_epoch = int(generator_optimizer.iterations.numpy() / train_epoch_step)

    finally_epoch = loop.train_and_eval(
        epochs=train.epochs + initial_epoch,
        initial_epoch=initial_epoch,
        steps_per_run=train.steps_per_run)
    """ Finish Training """
    loop.save_models(finally_epoch)


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
