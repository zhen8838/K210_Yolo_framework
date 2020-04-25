import tensorflow as tf
from pathlib import Path
from tools.base import INFO, ERROR, NOTE
import argparse
import numpy as np
from register import dict2obj, network_register, optimizer_register, helper_register, infer_register
from yaml import safe_load
import tensorflow_model_optimization as tfmot


def main(ckpt_path: Path, argmap: dict2obj, images_path: Path,
         results_path: Path):
  """ parser main function

    Parameters
    ----------
    ckpt_path : Path
        '*.h5' file path 
    argmap : dict2obj
        argmap 
    images_path : Path
        img path
    results_path : Path
        nncase infer path, 'Path' or None
    """
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  model, train, inference = argmap.model, argmap.train, argmap.inference
  h = helper_register[model.helper](**model.helper_kwarg)

  network = network_register[model.network]
  generator_model, discriminator_model, val_model = network(**model.network_kwarg)

  print(INFO, f'Load CKPT from {str(ckpt_path)}')
  if argmap.quantize.is_quantize == True:
    with tfmot.quantization.keras.quantize_scope():
      q_load_aware_train_model = tf.keras.models.load_model(str(ckpt_path))
  else:
    try:
      ckpt = tf.train.Checkpoint(
          g_model=generator_model, d_model=discriminator_model)
      ckpt.restore(str(ckpt_path))
    except tf.errors.DataLossError as e:
      if ckpt_path.suffix == '.h5':
        generator_model.load_weights(ckpt_path.as_posix(), by_name=True)
      else:
        raise ValueError(ckpt_path.as_posix() + ' is not valid path')

  h.infer(images_path, generator_model, results_path, **inference.infer_fn_kwarg)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config',
      type=str,
      help='config file path, default in same folder with `pre_ckpt`',
      default=None)
  parser.add_argument(
      '--results_path', type=str, help='inference results path', default=None)
  parser.add_argument('pre_ckpt', type=str, help='pre-train weights path')
  parser.add_argument('images_path', type=str, help='test images path')
  args = parser.parse_args()

  pre_ckpt = Path(args.pre_ckpt)

  if args.config == None:
    if pre_ckpt.suffix != '.h5':
      config_path = list(pre_ckpt.parent.parent.glob('*.yml'))[0]  # type: Path
    else:
      config_path = list(pre_ckpt.parent.glob('*.yml'))[0]  # type: Path
  else:
    config_path = Path(args.config)

  if args.results_path == None or args.results_path == 'None':
    args.results_path = None
  else:
    args.results_path = Path(args.results_path)

  with config_path.open('r') as f:
    cfg = safe_load(f)

  ArgMap = dict2obj(cfg)
  if not hasattr(ArgMap, 'quantize'):
    ArgMap.quantize = dict2obj({'is_quantize': False})
  main(Path(args.pre_ckpt), ArgMap, Path(args.images_path), args.results_path)
