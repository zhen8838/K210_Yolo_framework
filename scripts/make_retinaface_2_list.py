import numpy as np
from pathlib import Path
import argparse
import json


def main(root, output_file):
  root = Path(root)

  meta = {}
  for name in ['train', 'val']:
    label_root: Path = (root / name) / 'label'
    img_paths = []
    anns = []
    for json_file in label_root.glob('*.json'):
      with json_file.open('r') as f:
        ss = json.load(f)
        annotation = np.zeros((1, 4 + 8 * 2 + 1))
        for shape in ss['shapes']:
          if shape['shape_type'] == 'rectangle':
            annotation[0, 0] = shape['points'][0][0]  # x1
            annotation[0, 1] = shape['points'][0][1]  # y1
            annotation[0, 2] = shape['points'][1][0]  # x2
            annotation[0, 3] = shape['points'][1][1]  # y2
          if shape['shape_type'] == 'point':
            idx = int(shape['label'][1:]) - 1
            annotation[0, 4 + idx * 2] = shape['points'][0][0]  # l0_x-1
            annotation[0, 5 + idx * 2] = shape['points'][0][1]  # l0_y
        annotation[0, -1] = 1
      anns.append(annotation)
      img_paths.append(
          (label_root / ss['imagePath'].replace('\\', '/')).as_posix())

    meta[name] = np.array([
        (a, b.astype('float32')) for a, b in zip(img_paths, anns)
    ])

  np.save(output_file, meta, allow_pickle=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--root',
      type=str,
      help='retina face dataset root path',
      default='/home/zqh/workspace/retina_dataset')
  parser.add_argument(
      '--output_file',
      type=str,
      help='output file path',
      default='data/retinaface_img_ann.npy')
  args = parser.parse_args()
  main(args.root, args.output_file)
