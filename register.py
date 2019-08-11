from models.yolonet import yolo_mobilev1, yolo_mobilev2, tiny_yolo, yolo
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop


class dict_to_obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict_to_obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict_to_obj(b) if isinstance(b, dict) else b)


ArgDict = {
    'mode': 'train',

    # MODEL
    'model': {
        'name': 'yolo',

        'network': 'yolo_mobilev2',

        'depth_multiplier': 0.75,

        # net work input image size
        'input_size': [[320, 224]],
        # net work output image size
        'output_size': [[7, 10], [14, 20]],
        # trian class num
        'class_num': 20,

        'landmark_num': 5,  # for yolo alignment

        # NOTE loss weight control
        'obj_weight': 1,
        'noobj_weight': 1,
        'wh_weight': 1,
        'obj_thresh': 0.7,
        'iou_thresh': 0.5,
    },

    'train': {
        'dataset': 'voc',
        'augmenter': False,
        'batch_size': 16,
        'pre_ckpt': None,
        'rand_seed': 10101,
        'epochs': 10,
        'vail_split': 0.1,  # vaildation_split
        'log_dir': 'log',
        # optimizers
        'optimizer': 'adam',
        'optimizer_parm':
        {
            'lr': 0.0001,  # init_learning_rate
            'decay': 0  # learning_rate_decay_factor
        },

    },

    'prune': {
        'is_prune': False,
        'init_sparsity': 0.5,  # prune initial sparsity range = [0 ~ 1]
        'final_sparsity': 0.9,  # prune final sparsity range = [0 ~ 1]
        'end_epoch': 5,  # prune epochs NOTE: must < train epochs
        'frequency': 100,  # how many steps for prune once
    }
}


network_register = {
    'yolo_mobilev1': yolo_mobilev1,
    'yolo_mobilev2': yolo_mobilev2,
    'tiny_yolo': tiny_yolo,
    'yolo': yolo,
}

optimizer_register = {
    'Adam': Adam,
    'SGD': SGD,
    'RMSprop': RMSprop,
}

ArgMap = dict_to_obj(ArgDict)

type(ArgMap)
