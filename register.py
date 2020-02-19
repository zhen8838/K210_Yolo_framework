from models.networks import mbv1_facerec, mbv2_ctdet, yolo, tiny_yolo, pfld,\
    shuffle_ctdet, yolo3_nano, yolo_mbv1, mbv1_imgnet, mbv2_imgnet,\
    retinafacenet, retinaface_slim, retinaface_rfb, ullfd_slim
from models.receptivefieldnet import rffacedetnet
from models.audionet import dualmbv2net
from models.networks4k210 import yolo_mbv1_k210, yolo_mbv2_k210, yolo2_mbv1_k210,\
    yolov2algin_mbv1_k210, pfld_k210, mbv1_imgnet_k210, \
    mbv2_imgnet_k210, yoloalgin_mbv1_k210, retinafacenet_k210,\
    retinafacenet_k210_v1, retinafacenet_k210_v2, retinafacenet_k210_v3,\
    ullfd_k210, ullfd_k210_v1, ullfd_k210_v2, ullfd_k210_v3, mbv1_facerec_k210
import tensorflow as tf
from tools.custom import StepLR, CosineLR
from tools.yolo import YOLOHelper, YOLOLoss, yolo_infer, yolo_eval, MultiScaleTrain, YOLOIouLoss, YOLOMap
from tools.yoloalign import YOLOAlignHelper, YOLOAlignLoss, yoloalgin_infer
from tools.pfld import PFLDHelper, PFLDLoss, pfld_infer
from tools.ctdet import CtdetHelper, CtdetLoss, ctdet_infer
from tools.lffd import LFFDHelper, LFFDLoss
from tools.ssd import SSDHelper, SSDLoss, ssd_infer
from tools.retinaface import RetinaFaceHelper, RetinaFaceLoss, retinaface_infer
from tools.tinyimgnet import TinyImgnetHelper
from tools.imgnet import ImgnetHelper, ClassifyLoss
from tools.facerec import FcaeRecHelper, TripletLoss, Sparse_SoftmaxLoss, Sparse_AmsoftmaxLoss, Sparse_AsoftmaxLoss, FacerecValidation, facerec_eval
from tools.dcasetask2 import DCASETask2Helper, SemiBCELoss, LwlrapValidation
from yaml import safe_dump


class dict2obj(object):
    def __init__(self, dicts):
        """ convert dict to object , NOTE the `**kwargs` will not be convert 

        Parameters
        ----------
        object : [type]

        dicts : dict
            dict
        """
        for name, value in dicts.items():
            if isinstance(value, (list, tuple)):
                setattr(self, name, [dict2obj(x) if isinstance(x, dict) else x for x in value])
            else:
                if 'kwarg' in name:
                    setattr(self, name, value if value else dict())
                else:
                    if isinstance(value, dict):
                        setattr(self, name, dict2obj(value))
                    else:
                        setattr(self, name, value)


ArgDict = {
    'mode': 'train',

    # MODEL
    'model': {
        'name': 'yolo',

        'helper': 'YOLOHelper',
        'helper_kwarg': {
            'image_ann': 'data/voc_img_ann.npy',
            'class_num': 20,
            'anchors': 'data/voc_anchor.npy',
            'in_hw': [224, 320],
            'out_hw': [[7, 10], [14, 20]],
            'validation_split': 0.3,  # vaildation_split
        },

        'network': 'yolo_mbv2_k210',
        'network_kwarg': {
            'input_shape': [224, 320, 3],
            'anchor_num': 3,
            'class_num': 20,
            'alpha': 0.75  # depth_multiplier
        },


        'loss': 'YOLOLoss',
        'loss_kwarg': {
            'obj_thresh': 0.7,
            'iou_thresh': 0.5,
            'obj_weight': 1,
            'noobj_weight': 1,
            'wh_weight': 1,
        }
    },

    'train': {
        'jit': True,
        'augmenter': False,
        'batch_size': 16,
        'pre_ckpt': None,
        'rand_seed': 10101,
        'epochs': 10,
        'log_dir': 'log',
        'sub_log_dir': None,
        'debug': False,
        'verbose': 1,
        'vali_step_factor': 0.5,
        'optimizer': 'RAdam',
        'optimizer_kwarg': {
            'lr': 0.001,  # init_learning_rate
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': None,
            'decay': 0.  # learning_rate_decay_factor
        },
        'Lookahead': True,
        'Lookahead_kwarg': {
            'k': 5,
            'alpha': 0.5,
        },
        'earlystop': True,
        'earlystop_kwarg': {
            'monitor': 'val_loss',
            'min_delta': 0,
            'patience': 4,
            'verbose': 0,
            'mode': 'auto',
            'baseline': None,
            'restore_best_weights': False,
        },
        'modelcheckpoint': True,
        'modelcheckpoint_kwarg': {
            'monitor': 'val_loss',
            'verbose': 0,
            'save_best_only': True,
            'save_weights_only': False,
            'mode': 'auto',
            'save_freq': 'epoch',
            'load_weights_on_restart': False,
        },
    },

    'prune': {
        'is_prune': False,
        'init_sparsity': 0.5,  # prune initial sparsity range = [0 ~ 1]
        'final_sparsity': 0.9,  # prune final sparsity range = [0 ~ 1]
        'end_epoch': 5,  # prune epochs NOTE: must < train epochs
        'frequency': 100,  # how many steps for prune once
    },

    'inference': {
        'infer_fn': 'yolo_infer',
        'infer_fn_kwarg': {
            'obj_thresh': .7,
            'iou_thresh': .3
        },
    },
    'evaluate': {
        'eval_fn': 'yolo_eval',
        'eval_fn_kwarg': {
            'det_obj_thresh': 0.1,
            'det_iou_thresh': 0.3,
            'mAp_iou_thresh': 0.3
        },
    }
}


helper_register = {
    'YOLOHelper': YOLOHelper,
    'YOLOAlignHelper': YOLOAlignHelper,
    'PFLDHelper': PFLDHelper,
    'CtdetHelper': CtdetHelper,
    'FcaeRecHelper': FcaeRecHelper,
    'LFFDHelper': LFFDHelper,
    'TinyImgnetHelper': TinyImgnetHelper,
    'ImgnetHelper': ImgnetHelper,
    'RetinaFaceHelper': RetinaFaceHelper,
    'DCASETask2Helper': DCASETask2Helper,
    'SSDHelper': SSDHelper,
}


network_register = {
    'mbv1_facerec': mbv1_facerec,
    'mbv1_facerec_k210': mbv1_facerec_k210,
    'mbv2_ctdet': mbv2_ctdet,
    'mbv1_imgnet': mbv1_imgnet,
    'mbv2_imgnet': mbv2_imgnet,
    'yolo': yolo,
    'tiny_yolo': tiny_yolo,
    'yolo3_nano': yolo3_nano,
    'yolo_mbv1': yolo_mbv1,
    'pfld': pfld,
    'rffacedetnet': rffacedetnet,
    'retinafacenet': retinafacenet,
    'shuffle_ctdet': shuffle_ctdet,
    'yolo_mbv1_k210': yolo_mbv1_k210,
    'yolo_mbv2_k210': yolo_mbv2_k210,
    'yolo2_mbv1_k210': yolo2_mbv1_k210,
    'yolov2algin_mbv1_k210': yolov2algin_mbv1_k210,
    'yoloalgin_mbv1_k210': yoloalgin_mbv1_k210,
    'pfld_k210': pfld_k210,
    'mbv1_imgnet_k210': mbv1_imgnet_k210,
    'mbv2_imgnet_k210': mbv2_imgnet_k210,
    'retinafacenet_k210': retinafacenet_k210,
    'retinafacenet_k210_v1': retinafacenet_k210_v1,
    'retinafacenet_k210_v2': retinafacenet_k210_v2,
    'retinafacenet_k210_v3': retinafacenet_k210_v3,
    'retinaface_slim': retinaface_slim,
    'ullfd_slim': ullfd_slim,
    'ullfd_k210': ullfd_k210,
    'ullfd_k210_v1': ullfd_k210_v1,
    'ullfd_k210_v2': ullfd_k210_v2,
    'ullfd_k210_v3': ullfd_k210_v3,
    'dualmbv2net': dualmbv2net
}

loss_register = {
    'YOLOLoss': YOLOLoss,
    'YOLOIouLoss': YOLOIouLoss,
    'YOLOAlignLoss': YOLOAlignLoss,
    'PFLDLoss': PFLDLoss,
    'CtdetLoss': CtdetLoss,
    'TripletLoss': TripletLoss,
    'Sparse_SoftmaxLoss': Sparse_SoftmaxLoss,
    'Sparse_AmsoftmaxLoss': Sparse_AmsoftmaxLoss,
    'Sparse_AsoftmaxLoss': Sparse_AsoftmaxLoss,
    'LFFDLoss': LFFDLoss,
    'ClassifyLoss': ClassifyLoss,
    'RetinaFaceLoss': RetinaFaceLoss,
    'SemiBCELoss': SemiBCELoss,
    'SSDLoss': SSDLoss
}

callback_register = {
    'MultiScaleTrain': MultiScaleTrain,
    'EarlyStopping': tf.keras.callbacks.EarlyStopping,
    'ModelCheckpoint': tf.keras.callbacks.ModelCheckpoint,
    'TerminateOnNaN': tf.keras.callbacks.TerminateOnNaN,
    'YOLOMap': YOLOMap,
    'StepLR': StepLR,
    'CosineLR': CosineLR,
    'FacerecValidation': FacerecValidation,
    'LwlrapValidation': LwlrapValidation
}

optimizer_register = {
    'Adam': tf.keras.optimizers.Adam,
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam,
    'Ftrl': tf.keras.optimizers.Ftrl,
}

infer_register = {
    'yolo_infer': yolo_infer,
    'yoloalgin_infer': yoloalgin_infer,
    'pfld_infer': pfld_infer,
    'ctdet_infer': ctdet_infer,
    'retinaface_infer': retinaface_infer,
    'ssd_infer': ssd_infer
}

eval_register = {
    'yolo_eval': yolo_eval,
    'facerec_eval': facerec_eval
}

if __name__ == "__main__":
    with open('config/default.yml', 'w') as f:
        safe_dump(ArgDict, f, sort_keys=False)
