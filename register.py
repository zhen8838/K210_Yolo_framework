import tensorflow as tf
from yaml import safe_dump
from importlib import import_module


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
        setattr(self, name,
                [dict2obj(x) if isinstance(x, dict) else x for x in value])
      else:
        if 'kwarg' in name:
          setattr(self, name, value if value else dict())
        else:
          if isinstance(value, dict):
            setattr(self, name, dict2obj(value))
          else:
            setattr(self, name, value)

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()


class dict_warp():
  """ dict_warp for dynamic load class or function """
  __slots__ = ('d')

  def __init__(self, d):
    assert isinstance(d, dict)
    self.d: dict = d

  @staticmethod
  def dynamic_load(module):
    if isinstance(module, tuple):
      return getattr(import_module(module[0]), module[1])
    else:
      return module

  def __getitem__(self, idx):
    return self.dynamic_load(self.d[idx])

  def __str__(self):
    return self.d.__str__()

  def __repr__(self):
    return self.d.__repr__()


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

helper_register = dict_warp({
    'YOLOHelper': ('tools.yolo', 'YOLOHelper'),
    'YOLOAlignHelper': ('tools.yoloalign', 'YOLOAlignHelper'),
    'PFLDHelper': ('tools.pfld', 'PFLDHelper'),
    'PFLDV2Helper': ('tools.pfld_v2', 'PFLDV2Helper'),
    'CtdetHelper': ('tools.ctdet', 'CtdetHelper'),
    'FcaeRecHelper': ('tools.facerec', 'FcaeRecHelper'),
    'LFFDHelper': ('tools.lffd', 'LFFDHelper'),
    'TinyImgnetHelper': ('tools.tinyimgnet', 'TinyImgnetHelper'),
    'ImgnetHelper': ('tools.imgnet', 'ImgnetHelper'),
    'RetinaFaceHelper': ('tools.retinaface', 'RetinaFaceHelper'),
    'DCASETask2Helper': ('tools.dcasetask2', 'DCASETask2Helper'),
    'DCASETask5Helper': ('tools.dcasetask5', 'DCASETask5Helper'),
    'SSDHelper': ('tools.ssd', 'SSDHelper'),
    'DCASETask5FixMatchSSLHelper': ('tools.dcasetask5', 'DCASETask5FixMatchSSLHelper'),
    'KerasDatasetHelper': ('tools.kerasdataset', 'KerasDatasetHelper'),
    'KerasDatasetSemiHelper': ('tools.kerasdatasetsemi', 'KerasDatasetSemiHelper'),
    'KerasDatasetGanHelper': ('tools.dcgan', 'KerasDatasetGanHelper'),
    'AnimeGanHelper': ('tools.animegan', 'AnimeGanHelper'),
    'CMPFacadeHelper': ('tools.pix2pix', 'CMPFacadeHelper'),
    'OpenPoseHelper': ('tools.openpose', 'OpenPoseHelper'),
    'PhotoTransferHelper': ('tools.phototransfer', 'PhotoTransferHelper'),
})

network_register = dict_warp({
    'simpleclassifynet': ('models.simplenet', 'simpleclassifynet'),
    'mbv1_facerec': ('models.networks', 'mbv1_facerec'),
    'mbv1_facerec_k210': ('models.networks4k210', 'mbv1_facerec_k210'),
    'mbv1_facerec_k210_eager': ('models.networks4k210', 'mbv1_facerec_k210_eager'),
    'FMobileFaceNet_eager': ('models.networks', 'FMobileFaceNet_eager'),
    'mbv2_ctdet': ('models.networks', 'mbv2_ctdet'),
    'mbv1_imgnet': ('models.networks', 'mbv1_imgnet'),
    'mbv2_imgnet': ('models.networks', 'mbv2_imgnet'),
    'yolo': ('models.networks', 'yolo'),
    'tiny_yolo': ('models.networks', 'tiny_yolo'),
    'yolo3_nano': ('models.networks', 'yolo3_nano'),
    'yolo_mbv1': ('models.networks', 'yolo_mbv1'),
    'pfld': ('models.networks', 'pfld'),
    'rffacedetnet': ('models.receptivefieldnet', 'rffacedetnet'),
    'retinafacenet': ('models.networks', 'retinafacenet'),
    'shuffle_ctdet': ('models.networks', 'shuffle_ctdet'),
    'yolo_mbv1_k210': ('models.networks4k210', 'yolo_mbv1_k210'),
    'yolo_mbv2_k210': ('models.networks4k210', 'yolo_mbv2_k210'),
    'yolo2_mbv1_k210': ('models.networks4k210', 'yolo2_mbv1_k210'),
    'yolov2algin_mbv1_k210': ('models.networks4k210', 'yolov2algin_mbv1_k210'),
    'yoloalgin_mbv1_k210': ('models.networks4k210', 'yoloalgin_mbv1_k210'),
    'pfld_k210': ('models.networks4k210', 'pfld_k210'),
    'mbv1_imgnet_k210': ('models.networks4k210', 'mbv1_imgnet_k210'),
    'mbv2_imgnet_k210': ('models.networks4k210', 'mbv2_imgnet_k210'),
    'retinafacenet_k210': ('models.networks4k210', 'retinafacenet_k210'),
    'retinafacenet_wflw': ('models.networks', 'retinafacenet_wflw'),
    'retinafacenet_k210_v1': ('models.networks4k210', 'retinafacenet_k210_v1'),
    'retinafacenet_k210_v2': ('models.networks4k210', 'retinafacenet_k210_v2'),
    'retinafacenet_k210_v3': ('models.networks4k210', 'retinafacenet_k210_v3'),
    'retinaface_slim': ('models.networks', 'retinaface_slim'),
    'ullfd_slim': ('models.networks', 'ullfd_slim'),
    'ullfd_k210': ('models.networks4k210', 'ullfd_k210'),
    'ullfd_k210_v1': ('models.networks4k210', 'ullfd_k210_v1'),
    'ullfd_k210_v2': ('models.networks4k210', 'ullfd_k210_v2'),
    'ullfd_k210_v3': ('models.networks4k210', 'ullfd_k210_v3'),
    'dualmbv2net': ('models.audionet', 'dualmbv2net'),
    'dcasetask5basemodel': ('models.networks', 'dcasetask5basemodel'),
    'imageclassifierCNN13': ('models.networks', 'imageclassifierCNN13'),
    'dcgan_mnist': ('models.gannet', 'dcgan_mnist'),
    'pix2pix_facde': ('models.gannet', 'pix2pix_facde'),
    'animenet': ('models.gannet', 'animenet'),
    'cifar_infomax_ssl_v1': ('models.semisupervised', 'cifar_infomax_ssl_v1'),
    'MobileNetV1OpenPose': ('models.openpose', 'MobileNetV1OpenPose'),
    'ugatitnet': ('models.gannet', 'ugatitnet')
})

loss_register = dict_warp({
    'YOLOLoss': ('tools.yolo', 'YOLOLoss'),
    'YOLOIouLoss': ('tools.yolo', 'YOLOIouLoss'),
    'YOLOAlignLoss': ('tools.yoloalign', 'YOLOAlignLoss'),
    'PFLDLoss': ('tools.pfld', 'PFLDLoss'),
    'CtdetLoss': ('tools.ctdet', 'CtdetLoss'),
    'TripletLoss': ('tools.facerec', 'TripletLoss'),
    'SparseSoftmaxLoss': ('tools.facerec', 'SparseSoftmaxLoss'),
    'SparseAmsoftmaxLoss': ('tools.facerec', 'SparseAmsoftmaxLoss'),
    'SparseAsoftmaxLoss': ('tools.facerec', 'SparseAsoftmaxLoss'),
    'LFFDLoss': ('tools.lffd', 'LFFDLoss'),
    'ClassifyLoss': ('tools.imgnet', 'ClassifyLoss'),
    'RetinaFaceLoss': ('tools.retinaface', 'RetinaFaceLoss'),
    'SemiBCELoss': ('tools.dcasetask2', 'SemiBCELoss'),
    'SSDLoss': ('tools.ssd', 'SSDLoss')
})

callback_register = dict_warp({
    'MultiScaleTrain': ('tools.yolo', 'MultiScaleTrain'),
    'EarlyStopping': tf.keras.callbacks.EarlyStopping,
    'ModelCheckpoint': tf.keras.callbacks.ModelCheckpoint,
    'TerminateOnNaN': tf.keras.callbacks.TerminateOnNaN,
    'YOLOMap': ('tools.yolo', 'YOLOMap'),
    'StepLR': ('tools.custom', 'StepLR'),
    'CosineLR': ('tools.custom', 'CosineLR'),
    'ScheduleLR': ('tools.custom', 'ScheduleLR'),
    'FacerecValidation': ('tools.facerec', 'FacerecValidation'),
    'LwlrapValidation': ('tools.dcasetask2', 'LwlrapValidation'),
    'AugmenterStateSync': ('tools.dcasetask5', 'AugmenterStateSync')
})

optimizer_register = dict_warp({
    'Adam': tf.keras.optimizers.Adam,
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam,
    'Ftrl': tf.keras.optimizers.Ftrl,
})

infer_register = dict_warp({
    'yolo_infer': ('tools.yolo', 'yolo_infer'),
    'yoloalgin_infer': ('tools.yoloalign', 'yoloalgin_infer'),
    'pfld_infer': ('tools.pfld', 'pfld_infer'),
    'ctdet_infer': ('tools.ctdet', 'ctdet_infer'),
    'retinaface_infer': ('tools.retinaface', 'retinaface_infer'),
    'ssd_infer': ('tools.ssd', 'ssd_infer'),
    'imgnet_infer': ('tools.imgnet', 'imgnet_infer'),
    'openpose_infer': ('tools.openpose', 'openpose_infer'),
})

eval_register = dict_warp({
    'yolo_eval': ('tools.yolo', 'yolo_eval'),
    'facerec_eval': ('tools.facerec', 'facerec_eval'),
    'imgnet_eval': ('tools.imgnet', 'imgnet_eval')
})

trainloop_register = dict_warp({
    'Task5SupervisedLoop': ('tools.dcasetask5', 'Task5SupervisedLoop'),
    'FaceTripletTrainingLoop': ('tools.facerec', 'FaceTripletTrainingLoop'),
    'FaceSoftmaxTrainingLoop': ('tools.facerec', 'FaceSoftmaxTrainingLoop'),
    'Task5FixMatchSslLoop': ('tools.dcasetask5', 'Task5FixMatchSslLoop'),
    'UDASslLoop': ('tools.kerasdataset', 'UDASslLoop'),
    'MixMatchSslLoop': ('tools.kerasdataset', 'MixMatchSslLoop'),
    'FixMatchMixUpSslLoop': ('tools.kerasdataset', 'FixMatchMixUpSslLoop'),
    'InfoMaxLoop': ('tools.kerasdataset', 'InfoMaxLoop'),
    'InfoMaxSslV1Loop': ('tools.kerasdataset', 'InfoMaxSslV1Loop'),
    'InfoMaxSslV2Loop': ('tools.kerasdataset', 'InfoMaxSslV2Loop'),
    'DCGanLoop': ('tools.dcgan', 'DCGanLoop'),
    'Pix2PixLoop': ('tools.pix2pix', 'Pix2PixLoop'),
    'AnimeGanInitLoop': ('tools.animegan', 'AnimeGanInitLoop'),
    'AnimeGanLoop': ('tools.animegan', 'AnimeGanLoop'),
    'OpenPoseLoop': ('tools.openpose', 'OpenPoseLoop'),
    'PhotoTransferLoop': ('tools.phototransfer', 'PhotoTransferLoop'),
    'KerasDatasetLoop': ('tools.kerasdataset', 'KerasDatasetLoop'),
})


if __name__ == "__main__":
  with open('config/default.yml', 'w') as f:
    safe_dump(ArgDict, f, sort_keys=False)
