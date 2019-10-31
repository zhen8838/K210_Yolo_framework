import tensorflow as tf
from models.darknet import compose
k = tf.keras
kl = tf.keras.layers


def rffeatbranch(inputs: tf.Tensor, prefix: str,
                 filters: int) -> [tf.Tensor, tf.Tensor]:
    """ receptive field net featrue map branch 

    Parameters
    ----------
    inputs : tf.Tensor

        inputs tensor

    prefix : str

        net prefix name

    filters : int

        filters

    Returns
    -------

    [tf.Tensor, tf.Tensor]

        pred score
            shape = [featrue_size,featrue_size,2]
        pred bbox:
            shape = [featrue_size,featrue_size,4]

    """
    relu1 = compose(kl.Conv2D(filters, 1, 1, 'valid', name=prefix + '_1'),
                    kl.ReLU(name=prefix + 'relu_1'))(inputs)

    score = compose(kl.Conv2D(filters, 1, 1, 'valid', name=prefix + '_2'),
                    kl.ReLU(name=prefix + 'relu_2'),
                    kl.Conv2D(2, 1, 1, 'valid', name=prefix + '_score'))(relu1)

    bbox = compose(kl.Conv2D(filters, 1, 1, 'valid', name=prefix + '_3'),
                   kl.ReLU(name=prefix + 'relu_3'),
                   kl.Conv2D(4, 1, 1, 'valid', name=prefix + '_bbox'))(relu1)
    return score, bbox


def rffacedetnet(input_shape: list, num_filters: list = [64, 128]) -> [k.Model, k.Model]:
    """ receptive field network

    Parameters
    ----------
    input_shape : list

    num_filters : list, optional

        filters lists, by default [64, 128]


    Returns
    -------

    [k.Model, k.Model]

        infer model , train model

    """
    inputs = kl.Input(input_shape)

    conv1 = kl.Conv2D(num_filters[0], 3, 2, 'valid', name='conv1')(inputs)
    relu1 = kl.ReLU(name='relu_conv1')(conv1)

    conv2 = kl.Conv2D(num_filters[0], 3, 2, 'valid', name='conv2')(relu1)

    conv4 = compose(kl.ReLU(name='relu_conv2'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv3'),
                    kl.ReLU(name='relu_conv3'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv4'))(conv2)

    conv4 = kl.Add(name='residual_1')([conv2, conv4])  # residual 1

    conv6 = compose(kl.ReLU(name='relu_conv4'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv5'),
                    kl.ReLU(name='relu_conv5'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv6'),)(conv4)
    conv6 = kl.Add(name='residual_2')([conv4, conv6])  # residual 2

    conv8 = compose(kl.ReLU(name='relu_conv6'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv7'),
                    kl.ReLU(name='relu_conv7'),
                    kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv8'))(conv6)

    conv8 = kl.Add(name='residual_3')([conv6, conv8])  # residual 3
    relu8 = kl.ReLU(name='relu_conv8')(conv8)

    # loss 1 RF: 55 scale = [10,20]
    pred_score_1, pred_bbox_1 = rffeatbranch(relu8, 'conv8', num_filters[1])  # (16, 159, 159, 2) , (16, 159, 159, 4)

    conv9 = kl.Conv2D(num_filters[0], 3, 2, 'valid', name='conv9')(relu8)

    conv11 = compose(kl.ReLU(name='relu_conv9'),
                     kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv10'),
                     kl.ReLU(name='relu_conv10'),
                     kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv11'))(conv9)

    conv11 = kl.Add()([conv9, conv11])
    relu11 = kl.ReLU(name='relu_conv11')(conv11)

    # loss 2 RF:95  scale [20,40]
    pred_score_2, pred_bbox_2 = rffeatbranch(relu11, 'conv11', num_filters[1])

    conv12 = kl.Conv2D(num_filters[0], 3, 2, 'valid', name='conv12')(relu11)

    conv14 = compose(kl.ReLU(name='relu_conv12'),
                     kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv13'),
                     kl.ReLU(name='relu_conv13'),
                     kl.Conv2D(num_filters[0], 3, 1, 'same', name='conv14'))(conv12)

    conv14 = kl.Add()([conv12, conv14])
    relu14 = kl.ReLU(name='relu_conv14')(conv14)

    # loss 3 RF:175  scale [40,80]
    pred_score_3, pred_bbox_3 = rffeatbranch(relu14, 'conv14', num_filters[1])

    conv15 = kl.Conv2D(num_filters[1], 3, 2, 'valid', name='conv15')(relu14)

    conv17 = compose(kl.ReLU(name='relu_conv15'),
                     kl.Conv2D(num_filters[1], 3, 1, 'same', name='conv16'),
                     kl.ReLU(name='relu_conv16'),
                     kl.Conv2D(num_filters[1], 3, 1, 'same', name='conv17'))(conv15)
    conv17 = kl.Add()([conv15, conv17])
    relu17 = kl.ReLU(name='relu_conv17')(conv17)

    # loss 4 RF:335 scale [80, 160]
    pred_score_4, pred_bbox_4 = rffeatbranch(relu17, 'conv17', num_filters[1])

    # conv block 18 num_nonzero
    conv18 = kl.Conv2D(num_filters[1], 3, 2, 'valid', name='conv18')(relu17)
    conv20 = compose(kl.ReLU(name='relu_conv18'),
                     kl.Conv2D(num_filters[1], 3, 1, 'same', name='conv19'),
                     kl.ReLU(name='relu_conv19'),
                     kl.Conv2D(num_filters[1], 3, 1, 'same', name='conv20'))(conv18)
    conv20 = kl.Add()([conv18, conv20])
    relu20 = kl.ReLU(name='relu_conv20')(conv20)

    # loss 5 RF:655 scale [160,320]
    pred_score_5, pred_bbox_5 = rffeatbranch(relu20, 'conv20', num_filters[1])

    net = k.Model(inputs, [kl.Concatenate(name='l1')([pred_score_1, pred_bbox_1]),
                           kl.Concatenate(name='l2')([pred_score_2, pred_bbox_2]),
                           kl.Concatenate(name='l3')([pred_score_3, pred_bbox_3]),
                           kl.Concatenate(name='l4')([pred_score_4, pred_bbox_4]),
                           kl.Concatenate(name='l5')([pred_score_5, pred_bbox_5])])
    return net, net
