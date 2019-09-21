import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
from toolz import pipe
import numpy as np


class DCN(k.layers.Layer):
    """ from  https://github.com/LirongWu/doform_conv_tensorflow/blob/master/deform_con2v.py """

    def __init__(self, filters, kernel, strides, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.N = kernel * kernel   # Number of kernel elements in a bin

    def get_pn(self):
        # Create the pn [1, 1, 1, 2N]
        pn_x, pn_y = np.meshgrid(range(-(self.kernel - 1) // 2,
                                       (self.kernel - 1) // 2 + 1),
                                 range(-(self.kernel - 1) // 2,
                                       (self.kernel - 1) // 2 + 1), indexing="ij")

        # The order is [x1, x2, ..., y1, y2, ...]
        pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

        pn = np.reshape(pn, [1, 1, 1, 2 * self.N]).astype(np.float32)

        # Change the dtype of pn
        pn = tf.constant(pn, self.dtype)

        return pn

    def get_p0(self, h, w, C):
        # Create the p0 [1, h, w, 2N]
        p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
        p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)
        p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)
        p0 = np.concatenate((p0_x, p0_y), axis=3).astype(np.float32)

        # Change the dtype of p0
        p0 = tf.constant(p0, self.dtype)
        return p0

    def get_q(self, h, w):
        # Create the q [h, w, 2]
        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
        q_x = q_x.flatten().reshape(h, w, 1)
        q_y = q_y.flatten().reshape(h, w, 1)
        q = np.concatenate((q_x, q_y), axis=2).astype(np.float32)
        # Change the dtype of q
        q = tf.constant(q, self.dtype)

        return q

    def reshape_x_offset(self, x_offset):
        # Get the new_shape
        x_offset = [tf.reshape(x_offset[:, :, :, s:s + self.kernel, :],
                               [-1, self.h, self.w * self.kernel, self.C])
                    for s in range(0, self.N, self.kernel)]
        x_offset = tf.concat(x_offset, axis=2)

        # Reshape to final shape [-1, h*kernel, w*kernel, C]
        x_offset = tf.reshape(x_offset, [-1, self.h * self.kernel, self.w * self.kernel, self.C])

        return x_offset

    def build(self, input_shape: tf.TensorShape):

        batch, self.h, self.w, self.C = input_shape.as_list()
        self.offset_conv = k.layers.Conv2D(2 * self.N, self.kernel, self.strides, 'same', trainable=self.trainable)
        self.weight_conv = k.layers.Conv2D(self.C * self.N, self.kernel, self.strides, 'same', trainable=self.trainable)
        self.feature_conv = k.layers.Conv2D(self.filters, self.kernel, self.kernel, 'same', trainable=self.trainable)

        self.pn = self.get_pn()  # pn with shape [1, 1, 1, 2N]
        self.p0 = self.get_p0(self.h, self.w, self.C)  # p0 with shape [1, h, w, 2N]
        self.q = self.get_q(self.h, self.w)  # q with shape [h, w, 2]
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):

        # offset with shape [-1, h, w, 2N]
        offset = self.offset_conv(inputs, **kwargs)

        # delte_weight with shape [-1, h, w, N * C]
        delte_weight = self.weight_conv(inputs, **kwargs)
        delte_weight = tf.sigmoid(delte_weight)

        # p with shape [-1, h, w, 2N]
        p = self.pn + self.p0 + offset

        # Reshape p to [-1, h, w, 2N, 1, 1]
        p = tf.reshape(p, [-1, self.h, self.w, 2 * self.N, 1, 1])

        # Bilinear interpolation kernel G ([-1, h, w, N, h, w])
        gx = tf.maximum(1 - tf.abs(p[:, :, :, :self.N, :, :] - self.q[:, :, 0]), 0)
        gy = tf.maximum(1 - tf.abs(p[:, :, :, self.N:, :, :] - self.q[:, :, 1]), 0)
        G = gx * gy

        # Reshape G to [-1, h*w*self.N, h*w]
        G = tf.reshape(G, [-1, self.h * self.w * self.N, self.h * self.w])

        # Reshape x to [-1, h*w, C]
        x = tf.reshape(inputs, [-1, self.h * self.w, self.C])

        # x_offset with shape [-1, h, w, N, C]
        x_offset = tf.reshape(tf.matmul(G, x), [-1, self.h, self.w, self.N, self.C])

        # Reshape x_offset to [-1, h*kernel, w*kernel, C]
        x_offset = self.reshape_x_offset(x_offset)

        # Reshape delte_weight to [-1, h*kernel, w*kernel, C]
        delte_weight = tf.reshape(delte_weight, [-1, self.h * self.kernel, self.w * self.kernel, self.C])

        y = x_offset * delte_weight

        # Get the output of the deformable convolutional layer
        layer_output = self.feature_conv(y, **kwargs)

        return layer_output

    def compute_output_shape(self, input_shape: list):
        out_shape = tf.TensorShape(input_shape).as_list()
        out_shape[-1] = self.filters
        return tf.TensorShape(out_shape)


class DeconvLayer(object):
    def __init__(self, layer_num, filter_num, kernel_num):
        self.l = []
        for i in range(layer_num):
            kernel, padding, output_padding = self._get_cfg(kernel_num[i], i)
            self.l.append([DCN(filter_num[i], 3, 1),
                           kl.BatchNormalization(momentum=0.1),
                           kl.ReLU(),
                           kl.Conv2DTranspose(filter_num[i], kernel, 2, padding, output_padding, use_bias=False),
                           kl.BatchNormalization(momentum=0.1),
                           kl.ReLU()])

    def _get_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 'same'
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 'same'
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 'valid'
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def __call__(self, inputs: tf.Tensor):
        tmp = inputs
        for l in self.l:
            tmp = pipe(tmp, *l)
        return tmp
