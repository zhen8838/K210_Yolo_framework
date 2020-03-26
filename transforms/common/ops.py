import tensorflow_probability as tfp
import tensorflow as tf


def mixup(imga, anna, imgb, annb) -> [tf.Tensor, tf.Tensor]:
  rate = tfp.distributions.Beta(1., 1.).sample([])
  img = imga*rate + imgb * (1-rate)
  ann = tf.cast(anna, tf.float32) * rate + tf.cast(annb, tf.float32) * (1-rate)
  return img, ann
