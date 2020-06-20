import tensorflow as tf
k = tf.keras
kl = tf.keras.layers


def simpleclassifynet(input_shape: list,
                      class_num: int,
                      use_bottleneck: bool = False,
                      use_cos_out: bool = True):
  l = [
      kl.Input(shape=input_shape),
      kl.Conv2D(64, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(128, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(256, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(256, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.Conv2D(128, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.GlobalMaxPooling2D(),
      kl.Dense(128)]

  if use_bottleneck:
    l += [kl.BatchNormalization(),
          kl.ReLU(6),
          kl.Dense(3)]
  if use_cos_out:
    l += [kl.Lambda(lambda x: tf.nn.l2_normalize(x, 1), name='emmbeding'),
          kl.Dense(class_num, use_bias=False, kernel_constraint=k.constraints.unit_norm())]
  else:
    l += [kl.Lambda(lambda x: x, name='emmbeding'),
          kl.Dense(class_num)]

  softmax_model: k.Model = k.Sequential(l)

  infer_model = val_model = train_model = softmax_model

  return infer_model, val_model, train_model
