import tensorflow as tf
import tensorflow_addons as tfa


def power_to_db(magnitude, ref=1.0, amin=1e-10, top_db=80.0):
  ref_value = tf.abs(ref)
  log_spec = 10.0 * (tf.math.log(tf.maximum(amin, magnitude)) / tf.math.log(10.))
  log_spec -= 10.0 * (tf.math.log(tf.maximum(amin, ref_value)) / tf.math.log(10.))
  log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
  return log_spec


def freq_mask(mel: tf.Tensor, factor: float = 0.1, times: int = 1) -> tf.Tensor:
  """ mel spectogram freq mask (row mask)
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (tf.Tensor): mask factor (0. ~  1.)
			times (int): int, default = 1
	
	Returns:
			tf.Tensor: [freq, time] float32
	"""
  freq_max, time_max = mel.shape

  def body(idx, mel):
    max_w = tf.cast(factor * tf.cast(freq_max, tf.float32) / 2, tf.int32)
    coord = tf.random.uniform([], 0, freq_max, tf.int32)
    mask_w = tf.random.uniform([], 0, tf.maximum(max_w, 1), tf.int32)
    cut = tf.stack([coord - mask_w, coord + mask_w])
    cut = tf.clip_by_value(cut, 0, freq_max)
    mel = tf.concat(
        [mel[:cut[0]],
         tf.zeros_like(mel[cut[0]:cut[1]]), mel[cut[1]:]], 0)
    return idx + 1, mel

  cond = lambda idx, mel: (idx < times)
  init_idx = tf.constant(0)
  _, aug_mel = tf.while_loop(
      cond,
      body, [init_idx, mel],
      shape_invariants=[init_idx.shape,
                        tf.TensorShape((None, time_max))])
  return aug_mel


def time_mask(mel: tf.Tensor, factor: float = 0.1, times: int = 1) -> tf.Tensor:
  """ mel spectogram time mask (cloum mask)
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (tf.Tensor): mask factor (0. ~  1.)
			times (int): int, default = 1
	
	Returns:
			tf.Tensor: [freq, time] float32
	"""
  freq_max, time_max = mel.shape

  def body(idx, mel):
    max_w = tf.cast(factor * tf.cast(time_max, tf.float32) / 2, tf.int32)
    coord = tf.random.uniform([], 0, time_max, tf.int32)
    mask_w = tf.random.uniform([], 0, tf.maximum(max_w, 1), tf.int32)
    cut = tf.stack([coord - mask_w, coord + mask_w])
    cut = tf.clip_by_value(cut, 0, time_max)
    mel = tf.concat(
        [mel[:, :cut[0]],
         tf.zeros_like(mel[:, cut[0]:cut[1]]), mel[:, cut[1]:]], 1)
    return idx + 1, mel

  cond = lambda idx, mel: (idx < times)
  init_idx = tf.constant(0)
  _, aug_mel = tf.while_loop(
      cond,
      body, [init_idx, mel],
      shape_invariants=[init_idx.shape,
                        tf.TensorShape((freq_max, None))])
  return aug_mel


def freq_rescale(mel: tf.Tensor, factor: float = 0.1) -> tf.Tensor:
  """rescale mel freq axis
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (float, optional): rescle factor. Defaults to 0.1.
	
	Returns:
			tf.Tensor: [freq, time] float32
	"""
  freq_max, time_max = mel.shape
  choosen_factor = tf.random.uniform([], 1 - factor, 1 + factor)

  new_freq_size = tf.cast(
      tf.cast(freq_max, tf.float32) * choosen_factor, tf.int32)

  mel_aug = tf.squeeze(
      tf.image.resize(tf.expand_dims(mel, -1), [new_freq_size, time_max]), -1)

  def fn():
    pad_offset = tf.random.uniform([], 0, freq_max - new_freq_size, tf.int32)
    return tf.pad(mel_aug,
                  [[pad_offset, freq_max - new_freq_size - pad_offset], [0, 0]])

  mel_aug = tf.cond(
      choosen_factor < 1., lambda: fn(), lambda: mel_aug[0:freq_max,])
  return mel_aug


def time_rescale(mel: tf.Tensor, factor: tf.Tensor = 0.1) -> tf.Tensor:
  """rescale mel time axis
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (tf.Tensor, optional): rescle factor. Defaults to 0.1.
	
	Returns:
			tf.Tensor: [freq, time] float32
	"""
  freq_max, time_max = mel.shape
  choosen_factor = tf.random.uniform([], 1 - factor, 1 + factor)

  new_time_size = tf.cast(
      tf.cast(time_max, tf.float32) * choosen_factor, tf.int32)

  mel_aug = tf.squeeze(
      tf.image.resize(tf.expand_dims(mel, -1), [freq_max, new_time_size]), -1)

  def fn():
    pad_offset = tf.random.uniform([], 0, time_max - new_time_size, tf.int32)
    return tf.pad(mel_aug,
                  [[0, 0], [pad_offset, time_max - new_time_size - pad_offset]])

  mel_aug = tf.cond(
      choosen_factor < 1., lambda: fn(), lambda: mel_aug[:, 0:time_max])
  return mel_aug


def mel_dropout(mel: tf.Tensor, drop_prob: int = 0.05) -> tf.Tensor:
  """ mel drop out
	
	Args:
			mel (tf.Tensor): [freq, time] float32, float32
			drop_prob (int, optional): keep prob. Defaults to 0.05.
	
	Returns:
			tf.Tensor: [freq, time] float32, float32
	"""
  return tf.nn.dropout(mel, rate=1 - drop_prob)


def time_warping(mel: tf.Tensor, factor: float = 0.1,
                 npoints: int = 1) -> tf.Tensor:
  """ mel time warp use by `image_sparse_warp`
		choice source point       from `[time//4, time - time//4]` 
		choice warped time width  from `[- factor/2 * time, factor/2 * time]`
		
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (float, optional): NOTE factor should be [0., 1.]. Defaults to 0.1.
			npoints (int, optional): disort point num NOTE don't set npoints > 5, it will be terrible. Defaults to 1.
			
	Returns:
			tf.Tensor: [freq, time] float32
	"""

  freq_max, time_max = mel.shape

  freq_max = tf.cast(freq_max, tf.float32)
  time_max = tf.cast(time_max, tf.float32)

  # random choice some point, NOTE don't choose boundary
  src_pt_y = tf.random.shuffle(tf.range(freq_max - 1) + 1)[:npoints]
  tau_4 = tf.math.floordiv(time_max, 4)
  src_pt_x = tf.random.shuffle(tf.range(tau_4, time_max - tau_4))[:npoints]
  src_pt = tf.stack([src_pt_y, src_pt_x], -1)

  disort_width = tf.random.uniform([npoints], -time_max * factor / 2,
                                   time_max * factor / 2)
  dest_pt_y = src_pt_y
  dest_pt_x = src_pt_x + disort_width
  dest_pt = tf.stack([dest_pt_y, dest_pt_x], -1)
  # NOTE num_boundary_points=1 keep image boundary will not be disort
  mel_aug, _ = tfa.image.sparse_image_warp(
      mel[None, ..., None],
      src_pt[None, ...],
      dest_pt[None, ...],
      num_boundary_points=1)
  return mel_aug[0, ..., 0]


def freq_warping(mel: tf.Tensor, factor: float = 0.1,
                 npoints: int = 1) -> tf.Tensor:
  """ mel freq warp use by `image_sparse_warp`
		choice source point       from `[freq//4, freq - freq//4]` 
		choice warped time width  from `[- factor/2 * freq, factor/2 * freq]`
		
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (float, optional): NOTE factor should be [0., 1.]. Defaults to 0.1.
			npoints (int, optional): disort point num NOTE don't set npoints > 5, it will be terrible. Defaults to 1.
			
	Returns:
			tf.Tensor: [freq, time] float32
	"""

  freq_max, time_max = mel.shape
  freq_max = tf.cast(freq_max, tf.float32)
  # random choice some point, NOTE don't choose boundary
  freq_4 = tf.math.floordiv(freq_max, 4)
  src_pt_x = tf.random.shuffle(
      tf.range(tf.cast(time_max, tf.float32), dtype=tf.float32))[:npoints]
  src_pt_y = tf.random.shuffle(tf.range(freq_4, freq_max - freq_4))[:npoints]
  src_pt = tf.stack([src_pt_y, src_pt_x], -1)

  disort_width = tf.random.uniform([npoints], -freq_max * factor / 2,
                                   freq_max * factor / 2)
  dest_pt_y = src_pt_y + disort_width
  dest_pt_x = src_pt_x
  dest_pt = tf.stack([dest_pt_y, dest_pt_x], -1)
  # NOTE num_boundary_points=1 keep image boundary will not be disort
  mel_aug, _ = tfa.image.sparse_image_warp(
      mel[None, ..., None],
      src_pt[None, ...],
      dest_pt[None, ...],
      num_boundary_points=1)
  return mel_aug[0, ..., 0]


def mel_loudness(mel: tf.Tensor, factor: float = 0.1) -> tf.Tensor:
  """ mel spectrogram loudness control
	
	
	Args:
			mel (tf.Tensor): [freq, time] float32
			factor (float, optional): [0. ~ 1.]. Defaults to 0.1.
	
	Returns:
			tf.Tensor: [freq, time] float32
	"""
  min_v = tf.reduce_min(mel)
  return (mel-min_v) * tf.abs(1 - tf.random.uniform([], 0., factor)) + min_v
