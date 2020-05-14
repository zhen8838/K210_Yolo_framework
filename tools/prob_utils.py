from scipy.special import softmax, entr, rel_entr
import numpy as np
# from scipy.stats import entropy


def entropy(pk, qk=None, base=None, axis=0):
  """ 无归一化版 熵计算 """
  pk = np.asarray(pk)
  # pk = 1.0 * pk / np.sum(pk, axis=axis, keepdims=True)
  if qk is None:
    vec = entr(pk)
  else:
    qk = np.asarray(qk)
    if qk.shape != pk.shape:
      raise ValueError("qk and pk must have same shape.")
    # qk = 1.0 * qk / np.sum(qk, axis=axis, keepdims=True)
    vec = rel_entr(pk, qk)
  S = np.sum(vec, axis=axis)
  if base is not None:
    S /= np.log(base)
  return S


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = None, from_logits: bool = True) -> np.ndarray:
  if from_logits:
    p = softmax(p, axis=axis)
    q = softmax(q, axis=axis)
  return entropy(p, q, axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray, axis: int = None, from_logits: bool = True) -> np.ndarray:
  if from_logits:
    p = softmax(p, axis=axis)
    q = softmax(q, axis=axis)
  m = (p + q) / 2
  return (entropy(p, m, axis=axis) + entropy(q, m, axis=axis))
