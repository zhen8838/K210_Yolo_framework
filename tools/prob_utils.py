from scipy.stats import entropy
from scipy.special import softmax
import numpy as np


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
