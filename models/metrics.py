"""Defines metrics."""

import numpy as np
from sklearn import metrics
import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask=None):
  """Softmax cross-entropy loss with masking."""
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
  if mask != None:
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
  return tf.reduce_mean(loss)  
  

def masked_accuracy(preds, labels, mask=None, k=1):
  """Top-k accuracy with masking."""
  correct_prediction = tf.math.in_top_k(preds, tf.argmax(labels, 1), k)
  accuracy_all = tf.cast(correct_prediction, tf.float32)
  if mask != None:
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
  return tf.reduce_mean(accuracy_all)
  

def masked_f1(preds, labels, mask, sigmoid=False):
    """Micro f1 score with masking."""
    masked_labels = labels[mask,:]
    masked_preds = preds[mask,:]
    if not sigmoid:
        masked_labels = np.argmax(masked_labels, axis=1)
        masked_preds = np.argmax(masked_preds, axis=1)
    else:
        masked_preds[masked_preds > 0.5] = 1
        masked_preds[masked_preds <= 0.5] = 0
    return metrics.f1_score(masked_labels, masked_preds, average="micro")
