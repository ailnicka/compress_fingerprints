import math
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from keras import backend as K
 

def multi_category_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    """
    focal loss for multi label problem
    based on 
    """
    epsilon = 1.e-12
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    return tf.reduce_mean(fl)


def hamming_loss(y_true, y_pred, threshold=0.5):
    """Computes hamming loss.
    Hamming loss is the fraction of wrong labels to the total number
    of labels.
    In multi-label classification, hamming loss penalizes only the
    individual labels.
    """
    y_pred = y_pred > threshold
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
    return tf.reduce_mean(tf.divide(nonzero, tf.cast(y_pred.shape[-1], tf.float32)))


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def tanimoto_similarity(y_true, y_pred):
    y_pred = K.round(y_pred)
    num = K.sum(y_pred*y_true, axis=-1)
    den = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - num
    return K.mean(num/den)

def cos_similarity(y_true, y_pred):
    y_pred = K.round(y_pred)
    num = K.sum(y_pred*y_true, axis=-1)
    den = K.sqrt(K.sum(y_true, axis=-1) * K.sum(y_pred, axis=-1))
    return K.mean(num/den)

