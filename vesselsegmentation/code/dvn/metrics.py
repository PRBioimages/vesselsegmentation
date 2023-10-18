from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf
print(tf.__version__)
from sklearn.metrics import f1_score, precision_recall_curve

def dice_score(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten())

def dice_information(y_true, y_pred):
    prec, rec, thres = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    f1 = (2. * prec * rec) / (prec + rec)
    ind = np.argmax(f1)
    return prec[ind], rec[ind], f1[ind], thres[ind]

def threshold_accuracy(threshold=0.5):
    def metric(y_true, y_pred):
        pred = K.cast(K.greater_equal(y_pred, threshold),'int32')
        return K.mean(K.equal(K.cast(y_true, 'int32'), pred))
    return metric

def categorical_accuracy(axis=-1):
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=axis),K.argmax(y_pred, axis=axis)))
    return accuracy


def dice(smooth=1):
    def metric(y_true, y_pred):
        # adapting to sparse annotation
        c_true = K.cast(K.not_equal(y_true, 2), K.dtype(y_pred))
        y_true = y_true * c_true
        y_pred = y_pred * c_true

        intersection = K.sum(y_true * y_pred, axis=list(range(1, K.ndim(y_true))))
        print('y_true',y_true.shape)
        print('y_pred',y_pred.shape)
        print('intersection',intersection.shape)
        union = K.sum(y_true, axis=list(range(1, K.ndim(y_true)))) + K.sum(y_pred, axis=list(range(1, K.ndim(y_true))))
        print('union', union.shape)
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return metric


def metric_dice(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)