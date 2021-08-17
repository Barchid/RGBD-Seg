# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import numbers
import torch
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class ConfusionMatrixTensorflow:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        if self.n_classes < 255:
            self.dtype = tf.uint8
        else:
            self.dtype = tf.uint16

        self.overall_confusion_matrix = np.zeros((self.n_classes,
                                                  self.n_classes))
        self.cm_func = self.build_confusion_matrix_graph()

    def reset_conf_matrix(self):
        self.overall_confusion_matrix = np.zeros((self.n_classes,
                                                  self.n_classes))

    def update_conf_matrix(self, ground_truth, prediction):
        sess = tf.compat.v1.Session()

        current_confusion_matrix = \
            sess.run(self.cm_func, feed_dict={self.ph_cm_y_true: ground_truth,
                                              self.ph_cm_y_pred: prediction})

        # update or create confusion matrix
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix

    def build_confusion_matrix_graph(self):

        self.ph_cm_y_true = tf.compat.v1.placeholder(dtype=self.dtype,
                                                     shape=None)
        self.ph_cm_y_pred = tf.compat.v1.placeholder(dtype=self.dtype,
                                                     shape=None)

        return tf.math.confusion_matrix(labels=self.ph_cm_y_true,
                                        predictions=self.ph_cm_y_pred,
                                        num_classes=self.n_classes)

    def compute_miou(self):
        cm = self.overall_confusion_matrix.copy()

        # sum over the ground truth, the predictions and create a mask for
        # empty classes
        gt_set = cm.sum(axis=1)
        pred_set = cm.sum(axis=0)
        # mask_gt = gt_set > 0
        #
        # # calculate intersection over union and the mean of it
        # intersection = np.diag(cm)[mask_gt]
        # union = gt_set[mask_gt] + pred_set[mask_gt] - intersection

        # calculate intersection over union and the mean of it
        intersection = np.diag(cm)
        union = gt_set + pred_set - intersection

        # union might be 0. Then convert nan to 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = intersection / union.astype(np.float32)
            iou = np.nan_to_num(iou)
        miou = np.mean(iou)

        return miou, iou


if __name__ == '__main__':
    # test if pytorch confusion matrix and tensorflow confusion matrix
    # compute the same
    label = np.array([0, 0, 1, 2, 3])
    prediction = np.array([1, 1, 0, 2, 3])

    cm_tf = ConfusionMatrixTensorflow(4)

    cm_tf.update_conf_matrix(label, prediction)

    print(cm_tf.overall_confusion_matrix)

    print('mIoU tensorflow:', cm_tf.compute_miou())
