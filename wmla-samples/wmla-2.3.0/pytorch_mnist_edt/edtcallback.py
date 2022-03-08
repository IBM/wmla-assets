#! /usr/bin/env python

################################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2020 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
################################################################################


from __future__ import print_function

import sys
import os
import torch
from callbacks import Callback
from callbacks import LoggerCallback
from emetrics import EMetrics
from elog import ELog
import time


class EDTLoggerCallback(LoggerCallback):
    def __init__(self):
        self.gs = 0

    def log_train_metrics(self, loss, acc, completed_batch, worker=0):
        acc = acc / 100.0
        self.gs += 1
        with EMetrics.open() as em:
            em.record(EMetrics.TEST_GROUP, completed_batch, {'loss': loss, 'accuracy': acc})
        with ELog.open() as log:
            log.recordTrain("Train", completed_batch, self.gs, loss, acc, worker)

    def log_test_metrics(self, loss, acc, completed_batch, worker=0):
        acc = acc / 100.0
        with ELog.open() as log:
            log.recordTest("Test", loss, acc, worker)


class EDTTrainCallback(Callback):
    def __init__(self, output_model_path):
        super(EDTTrainCallback, self).__init__()
        self.output_model_path = output_model_path
        self.start_time = time.time()

    def log_message(self, message):
        with ELog.open() as log:
            log.recordText(message)

    def on_epoch_end(self, epoch, save_model=False):
        self.log_message(
            "on_epoch_end, epoch=%d, total_epochs=%d, save_model=%s" % (
                epoch, self.params['epochs'], str(save_model)))
        if epoch == self.params['epochs'] - 1:
            duration = (time.time() - self.start_time) / 60
            self.log_message("Train finished. Time cost: %.2f minutes" % duration)
            if save_model:
                output_model_onnx = os.path.join(self.output_model_path, "trained_model.onnx")
                x = torch.randn(1, 1, 28, 28, device='cuda', requires_grad=True)
                torch.onnx.export(self.model, x, output_model_onnx, export_params=True,
                                  keep_initializers_as_inputs=True)
                self.log_message("Model saved in path: %s" % output_model_onnx)
