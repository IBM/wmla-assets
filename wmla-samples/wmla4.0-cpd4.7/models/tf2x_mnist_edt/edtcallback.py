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

from callbacks import LoggerCallback
from emetrics import EMetrics
from elog import ELog

class EDTLoggerCallback(LoggerCallback):
    def __init__(self):
        self.gs =0

    def log_train_metrics(self, metrics, completed_batch, epoch, worker=0):
        self.gs += 1
        with EMetrics.open() as em:
            em.record(EMetrics.TEST_GROUP,completed_batch, metrics)
        with ELog.open() as log:
            log.recordTrain("Train", completed_batch, self.gs, metrics)

    def log_test_metrics(self, metrics, completed_batch, epoch):
        with ELog.open() as log:
            log.recordTest("Test", metrics)
