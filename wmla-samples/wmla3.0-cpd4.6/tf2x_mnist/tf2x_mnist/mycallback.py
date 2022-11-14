from __future__ import print_function
import tensorflow as tf

import sys
import os
from os import environ
from emetrics import EMetrics


def mq_record(batch, logs):
    loss = logs['loss']
    accuracy = logs['accuracy']
    with EMetrics.open() as em:
        em.record(EMetrics.TEST_GROUP, batch, {'loss': loss, 'accuracy': accuracy})


def get_mq_callback():
    c = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: mq_record(batch, logs))
    return c


def get_cp_callback(checkpointDir):
    checkpoint_path = checkpointDir + "/cp-{epoch:04d}.ckpt"
    c = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    return c
