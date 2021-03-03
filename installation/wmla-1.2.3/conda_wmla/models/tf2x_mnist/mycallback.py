from __future__ import print_function
import tensorflow as tf


def mq_record(batch, logs):
    loss = logs['loss']
    accuracy = logs['accuracy']


def get_mq_callback():
    cb = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: mq_record(batch, logs))
    return cb


def get_cp_callback(checkpointDir):
    checkpoint_path = checkpointDir + "/cp-{epoch:04d}.ckpt"
    cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    return cb
