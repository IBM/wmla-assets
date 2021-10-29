################################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2020 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
################################################################################

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import argparse

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import os
import time
from pathlib import PurePath

import mycallback as MyCallback

data_dir = os.environ.get("DATA_DIR", "/tmp")
result_dir = os.environ.get("RESULT_DIR", "/tmp")
print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))
os.makedirs(data_dir, exist_ok=True)

checkpoint_dir = str(PurePath(result_dir, "checkpoint"))

model_path = str(PurePath(result_dir, "model"))
os.makedirs(model_path, exist_ok=True)


def _get_available_devices():
    return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
    return name


def train(args):
    use_cuda = not args.no_cuda

    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=str(PurePath(data_dir, "mnist.npz")))

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    cp_callback = MyCallback.get_cp_callback(checkpoint_dir)
    mq_callback = MyCallback.get_mq_callback()

    if use_cuda:
        # support multiple gpu
        available_devices = _get_available_devices()
        available_devices = [_normalize_device_name(name)
                             for name in available_devices]
        gpu_names = [x for x in available_devices if '/gpu:' in x]
        num_gpus = len(gpu_names)

        if num_gpus <= 0:
            raise ValueError('Unable to find any gpu device ')

        print("Let's use gpus: " + str(gpu_names))

        if num_gpus > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(lr=args.lr),
                  metrics=['accuracy'])

    start_time = time.time()
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[cp_callback, mq_callback])
    duration = (time.time() - start_time) / 60
    print("Train finished. Time cost: %.2f minutes" % duration)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save model
    model_json = model.to_json()
    open(str(PurePath(model_path, 'mnist_arch.json')), 'w').write(model_json)

    output_weight = str(PurePath(model_path, 'mnist_weights.h5'))
    model.save_weights(output_weight, overwrite=True)
    print("Weight saved in path: %s" % output_weight)
