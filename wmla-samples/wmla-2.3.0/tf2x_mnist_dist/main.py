################################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2021 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
################################################################################

# https://www.tensorflow.org/tutorials/distribute/keras
# https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
# https://tensorflow.google.cn/tutorials/distribute/multi_worker_with_keras
# https://github.com/tensorflow/tensorflow/issues/35442
# https://github.com/tensorflow/tensorflow/issues/36846

from __future__ import print_function
import argparse
import os
import json
import time
from pathlib import PurePath
import mycallback as MyCallback

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

BUFFER_SIZE = 10000

data_dir = os.environ.get("DATA_DIR", "/tmp")
os.makedirs(data_dir, exist_ok=True)

result_dir = os.environ.get("RESULT_DIR", "/tmp")
os.makedirs(result_dir, exist_ok=True)

log_dir = os.environ.get("LOG_DIR", str(PurePath(result_dir, "log")))
os.makedirs(log_dir, exist_ok=True)

checkpoint_dir = str(PurePath(result_dir, "checkpoint"))
os.makedirs(checkpoint_dir, exist_ok=True)

model_path = str(PurePath(result_dir, "model"))
os.makedirs(model_path, exist_ok=True)

print("data_dir=%s, result_dir=%s" % (data_dir, result_dir))


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name='mnist', data_dir=data_dir, download=True, with_info=True, as_supervised=True)
    mnist_train = datasets['train'].map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
        BUFFER_SIZE)
    mnist_test = datasets['test'].map(scale)

    return mnist_train, mnist_test


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def _is_chief(task_type, task_id):
    return (task_type == 'worker' and task_id == 0) or task_type is None


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


def main():
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # args = parser.parse_args()
    # wmla distribute mode will also pass --worker_hosts, --task_id and --job_name
    args, unknown = parser.parse_known_args()
    print(args)
    print(unknown)

    use_cuda = not args.no_cuda

    tf_config = os.environ.get("TF_CONFIG")
    if tf_config:
        # {"cluster": {"worker": ["dlw10.aus.stglabs.ibm.com:46001", "dlw10.aus.sMultiWorkerMirroredStrategytglabs.ibm.com:37927"]}, "task": {"index": 1, "type": "worker"}}
        print("TF_CONFIG is founded")
        print(tf_config)
        tf_config_json = json.loads(tf_config)
        num_workers = len(tf_config_json['cluster']['worker'])
        task_type = tf_config_json['task']['type']
        task_id = int(tf_config_json['task']['index'])
    else:
        print("TF_CONFIG is not founded")
        num_workers = 1
        task_type = "worker"
        task_id = 0

    is_chief = _is_chief(task_type, task_id)

    print("Let's use {} workers. is_chief = {}".format(str(num_workers), str(is_chief)))

    if num_workers > 1:
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = build_and_compile_cnn_model()

        global_batch_size = args.batch_size * num_workers
        mnist_train, mnist_test = make_datasets_unbatched()
        train_datasets = mnist_train.batch(global_batch_size)
        eval_dataset = mnist_test.batch(global_batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA  # AutoShardPolicy.OFF can work too.
        train_datasets = train_datasets.with_options(options)
    else:
        if use_cuda:
            strategy = tf.distribute.MirroredStrategy()
            print("Let's use {} gpus".format(str(strategy.num_replicas_in_sync)))

            global_batch_size = args.batch_size * strategy.num_replicas_in_sync
            mnist_train, mnist_test = make_datasets_unbatched()
            train_datasets = mnist_train.batch(global_batch_size)
            eval_dataset = mnist_test.batch(global_batch_size)

            with strategy.scope():
                model = build_and_compile_cnn_model()
        else:
            print("Let's use cpu")
            global_batch_size = args.batch_size
            mnist_train, mnist_test = make_datasets_unbatched()
            train_datasets = mnist_train.batch(global_batch_size)
            eval_dataset = mnist_test.batch(global_batch_size)

            model = build_and_compile_cnn_model()

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_path = write_filepath(checkpoint_dir, task_type, task_id)

    cp_callback = MyCallback.get_cp_callback(checkpoint_path)
    mq_callback = MyCallback.get_mq_callback()
    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
        cp_callback,
        mq_callback
    ]

    start_time = time.time()
    model.fit(x=train_datasets,
              epochs=args.epochs,
              verbose=1,
              callbacks=callbacks)
    duration = (time.time() - start_time) / 60
    print("Train finished. Time cost: %.2f minutes" % duration)

    start_time = time.time()
    score = model.evaluate(eval_dataset, verbose=0)
    duration = (time.time() - start_time) / 60
    print("Test finished. Time cost: %.2f minutes. Test loss: %f, Test accuracy: %f" % (duration, score[0], score[1]))

    # save model
    write_model_path = write_filepath(model_path, task_type, task_id)
    model.save(write_model_path)
    print("Model saved in path: %s" % write_model_path)


if __name__ == '__main__':
    main()
