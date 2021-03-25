from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()  # run in tf1
import argparse
import os
import sys

from uncompiled_models import models_factory
from input_data import data_factory

import tf_parameter_mgr
from monitor_tf import TFKerasMonitor

def parse_args():
    
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='image classification')
    
    ##### WML-A integration parameters #######
    parser.add_argument('--train_dir', type=str, default='./train_dir', help='input the path of model checkpoint file path')
    parser.add_argument('--weights', type=str, default=None, help='input the path of initial weight file')

    ##### model parameters #######
    parser.add_argument('--network', dest='network', type=str, default='vgg19', 
                        help='name of the network used to train the classification model, \
                        one of vgg19, inceptionv3, mobilenetv2, resnet50, resnet101, resnet152, densenet')
    parser.add_argument('--class_num', dest='class_num', type=int, default=5, help='number of labels of the classification model to train')
    parser.add_argument('--train_backbone', dest='train_backbone', type=bool, default=False, help='train the whole model or just the head')
    parser.add_argument('--backbone_weights',dest='backbone_weights', type=str, default='imagenet', help='initial weight of the vgg19 conv backbone')
    
    args = parser.parse_args()

    return args

def get_init_weight_file():
    weight_dir = args.weights
    for file in os.listdir(weight_dir):
        if file.endswith('.h5'):
            return os.path.join(weight_dir, file)

def train_one_step(model, optimizer, loss_fn, images, labels):
    
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = loss_fn(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss, logits, grads

def test_one_step(model, loss_fn, images, labels):
    logits = model(images)
    loss = loss_fn(labels, logits)
    
    return loss, logits

BATCH_SIZE = tf_parameter_mgr.getTrainBatchSize()
    
if __name__ == '__main__':
    args = parse_args()

    train_data = data_factory.get_dataset_from_tfrecords(args.network, tf_parameter_mgr.getTrainData(), tf_parameter_mgr.getTrainBatchSize())
    test_data = data_factory.get_dataset_from_tfrecords(args.network, tf_parameter_mgr.getTestData(), tf_parameter_mgr.getTrainBatchSize())

    # Output total iterations info for deep learning insights
    epochs = tf_parameter_mgr.getMaxSteps()
    print("Total iterations: %s" % (len(list(train_data.as_numpy_iterator())) * epochs))
    
    LR_POLICY = tf_parameter_mgr.getLearningRate()
    optimizer = tf_parameter_mgr.getOptimizer(LR_POLICY)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    #model = models_factory.get_uncompiled_model(args.network, class_num=args.class_num, trainable=args.train_backbone, weights=args.backbone_weights) #get_uncompiled_model()
    model = models_factory.get_uncompiled_model(args.network, class_num=args.class_num, trainable=True, weights=None)
    model.summary()
    
    weight_file = get_init_weight_file()
    if weight_file:
        print('loading weights')
        model.load_weights(weight_file)
    
    tfmonitor = TFKerasMonitor(model)
    step = 0
    for epoch in range(tf_parameter_mgr.getMaxSteps()):
        for batch, (images, labels) in enumerate(train_data):
            loss, logits, grads = train_one_step(model, optimizer, loss_fn, images, labels)
            if step % 50 == 0:
                tf.print(tf.strings.format('epoch {}, step {}, loss {}, accuracy {}', (epoch, step, loss, compute_accuracy(labels, logits))))
                tfmonitor.log_train(step, grads, loss, compute_accuracy(labels, logits), images)
                compute_accuracy.reset_states()
                for test_image, test_labels in test_data:
                    test_loss, test_logits = test_one_step(model, loss_fn, test_image, test_labels)
                    tf.print(tf.strings.format('epoch {}, step {}, test loss {}, test accuracy {}', (epoch, step, test_loss, compute_accuracy(test_labels, test_logits))))
                    tfmonitor.log_test(step, test_loss, compute_accuracy(test_labels, test_logits))  #TODO: accuracy is not correct
                    break
                compute_accuracy.reset_states()
            step += 1
        #tf.print(tf.strings.format('epoch {}, step {}, loss {}, accuracy {}', (epoch, batch, loss, compute_accuracy.result())))
        #compute_accuracy.reset_states()
    
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    #model.save(args.train_dir + '/model_vgg19.h5')
    #model.save_weights(args.train_dir +'/weights_vgg19.h5')
    #tf.keras.experimental.export_saved_model(model, args.train_dir)
    #tf.compat.v1.keras.experimental.export_saved_model(model, args.train_dir)
    tf.saved_model.save(model, args.train_dir)
    model.save_weights(args.train_dir +'/weights_vgg19.h5')
