from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import argparse
import os

from uncompiled_models import models_factory
from input_data import data_factory

import tf_parameter_mgr
from monitor_tf import TFKerasMonitorCallback

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='image classification')
    
    ##### WML-A integration parameters #######
    parser.add_argument('--train_dir', type=str, default='./train_dir', help='input the path of model checkpoint file path')
    parser.add_argument('--weights', type=str, default=None, help='input the path of initial weight file')

    ##### model parameters #######
    parser.add_argument('--network', dest='network', type=str, default='vgg19', 
                        help='name of the network used to train the classification model, \
                        one of vgg19, inceptionv3, mobilenetv2, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2, densenet121, densenet169, densenet201')
    parser.add_argument('--class_num', dest='class_num', type=int, default=5, help='number of labels of the classification model to train')
    parser.add_argument('--train_backbone', dest='train_backbone', type=bool, default=False, help='train the whole model or just the head')
    parser.add_argument('--backbone_weights',dest='backbone_weights', type=str, default='imagenet', help='initial weight of the vgg19 conv backbone')
    
    args = parser.parse_args()

    return args

def get_class_num():
    class_num = args.class_num
    label_file = os.path.join(os.path.dirname(tf_parameter_mgr.getTrainData()[0]), '..', 'labels.txt')
    if os.path.exists(label_file):
        class_num = len(open(label_file).readlines())
    return class_num

def get_init_weight_file():
    weight_dir = args.weights
    for file in os.listdir(weight_dir):
        if file.endswith('.h5'):
            return os.path.join(weight_dir, file)
    
def get_compiled_model():
    #model = models_factory.get_uncompiled_model(args.network, class_num=get_class_num(), trainable=args.train_backbone, weights=args.backbone_weights) #get_uncompiled_model()
    model = models_factory.get_uncompiled_model(args.network, class_num=get_class_num(), trainable=True, weights=None)
    model.summary()
    
    LR_POLICY = tf_parameter_mgr.getLearningRate()
    OPTIMIZER = tf_parameter_mgr.getOptimizer(LR_POLICY)
    model.compile(optimizer=OPTIMIZER,
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model

#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

if __name__ == '__main__':
    args = parse_args()

    train_data = data_factory.get_dataset_from_tfrecords(args.network, tf_parameter_mgr.getTrainData(), tf_parameter_mgr.getTrainBatchSize())
    test_data = data_factory.get_dataset_from_tfrecords(args.network, tf_parameter_mgr.getTestData(), tf_parameter_mgr.getTrainBatchSize())

    # Output total iterations info for deep learning insights
    epochs = tf_parameter_mgr.getMaxSteps()
    print("Total iterations: %s" % (len(list(train_data.as_numpy_iterator())) * epochs))
    
    #with strategy.scope():
    model = get_compiled_model()

    weight_file = get_init_weight_file()
    if weight_file:
        print('loading weights')
        model.load_weights(weight_file)
        
    history=model.fit(train_data, epochs=tf_parameter_mgr.getMaxSteps(), callbacks=[TFKerasMonitorCallback(test_data)])
    print(history.history)
    
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    
    #tf.compat.v1.keras.experimental.export_saved_model(model, args.train_dir)
    tf.saved_model.save(model, args.train_dir)
    #model.save(args.train_dir + '/model_vgg19.h5')
    model.save_weights(args.train_dir +'/weights_vgg19.h5')
