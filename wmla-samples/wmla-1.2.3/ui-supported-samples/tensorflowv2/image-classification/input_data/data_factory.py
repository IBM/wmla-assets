import tensorflow as tf

from . import common_preprocessing

preprocessing_map = {
    'vgg19': common_preprocessing,
    'inceptionv3': common_preprocessing,
    'resnet50': common_preprocessing,
    'resnet101': common_preprocessing,
    'resnet152': common_preprocessing,
    'resnet50v2': common_preprocessing,
    'resnet101v2': common_preprocessing,
    'resnet152v2': common_preprocessing,
    'densenet121': common_preprocessing,
    'densenet169': common_preprocessing,
    'densenet201': common_preprocessing
    }

def _parse_image_function(example_proto):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    #tf.print(parsed_features)
    return parsed_features['image_raw'], parsed_features['label'], parsed_features['height'], parsed_features['width']


def get_dataset_from_tfrecords(model_name, tfrecords, batchsize, image_size=None, is_train=True):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecords)
    if is_train:
        preprocess_func = preprocessing_map[model_name].preprocess_raw_image
    else:
        preprocess_func = preprocessing_map[model_name].preprocess_raw_image_validate

    if not image_size:
        from uncompiled_models import models_factory
        image_size = models_factory.get_default_size(model_name)
    dataset = raw_image_dataset.map(_parse_image_function).map(lambda image_raw, label, height, width: preprocess_func(image_raw, label, height, width,image_size))
    
    if is_train:
        dataset = dataset.shuffle(buffer_size=batchsize*20)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batchsize)
    
    return dataset

def get_dataset_from_images(model_name, images_dir, batchsize, image_size=None, is_train=True):
    pass

def get_dataset_from_images_inference(model_name, images_dir, batchsize, image_size=None):
    import glob
    import os
    
    input_dir = os.path.expanduser(images_dir)
    imagefiles = glob.iglob(input_dir + '/*.*')
    imagefiles = [im_f for im_f in imagefiles
                if im_f.endswith(".jpg") or im_f.endswith(".jpeg") or im_f.endswith(".png")]
    imageBaseNames = [os.path.basename(image_file) for image_file in imagefiles]
    preprocess_func = preprocessing_map[model_name].preprocess_bin_image_inference
    if not image_size:
        from uncompiled_models import models_factory
        image_size = models_factory.get_default_size(model_name)
            
    dataset = tf.data.Dataset.from_tensor_slices((imageBaseNames, imagefiles))
    dataset = dataset.map(lambda imageBaseNames, image_file: preprocess_func(imageBaseNames, image_file, image_size))
    
    dataset = dataset.batch(batchsize)
    return dataset

 