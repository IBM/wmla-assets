import tensorflow as tf
import os

def preprocess_raw_image(image_raw, label, height, width, image_size=224, image_depth=3):

    raw_image=tf.io.decode_raw(image_raw, tf.dtypes.uint8)
    uint8image = tf.reshape(raw_image, [height, width, image_depth])
    image = tf.image.resize_with_crop_or_pad(uint8image, image_size, image_size)
    image = tf.dtypes.cast(image,tf.dtypes.float32)
    image /= 127.5
    image -= 1.
    
    return image, label

def preprocess_raw_image_validate(image_raw, label, height, width, image_size=224, image_depth=3):

    raw_image=tf.io.decode_raw(image_raw, tf.dtypes.uint8)
    uint8image = tf.reshape(raw_image, [height, width, image_depth])
    image = tf.image.resize_with_crop_or_pad(uint8image, image_size, image_size)
    image = tf.dtypes.cast(image,tf.dtypes.float32)
    image /= 127.5
    image -= 1.
    
    return tf.constant(-1, dtype=tf.int32), image, label

def preprocess_bin_image_inference(image_name, image_file, image_size=224, image_depth=3):
    
    image_raw = tf.io.read_file(image_file)
    raw_image=tf.io.decode_image(image_raw, channels=image_depth)
    image = tf.image.resize_with_crop_or_pad(raw_image, image_size, image_size)
    image = tf.dtypes.cast(image,tf.dtypes.float32)
    image /= 127.5
    image -= 1.
    
    return image_name, image, tf.constant(-1, dtype=tf.int32)

