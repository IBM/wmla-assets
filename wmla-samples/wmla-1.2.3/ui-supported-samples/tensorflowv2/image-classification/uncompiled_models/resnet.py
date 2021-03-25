import tensorflow as tf

default_image_size = 224
default_image_depth = 3

resnet_map = {
    'resnet50': tf.keras.applications.ResNet50,
    #'resnet101': tf.keras.applications.ResNet101,
    #'resnet152': tf.keras.applications.ResNet152,
    #'resnet50v2': tf.keras.applications.ResNet50V2,
    #'resnet101v2': tf.keras.applications.ResNet101V2,
    #'resnet152v2': tf.keras.applications.ResNet152V2
    }

def get_uncompiled_model(name, class_num, image_size=default_image_size, image_depth=default_image_depth, trainable=False, weights='imagenet', **kwargs):
    
    backbone_net = resnet_map[name](input_shape=(image_size,image_size,image_depth), include_top=False, weights=weights, **kwargs)
    backbone_net.trainable=trainable

    x=backbone_net(backbone_net.inputs)
    x=tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x=tf.keras.layers.Dense(class_num, activation = 'softmax', name='predictions')(x)
    
    model=tf.keras.Model(backbone_net.inputs, x)
    
    return model

