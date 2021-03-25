from . import vgg19
from . import inceptionv3
from . import resnet
from . import densenet

networks_map = {
    'vgg19': vgg19,
    'inceptionv3': inceptionv3,
    'resnet50': resnet,
    'resnet101': resnet,
    'resnet152': resnet,
    'resnet50v2': resnet,
    'resnet101v2': resnet,
    'resnet152v2': resnet,
    'densenet121': densenet,
    'densenet169': densenet,
    'densenet201': densenet
    }

def get_uncompiled_model(name, *args, **kwargs):
    
    return networks_map[name].get_uncompiled_model(name, *args, **kwargs)

def get_default_size(name):
    return networks_map[name].default_image_size

