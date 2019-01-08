from __future__ import absolute_import

from .resnet_mgn import ResNet_mgn_model
from .resnet_mgn_lr import ResNet_mgn_lr_model
from .resnet_maxout2 import ResNet_MaxOut2
from .resnet_reid import ResNet_reid
from .resnet_channel import ResNet_channel
from .resnet_channel2 import ResNet_channel2
from .resnet_channel3 import ResNet_channel3
from .resnet_bypath import ResNet_bypath


__factory = {
    'resnet_mgn_lr': ResNet_mgn_lr_model,
    'resnet_mgn': ResNet_mgn_model,
    'resnet_maxout2': ResNet_MaxOut2,
    'resnet_reid': ResNet_reid,
    'resnet_channel': ResNet_channel,
    'resnet_channel2': ResNet_channel2,
    'resnet_channel3': ResNet_channel3,
    'resnet_bypath': ResNet_bypath,

}



def names():
    return sorted(__factory.keys())

def model_creator(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        The model name. Can be one of 'resnet_mgn', 'resnet_mgn_lr', 'resnet_maxout2',
        'resnet_reid'.
    root : str
        The path to the model directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the model. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)