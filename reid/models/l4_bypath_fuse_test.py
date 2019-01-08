import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from copy import deepcopy
from torch.nn.parameter import Parameter


def parallel_bypath(module, num_bypath, enlarge_rf=True):
    print(module)
    if isinstance(module, Bottleneck):
        for sub_module in module.children():
            parallel_bypath(sub_module, num_bypath)
    elif isinstance(module, nn.Sequential):
        for sub_module in module.children():
            parallel_bypath(sub_module, num_bypath)
    elif isinstance(module, nn.Conv2d):
        module.in_channels = module.in_channels * num_bypath
        module.out_channels = module.out_channels * num_bypath
        module.groups = num_bypath
        new_weight = [module.weight] * num_bypath
        module.weight = Parameter(torch.cat(new_weight, 0))
        if enlarge_rf and module.stride == (2, 2):
            module.stride = (1, 1)
        if module.bias is not None :
            new_bias = [module.bias] * num_bypath
            module.bias = Parameter(torch.cat(new_bias, 0))
    elif isinstance(module, nn.BatchNorm2d):

        module.num_features = module.num_features * num_bypath
        new_weight = [module.weight] * num_bypath
        module.weight = Parameter(torch.cat(new_weight, 0))

        new_running_mean = [module.running_mean] * num_bypath
        module.running_mean = torch.cat(new_running_mean, 0)

        new_running_var = [module.running_var] * num_bypath
        module.running_var = torch.cat(new_running_var, 0)
        if module.bias is not None :
            new_bias = [module.bias] * num_bypath
            module.bias = Parameter(torch.cat(new_bias, 0))
    elif isinstance(module, nn.ReLU):
    	return
    else:

        raise RuntimeError('not conv or bn or bottleneck or sequential')


import pdb

res50 = models.resnet50(pretrained=True)
layer4 = res50.layer4

layer4_3path = deepcopy(layer4)
pdb.set_trace()
num_bypath = 3
for sub_module in layer4_3path.children():
    parallel_bypath(sub_module, num_bypath)
test = torch.rand(2,1024*3,16,8)
pdb.set_trace()
print(layer4_3path)
