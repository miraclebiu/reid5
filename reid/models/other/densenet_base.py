import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import pdb

__all__ = ['DenseNet_Biu']


# class _DenseLayer(nn.Module):   
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseLayer, self).__init__()
#         self.norm1 = nn.BatchNorm2d(num_input_features)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(num_input_features, bn_size *
#                         growth_rate, kernel_size=1, stride=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=1, bias=False)
#         self.drop_rate = drop_rate

#     def forward(self, x):
#         # pdb.set_trace()
#         new_features = self.norm1(x)
#         new_features = self.relu1(new_features)
#         new_features = self.conv1(new_features)

#         new_features = self.norm2(new_features)
#         new_features = self.relu2(new_features)
#         new_features = self.conv2(new_features)

#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return torch.cat([x, new_features], 1)


# class _DenseBlock(nn.Sequential):
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
#             self.add_module('denselayer%d' % (i + 1), layer)


# class _Transition(nn.Module):
#     def __init__(self, num_input_features, num_output_features):
#         super(_Transition, self).__init__()
#         self.norm = nn.BatchNorm2d(num_input_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv2d(num_input_features, num_output_features,
#                                           kernel_size=1, stride=1, bias=False)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#     def forward(self,x):
#         out = self.norm(x)
#         out = self.relu(out)
#         out = self.conv(out)
#         return self.pool(out)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_Biu(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 num_classes=751, num_features=1024,norm=False, dropout=0, use_side_out=True):

        super(DenseNet_Biu, self).__init__()

        self.num_features = num_features
        self.dropout = dropout
        self.pretrained = pretrained
        self.use_side_out = use_side_out
        # First convolution
    
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_layers = block_config
        num_features = num_init_features

        self.layer1 = _DenseBlock(num_layers[0], num_features, bn_size, growth_rate, drop_rate)
        num_features = num_features + num_layers[0] * growth_rate
        self.trans1 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer2 = _DenseBlock(num_layers[1], num_features, bn_size, growth_rate, drop_rate)
        num_features = num_features + num_layers[1] * growth_rate
        self.trans2 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.layer3 = _DenseBlock(num_layers[2], num_features, bn_size, growth_rate, drop_rate)
        num_features = num_features + num_layers[2] * growth_rate
        self.trans3 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer4 = _DenseBlock(num_layers[3], num_features, bn_size, growth_rate, drop_rate)
        num_features = num_features + num_layers[3] * growth_rate



        self.features_norm5 = nn.BatchNorm2d(num_features)

        # Linear layer
        self.drop = nn.Dropout(self.dropout)
        self.feat = nn.Linear(num_features,self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.classifier = nn.Linear(self.num_features, num_classes)
        if not self.pretrained:
            self.reset_params()
        # self.base = nn.Sequential(self.layer1,self.trans1,self.layer2,self.trans2,self.layer3,self.trans3,self.layer4,self.features_norm5)

    def forward(self, x):
        out = self.features(x)

        out = self.layer1(out)
        out = self.trans1(out)

        out = self.layer2(out)
        out = self.trans2(out)

        l2_maxp = self.layer2_maxpool(out)  #batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))
        
        out = self.layer3(out)
        out = self.trans3(out)

        l3_side = F.normalize(out.pow(2).mean(1).view(out.size(0),-1)).view(out.size(0),1,out.size(2),out.size(3))  #batch*1*8*4


        out = self.layer4(out)
        out =self.features_norm5(out)
        out = F.relu(out, inplace=True)
        l4_side = F.normalize(out.pow(2).mean(1).view(out.size(0),-1)).view(out.size(0),1,out.size(2),out.size(3))  #batch*1*8*4

        if self.use_side_out:
            # pdb.set_trace()
            ll_concate1 = torch.add(l2_side,l3_side) 
            ll_concate = torch.add(ll_concate1,l4_side) 

            out =torch.mul(out,ll_concate)


        out = F.avg_pool2d(out, out.size()[2:]).view(x.size(0), -1)
        out = self.feat(out)
        out = self.feat_bn(out)
        # add relu for test in 2018-03-14
        # remove relu in 18-03-19
        # out = F.relu(out)
        if self.dropout >0:
            out = self.drop(out)
        out = self.classifier(out)
        return out,l2_side,l3_side,l4_side


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)                
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
