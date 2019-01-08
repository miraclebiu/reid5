from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import numpy as np

__all__ = ['DenseNet']


class DenseNet(nn.Module):
    __factory = {
        121: torchvision.models.densenet121,
    }

    def __init__(self, depth, pretrained=True, num_features=1024, 
        norm=False, dropout=0, num_classes=0):
        super(DenseNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in DenseNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = DenseNet.__factory[depth](pretrained=pretrained)


        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes
        # pdb.set_trace()

        out_planes = self.base.classifier.in_features


        # Append new layers
        if self.has_embedding:
            self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
            self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal(self.feat.weight, mode='fan_out')
            init.constant(self.feat.bias, 0)
            init.constant(self.feat_bn.weight, 1)
            init.constant(self.feat_bn.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.classifier.weight, std=0.001)
            init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        side_output = {};
        res_side=[]
        for name, module in self.base._modules.items():
            if name == 'features':
                temp = self.base._modules[name]
                pdb.set_trace()
                for name_sub, module_sub in temp :
                break
            # pdb.set_trace()
            
            x = module(x)
            if name == 'denseblock2':
                side_output['layer2']=x
            elif name == 'denseblock3':
                side_output['layer3']=x
            elif name == 'denseblock4':
                side_output['layer4']=x
            else:
                continue

        l2_maxp = self.layer2_maxpool(side_output['layer2'])  #batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4

        l3_maxp = self.layer3_maxpool(side_output['layer3'])  #batch*1024*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        l4_maxp = side_output['layer4']
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            #x = F.relu(x)
            pass
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x,l2_side,l3_side,l4_side

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

