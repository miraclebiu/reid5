from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import numpy as np

__all__ = ['ResNet']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, 
        norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)


        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = self.base.fc.in_features

        # Append new layers
        if self.has_embedding:
            self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
            self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
            self.l23_conv = nn.Conv2d(1024, 512,  kernel_size=3,stride=2,padding=1)
            self.l43_conv = nn.Conv2d(2048, 1024, kernel_size=3,stride=2,padding=1)
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal(self.l23_conv.weight, mode='fan_out')
            init.constant(self.l23_conv.bias, 0)
            init.kaiming_normal(self.l43_conv.weight, mode='fan_out')
            init.constant(self.l43_conv.bias, 0)
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
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer2':
                side_output['layer2']=x
            elif name == 'layer3':
                side_output['layer3']=x
            elif name == 'layer4':
                side_output['layer4']=x
            else:
                continue
        # pdb.set_trace()
        l2_maxp = self.layer2_maxpool(side_output['layer2'])  #batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4

        l3_maxp = self.layer3_maxpool(side_output['layer3'])  #batch*1024*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        l4_maxp = side_output['layer4']
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4
        # print(l2_side[127,:],l3_side[127,:],l4_side[127,:])
        # pdb.set_trace()


        x = F.avg_pool2d(l4_maxp, l4_maxp.size()[2:])
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
            cls_out = self.classifier(x)
        return cls_out,l2_side,l3_side,l4_side,x

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

