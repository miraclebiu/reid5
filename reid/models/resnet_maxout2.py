from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb

if torch.__version__ == '0.3.0.post4':
    from torch.nn.init import constant as init_constant
    from torch.nn.init import kaiming_normal as init_kaiming_normal
    from torch.nn.init import normal as init_normal
else:
    from torch.nn.init import constant_ as init_constant
    from torch.nn.init import kaiming_normal_ as init_kaiming_normal
    from torch.nn.init import normal_ as init_normal

__all__ = ['ResNet_MaxOut2']


class ResNet_MaxOut2(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, 
        norm=False, dropout=0, num_classes=0, add_l3_softmax=False, test_stage=False):
        super(ResNet_MaxOut2, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.before_cls = True
        self.test_stage=test_stage

        # Construct base (pretrained) resnet
        if depth not in ResNet_MaxOut2.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet_MaxOut2.__factory[depth](pretrained=pretrained)
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.num_classes = num_classes

        out_planes = self.base.fc.in_features

        # Append new layers

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #last embedding
        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)

        init_kaiming_normal(self.feat.weight, mode='fan_out')
        init_constant(self.feat.bias, 0)
        init_constant(self.feat_bn.weight, 1)
        init_constant(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            # self.classifier = nn.Linear(out_planes, self.num_classes)
            init_normal(self.classifier.weight, std=0.001)
            init_constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        side_output = {};
        for name, module in self.base._modules.items():
            # pdb.set_trace()
            if name == 'avgpool':
                break
            # print(name)
            x = module(x)
            if name == 'layer2':
                side_output['layer2']=x
            elif name == 'layer3':
                side_output['layer3']=x
            elif name == 'layer4':
                side_output['layer4']=x
        # pdb.set_trace()
        l2_maxp = self.layer2_maxpool(side_output['layer2'])  #batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4

        l3_maxp = self.layer3_maxpool(side_output['layer3'])  #batch*1024*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        l4_maxp = side_output['layer4']
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4
        
        ll_concate1 = torch.add(l2_side,l3_side)
        ll_concate = torch.add(ll_concate1,l4_side)
       
        ll_attention_like_map = torch.mul(x,ll_concate)
        # pdb.set_trace()
        # l2_side_flatten = l2_side.view(l2_side.size(0),-1)
        # l3_side_flatten = l3_side.view(l3_side.size(0), -1)
        # l4_side_flatten = l4_side.view(l4_side.size(0), -1)
        # side_feat = torch.cat([l2_side_flatten,l3_side_flatten,l4_side_flatten],1)
        # ll_attention_like_map = x
        x = F.avg_pool2d(ll_attention_like_map, ll_attention_like_map.size()[2:])
        x = x.view(x.size(0), -1)
        # x = torch.cat([x,side_feat],1)

        x = self.feat(x)
        x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)

        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            cls_out = self.classifier(x)
            trip_out = cls_out
        if self.test_stage:
            return cls_out,trip_out,l2_side,l3_side,l4_side,x   
        return cls_out,trip_out,l2_side,l3_side,l4_side,x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init_constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init_constant(m.weight, 1)
                init_constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_normal(m.weight, std=0.001)
                if m.bias is not None:
                    init_constant(m.bias, 0)
