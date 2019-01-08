from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import numpy as np
from .DRN_ori import drn_d_54

__all__ = ['DRN_1']


class DRN_1(nn.Module):
    __factory = {
        54: drn_d_54,
    }

    def __init__(self, depth, pretrained=True, num_features=0, 
        norm=False, dropout=0, num_classes=0):
        super(DRN_1, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in DRN_1.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = drn_d_54(pretrained=pretrained)


        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes
        # pdb.set_trace()
        out_planes = (self.base.fc.in_channels) # DRN's fc is a Conv2d layer

        # Append new layers
        if self.has_embedding:

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
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer4':
                side_output['layer4']=x
            elif name == 'layer7':
                side_output['layer7']=x
            elif name == 'layer8':
                side_output['layer8']=x
            else:
                continue
        l2_maxp = side_output['layer4']  #batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4


        l3_maxp = side_output['layer7'] #batch*1024*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4


        l4_maxp = side_output['layer8']
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4
        # print(l2_side[127,:],l3_side[127,:],l4_side[127,:])
        # pdb.set_trace()
        ll_concate1 = torch.add(l2_side,l3_side) 
        ll_concate = torch.add(ll_concate1,l4_side) 

        xxx = l2_maxp + l3_maxp + l4_maxp
        ll_attention_like_map = torch.mul(xxx,ll_concate)
        # concatenation is not kaopu! try add
        # ll_attention_like_map_layer6 = torch.mul(l2_maxp, l2_side)
        # ll_attention_like_map_layer7 = torch.mul(l3_maxp, l3_side)
        # ll_attention_like_map = torch.cat([ll_attention_like_map,ll_attention_like_map_layer7,ll_attention_like_map_layer6],1)

        x = F.avg_pool2d(ll_attention_like_map, ll_attention_like_map.size()[2:])
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

