from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import numpy as np
import torch.utils.model_zoo as model_zoo
from .DRN_ori import drn_d_54,drn_c_58

__all__ = ['DRN2']

model_urls = {

    'drn-d-54': 'https://tigress-web.princeton.edu/~fy/drn/models/drn_d_54-0e0534ff.pth',
}

class DRN2(nn.Module):
    __factory = {
        54: drn_d_54,
    }

    def __init__(self, depth, pretrained=True, num_features=0, 
        norm=False, dropout=0, num_classes=751):
        super(DRN2, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in DRN2.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = drn_d_54(pretrained=False)

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.num_classes = num_classes
        self.g_reduce = self._cbr2()
        self.g_classifier = nn.Linear(256, self.num_classes)
        self.l4_reduce = self._cbr2()
        self.l4_classifier = nn.Linear(256, self.num_classes)
        self.l7_reduce = self._cbr2()
        self.l7_classifier = nn.Linear(256, self.num_classes)

        self.reset_params()
        self.base.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))

    def _cbr2(self, in_channel=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            )
        return op_s

    def forward(self, x):
        side_output = {}
        # res_side=[]
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer4':
                side_output['layer4']=x
            # elif name == 'layer6':
            #     side_output['layer6']=x
            elif name == 'layer7':
                side_output['layer7']=x
            elif name == 'layer8':
                side_output['layer8']=x
            else:
                continue
        # pdb.set_trace()
        l2_maxp = side_output['layer4']  # drn2 ori: layer4,no multi
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4

        l3_maxp = side_output['layer7'] #batch*512*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        l4_maxp = side_output['layer8']
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4
        
        # xxx = side_output['layer4'] + side_output['layer7'] + side_output['layer8']
        # ll_attention_like_map = side_output['layer8']
        # no multiply performance is higher than multiply

        l4 = side_output['layer4']
        l4 = F.avg_pool2d(l4, l4.size()[2:])
        l4_rd = self.l4_reduce(l4)
        l4_rd = l4_rd.squeeze()
        l4_sm = self.g_classifier(l4_rd)

        l7 = side_output['layer7']
        l7 = F.avg_pool2d(l7, l7.size()[2:])
        l7_rd = self.l7_reduce(l7)
        l7_rd = l7_rd.squeeze()
        l7_sm = self.l7_classifier(l7_rd)



        g = side_output['layer8']
        g = F.avg_pool2d(g, g.size()[2:])
        g_rd = self.g_reduce(g)
        g_rd = g_rd.squeeze()
        g_sm = self.g_classifier(g_rd)


        fea_trip = [g_rd,l7_rd,l4_rd]
        scores = [g_sm,l7_sm,l4_sm]
        fea_for_test = [g_rd,l7_rd,l4_rd]
        fea_for_test = torch.cat(fea_for_test, 1)
        if self.norm:
            fea_for_test = F.normalize(fea_for_test)
        return scores, fea_trip, l2_side, l3_side, l4_side, fea_for_test

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

