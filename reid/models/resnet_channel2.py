from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from .layer import MultiHeadAttention, MultiHeadAttention_ca, _initialize_weights, parallel_bypath, make_layer

import copy

#####run it in pytorch04

if torch.__version__ == '0.3.0.post4':
    from torch.nn.init import constant as init_constant
    from torch.nn.init import kaiming_normal as init_kaiming_normal
    from torch.nn.init import normal as init_normal
else:
    from torch.nn.init import constant_ as init_constant
    from torch.nn.init import kaiming_normal_ as init_kaiming_normal
    from torch.nn.init import normal_ as init_normal

__all__ = ['ResNet_channel2']


class ResNet_channel2(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0,
                 norm=False, dropout=0, num_classes=0, add_l3_softmax=False, test_stage=False):
        super(ResNet_channel2, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.add_l3_softmax = add_l3_softmax
        self.test_stage = test_stage
        self.sac = True
        self.num_bypath = 2

        self.norm = norm
        self.dropout = dropout

        self.num_features = num_features
        self.num_classes = num_classes

        if depth not in ResNet_channel2.__factory:
            raise KeyError("Unsupported depth:", depth)
        base = ResNet_channel2.__factory[depth](pretrained=pretrained)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = []
        for i in range(self.num_bypath):
            current_l4 = copy.deepcopy(base.layer4)
            current_l4[0].conv2.stride = (1,1)
            current_l4[0].downsample[0].stride = (1,1)
            self.layer4.append(current_l4)
        self.layer4 = nn.ModuleList(self.layer4)


        self.layer2_mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3_ca = MultiHeadAttention_ca(spatial_size = 16 * 8, model_dim = 1024, middle_ss = 32 * 16, norm_type = 'bn')
        self.layer4_ca = MultiHeadAttention_ca(spatial_size = 16 * 8, model_dim = 2048, middle_ss = 32 * 16, norm_type = 'bn')

        self.layer2_reduce = self._cbr2(in_channels=512, out_channels=num_features)
        self.layer3_reduce = self._cbr2(in_channels=1024, out_channels=num_features)
        self.layer4_reduce = [self._cbr2(in_channels = 2048, out_channels = num_features)] * (self.num_bypath)
        self.layer4_reduce = nn.ModuleList(self.layer4_reduce)

        self.layer2_cls = nn.Linear(num_features, num_classes)
        self.layer3_cls = nn.Linear(num_features, num_classes)
        self.layer4_cls = [nn.Linear(num_features, num_classes)] * (self.num_bypath)
        self.layer4_cls = nn.ModuleList(self.layer4_cls)



        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            _initialize_weights(self.modules())

    def _cbr2(self, in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        return op_s

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l2_ca = self.layer2(x)

        l2_column = l2_ca.pow(2).mean(1).view(l2_ca.size(0), -1)
        l2_column_min = l2_column.min(1)[0].unsqueeze(1)
        l2_column = l2_column - l2_column_min
        l2_side = F.softmax(l2_column, 1).view(l2_ca.size(0), 1, l2_ca.size(2), l2_ca.size(3))
        l2_side = self.layer2_mp(l2_side)


        l3_out = self.layer3(l2_ca)
       
        l3_reshape = l3_out.view(l3_out.size(0), l3_out.size(1), -1)
        l3_ca, l3_attw = self.layer3_ca(l3_reshape, l3_reshape, l3_reshape)
        l3_ca = l3_ca.contiguous().view(l3_out.size())

        l3_column = l3_ca.pow(2).mean(1).view(l3_ca.size(0), -1)
        l3_column_min = l3_column.min(1)[0].unsqueeze(1)
        l3_column = l3_column - l3_column_min
        l3_side = F.softmax(l3_column, 1).view(l3_ca.size(0), 1, l3_ca.size(2), l3_ca.size(3))

        l4_fea = []
        l4_fsa = self.layer4[0](l3_ca)
        l4_fea.append(l4_fsa)

        l4_column = l4_fsa.pow(2).mean(1).view(l4_fsa.size(0), -1)
        l4_column_min = l4_column.min(1)[0].unsqueeze(1)
        l4_column = l4_column - l4_column_min
        l4_side = F.softmax(l4_column, 1).view(l4_fsa.size(0), 1, l4_fsa.size(2), l4_fsa.size(3))

        l4_fca = self.layer4[1](l3_ca)
        l4_reshape = l4_fca.view(l4_fca.size(0), l4_fca.size(1), -1)
        l4_ca, l4_attw = self.layer4_ca(l4_reshape, l4_reshape, l4_reshape)
        l4_ca = l4_ca.contiguous().view(l4_fca.size())


        l4_fea.append(l4_ca)



        # SideAdd = torch.add(l3_side, l4_side)
        # l4_fea = torch.mul(l4_fea, SideAdd)

        # l3_ca = torch.mul(l3_ca, SideAdd)
        l3_avg = F.avg_pool2d(l3_ca, l3_ca.size()[2:])
        l3_rd = self.layer3_reduce(l3_avg)
        l3_rd = l3_rd.squeeze()
        # l3_rd = l3_rd.view(batch_size,-1)
        l3_sm = self.layer3_cls(l3_rd)

        l4_rd = []
        l4_sm = []
        # current implementation ,layer4 outputs use different classifier!!! maybe use the same one is better
        for cur_red, cur_cls, cur_fea in zip(self.layer4_reduce, self.layer4_cls, l4_fea):

            cur_avg = F.avg_pool2d(cur_fea, cur_fea.size()[2:])
            cur_rd = cur_red(cur_avg)
            cur_rd = cur_rd.squeeze()
            # cur_rd = cur_rd.view(batch_size,-1)
            cur_sm = cur_cls(cur_rd)
            l4_rd.append(cur_rd)
            # l4_rd.append(cur_sm)
            l4_sm.append(cur_sm)


        g_rd = [l3_rd] + l4_rd
        g_sm = [l3_sm] + l4_sm



        if self.norm:
            net_rd = []
            for each in g_rd:
                net_rd.append(F.normalize(each))
        else:
            net_rd = g_rd
        net_rd = torch.cat(net_rd, 1)


        if self.dropout > 0:
            net_rd = self.drop(net_rd)

        if self.test_stage:
            # print('in')
            return l3_attw, l4_attw, g_sm, g_rd, l2_side, l3_side, l4_side, net_rd
        return g_sm, g_rd, l2_side, l3_side, l4_side, net_rd

