from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from .layer import MultiHeadAttention, _initialize_weights, parallel_bypath, make_layer
#####run it in pytorch04
import copy

if torch.__version__ == '0.3.0.post4':
    from torch.nn.init import constant as init_constant
    from torch.nn.init import kaiming_normal as init_kaiming_normal
    from torch.nn.init import normal as init_normal
else:
    from torch.nn.init import constant_ as init_constant
    from torch.nn.init import kaiming_normal_ as init_kaiming_normal
    from torch.nn.init import normal_ as init_normal

__all__ = ['ResNet_channel2']

#############################l2,l3,l4 self-attention , l4 multiply sac and add extra supervised signal for l2_ca and l3_ca ,ca is channel attention

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
        self.sac = False

        self.norm = norm
        self.dropout = dropout
        self.num_bypath = 2

        self.num_features = num_features
        self.num_classes = num_classes

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        

        if depth not in ResNet_channel2.__factory:
            raise KeyError("Unsupported depth:", depth)
        base = ResNet_channel2.__factory[depth](pretrained=pretrained)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        self.layer3 = []
        for i in range(self.num_bypath):
            current_l3 = copy.deepcopy(base.layer3)
            self.layer3.append(current_l3)
        self.layer3 = nn.ModuleList(self.layer3)

        self.layer4 = []
        for i in range(self.num_bypath):
            current_l4 = copy.deepcopy(base.layer4)
            current_l4[0].conv2.stride = (1,1)
            current_l4[0].downsample[0].stride = (1,1)
            self.layer4.append(current_l4)
        self.layer4 = nn.ModuleList(self.layer4)

        self.layer3_ca = MultiHeadAttention(model_dim = 1024,  num_heads = 4, dropout = 0.0, compression = 2)
        self.layer4_ca = MultiHeadAttention(model_dim = 2048,  num_heads = 4, dropout = 0.0, compression = 2)

        self.layer3_reduce = [self._cbr2(in_channels = 1024, out_channels = num_features)] * (self.num_bypath)
        self.layer4_reduce = [self._cbr2(in_channels = 2048, out_channels = num_features)] * (self.num_bypath)
        self.l34_reduce = self.layer3_reduce + self.layer4_reduce
        self.l34_reduce = nn.ModuleList(self.l34_reduce)
        
        self.layer3_cls = [nn.Linear(num_features, num_classes)] * (self.num_bypath)
        self.layer4_cls = [nn.Linear(num_features, num_classes)] * (self.num_bypath)
        self.l34_cls = self.layer3_cls + self.layer4_cls
        self.l34_cls = nn.ModuleList(self.l34_cls)


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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        l2_out = self.layer2(x)
#####################################

        output_fea = []


        l3_fsa = self.layer3[0](l2_out)
        output_fea.append(l3_fsa)

        l3_fca = self.layer3[1](l2_out)

        l3_reshape = l3_fca.view(l3_fca.size(0), l3_fca.size(1), -1).transpose(2, 1)
        l3_ca, l3_attw = self.layer3_ca(l3_reshape, l3_reshape, l3_reshape)
        l3_ca = l3_ca.transpose(2, 1).view(l3_fca.size())

        output_fea.append(l3_ca)

        l3_add = l3_fsa + l3_ca
###############################################################

        l4_fsa = self.layer4[0](l3_add)
        output_fea.append(l4_fsa)    


        l4_fca = self.layer4[1](l3_add)
        l4_reshape = l4_fca.view(l4_fca.size(0), l4_fca.size(1), -1).transpose(2, 1)
        l4_ca, l4_attw = self.layer4_ca(l4_reshape, l4_reshape, l4_reshape)
        l4_ca = l4_ca.transpose(2, 1).view(l4_fca.size())

        output_fea.append(l4_ca)

        l4_add = l4_fsa + l4_ca


        l2_maxp = self.layer2_maxpool(l2_out)
        l2_column = l2_maxp.pow(2).mean(1).view(l2_maxp.size(0), -1)
        l2_column_min = l2_column.min(1)[0].unsqueeze(1)
        l2_column = l2_column - l2_column_min
        l2_side = F.softmax(l2_column, 1).view(l2_maxp.size(0), 1, l2_maxp.size(2), l2_maxp.size(3))

        l3_maxp = l3_add  # batch*1024*8*4
        l3_column = l3_maxp.pow(2).mean(1).view(l3_maxp.size(0), -1)
        l3_column_min = l3_column.min(1)[0].unsqueeze(1)
        l3_column = l3_column - l3_column_min
        l3_side = F.softmax(l3_column, 1).view(l3_maxp.size(0), 1, l3_maxp.size(2), l3_maxp.size(3))

        l4_maxp = l4_add
        l4_column = l4_maxp.pow(2).mean(1).view(l4_maxp.size(0), -1)
        l4_column_min = l4_column.min(1)[0].unsqueeze(1)
        l4_column = l4_column - l4_column_min
        l4_side = F.softmax(l4_column, 1).view(l4_maxp.size(0), 1, l4_maxp.size(2), l4_maxp.size(3))

        g_rd = []
        g_sm = []
        for cur_fea, cur_red, cur_cls in zip(output_fea, self.l34_reduce, self.l34_cls):
            cur_avg = F.avg_pool2d(cur_fea, cur_fea.size()[2:])
            cur_rd = cur_red(cur_avg)
            cur_rd = cur_rd.squeeze()
            cur_sm = cur_cls(cur_rd)
            g_rd.append(cur_rd)
            g_sm.append(cur_sm)
        net_rd = torch.cat(g_rd, 1)

        if self.norm:
            net_rd = F.normalize(net_rd)

        if self.dropout > 0:
            net_rd = self.drop(net_rd)
        # if self.test_stage:
        #     return [l2_attw,l3_attw,l4_attw], g_sm, g_rd, l2_side, l3_side, l4_side, net_rd
        return g_sm, g_rd, l2_side, l3_side, l4_side, net_rd

