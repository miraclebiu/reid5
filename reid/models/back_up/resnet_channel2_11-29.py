from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from .layer import MultiHeadAttention, _initialize_weights, parallel_bypath, make_layer
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


### l234ca and l234 cls and without modify stride in the layer4
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

        self.num_features = num_features
        self.num_classes = num_classes

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        

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
        self.layer4 = base.layer4


        self.layer2_ca = MultiHeadAttention(512,8)
        self.layer3_ca = MultiHeadAttention(1024,8)
        self.layer4_ca = MultiHeadAttention(2048,8)

        self.layer2_reduce = self._cbr2(in_channels = 512, out_channels = num_features)
        self.layer3_reduce = self._cbr2(in_channels = 1024, out_channels = num_features)
        self.layer4_reduce = self._cbr2(in_channels = 2048, out_channels = num_features)

        
        self.layer2_cls = nn.Linear(num_features, num_classes)
        self.layer3_cls = nn.Linear(num_features, num_classes)
        self.layer4_cls = nn.Linear(num_features, num_classes)


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
        l2_reshape = l2_out.view(l2_out.size(0), l2_out.size(1), -1).transpose(2, 1)
        l2_ca, l2_attw = self.layer2_ca(l2_reshape, l2_reshape, l2_reshape)
        l2_ca = l2_ca.transpose(2, 1).view(l2_out.size())
        # l2_ca = F.relu(l2_ca,inplace=True)


        l3_out = self.layer3(l2_ca)
        l3_reshape = l3_out.view(l3_out.size(0), l3_out.size(1), -1).transpose(2, 1)
        l3_ca, l3_attw = self.layer3_ca(l3_reshape, l3_reshape, l3_reshape)
        l3_ca = l3_ca.transpose(2, 1).view(l3_out.size())
        # l3_ca = F.relu(l3_ca,inplace=True)

        l4_out = self.layer4(l3_ca)
        l4_reshape = l4_out.view(l4_out.size(0), l4_out.size(1), -1).transpose(2, 1)
        l4_ca, l4_attw = self.layer4_ca(l4_reshape, l4_reshape, l4_reshape)
        l4_ca = l4_ca.transpose(2, 1).view(l4_out.size())
        # l4_ca = F.relu(l4_ca,inplace=True)

        l2_maxp = self.layer2_maxpool(l2_ca)
        l2_column = l2_maxp.pow(2).mean(1).view(l2_maxp.size(0), -1)
        l2_column_min = l2_column.min(1)[0].unsqueeze(1)
        l2_column = l2_column - l2_column_min
        l2_side = F.softmax(l2_column, 1).view(l2_maxp.size(0), 1, l2_maxp.size(2), l2_maxp.size(3))

        l3_maxp = self.layer3_maxpool(l3_ca)  # batch*1024*8*4
        l3_column = l3_maxp.pow(2).mean(1).view(l3_maxp.size(0), -1)
        l3_column_min = l3_column.min(1)[0].unsqueeze(1)
        l3_column = l3_column - l3_column_min
        l3_side = F.softmax(l3_column, 1).view(l3_maxp.size(0), 1, l3_maxp.size(2), l3_maxp.size(3))

        l4_column = l4_ca.pow(2).mean(1).view(l4_ca.size(0), -1)
        l4_column_min = l4_column.min(1)[0].unsqueeze(1)
        l4_column = l4_column - l4_column_min
        l4_side = F.softmax(l4_column, 1).view(l4_ca.size(0), 1, l4_ca.size(2), l4_ca.size(3))

        l2_avg = F.avg_pool2d(l2_ca, l2_ca.size()[2:])
        l2_rd = self.layer2_reduce(l2_avg)
        l2_rd = l2_rd.squeeze()
        l2_sm = self.layer2_cls(l2_rd)

        l3_avg = F.avg_pool2d(l3_ca, l3_ca.size()[2:])
        l3_rd = self.layer3_reduce(l3_avg)
        l3_rd = l3_rd.squeeze()
        l3_sm = self.layer3_cls(l3_rd)

        if self.sac:
            l23_SideAdd = torch.add(l2_side,l3_side)
            SideAdd = torch.add(l23_SideAdd,l4_side)
            l4_ca = torch.mul(l4_ca,SideAdd)

        l4_avg = F.avg_pool2d(l4_ca, l4_ca.size()[2:])
        l4_rd = self.layer4_reduce(l4_avg)
        l4_rd = l4_rd.squeeze()
        l4_sm = self.layer4_cls(l4_rd)

        g_rd = [l2_rd, l3_rd, l4_rd]
        g_sm = [l2_sm, l3_sm, l4_sm]


        # pdb.set_trace()

        if self.norm:
            for each in g_rd:
                each = F.normalize(each)
            # net_rd = F.normalize(net_rd)

        # if self.dropout > 0:
        #     net_rd = self.drop(net_rd)
        net_rd = torch.cat(g_rd, 1)
        if self.test_stage:
            return [l2_attw,l3_attw,l4_attw], g_sm, g_rd, l2_side, l3_side, l4_side, net_rd
        return g_sm, g_rd, l2_side, l3_side, l4_side, net_rd

