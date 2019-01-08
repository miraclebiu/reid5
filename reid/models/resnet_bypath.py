from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import copy
from .layer import convLR, convDU, MultiHeadAttention, _initialize_weights, parallel_bypathm, make_layer

if torch.__version__ == '0.3.0.post4':
    from torch.nn.init import constant as init_constant
    from torch.nn.init import kaiming_normal as init_kaiming_normal
    from torch.nn.init import normal as init_normal
else:
    from torch.nn.init import constant_ as init_constant
    from torch.nn.init import kaiming_normal_ as init_kaiming_normal
    from torch.nn.init import normal_ as init_normal

__all__ = ['ResNet_bypath']





class ResNet_bypath(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, num_bypath = 3, pretrained=True, num_features=256, 
        norm=False, dropout=0, num_classes=0, add_l3_softmax=False, test_stage=False):
        super(ResNet_bypath, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.add_l3_softmax = add_l3_softmax
        self.num_bypath = num_bypath
        self.test_stage = test_stage


        self.norm = norm
        self.dropout = dropout
        self.sac = False

        self.num_features = num_features
        self.num_classes = num_classes

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)


        # Construct base (pretrained) resnet
        if depth not in ResNet_bypath.__factory:
            raise KeyError("Unsupported depth:", depth)
        base = ResNet_bypath.__factory[depth](pretrained=pretrained)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # for other bypath
        self.layer4_multi = make_layer(inplanes=1024, planes=512, blocks=3, stride=1, groups=self.num_bypath-1)


        self.layer4_reduce = [self._cbr2(in_channels = 2048, out_channels = num_features)] * (self.num_bypath)
        self.layer4_reduce = nn.ModuleList(self.layer4_reduce)
        self.layer4_cls = [nn.Linear(num_features, num_classes)] * (self.num_bypath+1)
        self.layer4_cls = nn.ModuleList(self.layer4_cls)

        self.channel_attention = MultiHeadAttention(2048,4)
        self.LR_attention = convLR(2048)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            _initialize_weights(self.modules())


    def _cbr2(self, in_channel=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
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
        x = self.layer2(x)
        l3_out = self.layer3(x)

        if self.sac:
            l2_maxp = self.layer2_maxpool(x)
            l2_side =  F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))

            l3_maxp = self.layer3_maxpool(l3_out)  #batch*1024*8*4
            l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))

        l4_fea = []

		
		l4_ori = self.layer4(l3_out)
		l4_side = F.normalize(l4_ori.pow(2).mean(1).view(l4_ori.size(0),-1)).view(l4_ori.size(0),1,l4_ori.size(2),l4_ori.size(3))
		l4_fea.append(l4_ori)

		l3_multi = torch.cat([l3_out] * self.num_bypath, 1)
		l4_multi = self.layer4_multi(l3_multi)
		l4_outs = l4_multi.split(2048,1)
		
		l4_c = l4_outs[0]
        channel_prior = l4_c.view(l4_c.size(0),l4_c.size(1),-1).transpose(2,1)
        x_channel, attention_weight = self.channel_attention(channel_prior,channel_prior,channel_prior)
        l4_ca = x_channel.transpose(2,1).view(l4_c.size())
        l4_fea.append(l4_ca)

        l4_lr = self.LR_attention(l4_outs[1])
        l4_fea.append(l4_lr)
        
        g_rd = []
        g_sm = []


        for cur_red, cur_cls, curl4_fea in zip(self.layer4_reduce, self.layer4_cls,l4_fea):


            l4_avg = F.avg_pool2d(curl4_fea,curl4_fea.size()[2:])
            l4_rd = cur_red(l4_avg)
            l4_rd = l4_rd.squeeze()
            if self.norm:
                l4_rd = F.normalize(l4_rd)
            l4_sm = cur_cls(l4_rd)

            g_rd.append(l4_rd)
            g_sm.append(l4_sm)

        fea = torch.cat(g_rd,1)

        if self.test_stage:
            if self.sac:
                return attention_weight, g_sm, g_rd, l2_side, l3_side, l4_side, fea
            else:
                return attention_weight, g_sm, g_rd, fea

        if self.sac:
            return g_sm, g_rd, l2_side, l3_side, l4_side, fea
        else:
            return g_sm, g_rd, fea


