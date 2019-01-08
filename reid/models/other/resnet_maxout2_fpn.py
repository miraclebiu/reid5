from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb


__all__ = ['ResNet_MaxOut2_FPN']


class ResNet_MaxOut2_FPN(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=1024,
        norm=False, dropout=0, num_classes=0, ):
        super(ResNet_MaxOut2_FPN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet_MaxOut2_FPN.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet_MaxOut2_FPN.__factory[depth](pretrained=pretrained)
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = self.base.fc.in_features


        self.layer2_maxpool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.layer4_reduce = nn.Conv2d(2048,256,kernel_size=1,stride=1)
        self.layer3_reduce = nn.Conv2d(1024,256,kernel_size=1,stride=1)
        self.l43_fus = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        self.layer2_reduce = nn.Conv2d(512,256,kernel_size=1,stride=1)
        self.l32_fus = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        self.l4_bn = nn.BatchNorm2d(256)
        self.l3_bn = nn.BatchNorm2d(256)
        self.l2_bn = nn.BatchNorm2d(256)
        init.constant(self.l4_bn.weight, 1)
        init.constant(self.l4_bn.bias, 0)
        init.constant(self.l3_bn.weight, 1)
        init.constant(self.l3_bn.bias, 0)
        init.constant(self.l2_bn.weight, 1)
        init.constant(self.l2_bn.bias, 0)

        self.feat = nn.Linear(2048, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)

        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)


        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier_l2 = nn.Linear(256, self.num_classes)
            self.classifier_l3 = nn.Linear(256, self.num_classes)
            self.classifier_l4 = nn.Linear(256, self.num_classes)
            # self.classifier = nn.Linear(out_planes, self.num_classes)
            init.normal(self.classifier_l2.weight, std=0.001)
            init.constant(self.classifier_l2.bias, 0)
            init.normal(self.classifier_l3.weight, std=0.001)
            init.constant(self.classifier_l3.bias, 0)
            init.normal(self.classifier_l4.weight, std=0.001)
            init.constant(self.classifier_l4.bias, 0)

            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.classifier.weight, std=0.001)
            init.constant(self.classifier.bias, 0)
        if not self.pretrained:
            self.reset_params()

    # def _cbr2(self, in_channel=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
    #     op_s = nn.Sequential(
    #         nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
    #         nn.BatchNorm2d(out_channels),
    #         # nn.ReLU(inplace=True)
    #         )
    #     return op_s

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

        l4 = self.layer4_reduce(side_output['layer4'])
        l4_bn_out = self.l4_bn(l4)


        l4_up = F.upsample(l4,scale_factor=2,mode='bilinear')
        l3_reduce = self.layer3_reduce(side_output['layer3'])
        l43 = l3_reduce + l4_up
        l3 = self.l43_fus(l43)
        l3_bn_out = self.l3_bn(l3)

        l3_up = F.upsample(l3,scale_factor=2,mode='bilinear')
        l2_reduce = self.layer2_reduce(side_output['layer2'])
        l32 = l2_reduce+l3_up
        l2 = self.l32_fus(l32)
        l2_bn_out = self.l2_bn(l2)

        x_avg = F.avg_pool2d(x,x.size()[2:])
        x_avg = x_avg.view(x_avg.size(0),-1)
        l_all = self.feat(x_avg)
        l_all = self.feat_bn(l_all)
        l_score = self.classifier(l_all)


        # l2_maxp = self.layer2_maxpool(side_output['layer2'])  #batch*512*8*4
        # l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0),-1)).view(l2_maxp.size(0),1,l2_maxp.size(2),l2_maxp.size(3))  #batch*1*8*4

        # l3_maxp = self.layer3_maxpool(side_output['layer3'])  #batch*1024*8*4
        # l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        # l4_maxp = side_output['layer4']
        # l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4
        
        # ll_concate1 = torch.add(l2_side,l3_side)
        # ll_concate = torch.add(ll_concate1,l4_side)
        # ll_attention_like_map = x
        # ll_attention_like_map = torch.mul(x,ll_concate)
        # pdb.set_trace()
        # l2_side_flatten = l2_side.view(l2_side.size(0),-1)
        # l3_side_flatten = l3_side.view(l3_side.size(0), -1)
        # l4_side_flatten = l4_side.view(l4_side.size(0), -1)
        # side_feat = torch.cat([l2_side_flatten,l3_side_flatten,l4_side_flatten],1)
        # ll_attention_like_map = x
        l2_avg = F.avg_pool2d(l2_bn_out,l2_bn_out.size()[2:])
        l2_avg = l2_avg.view(l2_avg.size(0),-1)
        l2_score = self.classifier_l2(l2_avg)

        l3_avg = F.avg_pool2d(l3_bn_out,l3_bn_out.size()[2:])
        l3_avg = l3_avg.view(l3_avg.size(0),-1)
        l3_score = self.classifier_l3(l3_avg)

        l4_avg = F.avg_pool2d(l4_bn_out,l4_bn_out.size()[2:])
        l4_avg = l4_avg.view(l4_avg.size(0),-1)
        l4_score = self.classifier_l4(l4_avg)
        scores = [l_score, l4_score, l3_score, l2_score]
        fea_trip = [l_all, l4_avg, l3_avg, l2_avg]
        fea_for_test = torch.cat(fea_trip,1)

        if self.norm:
            fea_for_test = F.normalize(fea_for_test)

        return scores, fea_trip, fea_for_test

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
