import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torch.nn import functional as F
from torch.nn import init


import pdb


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

__all__ = [ 'resnet50_aspp', 'resnet101_aspp']




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_aspp(nn.Module):

    def __init__(self, block, layers, num_features=0, norm=False, dropout=0, num_classes=751,pretrained=True):
        self.inplanes = 64
        super(ResNet_aspp, self).__init__()
        self.num_features = num_features
        self.norm = norm
        # self.rates = [1, 6, 12, 18]
        self.rates = [1, 2, 4, 8]
        # self.base = models.resnet50(pretrained=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, rate=1, is_last=False)
        self.layer4_mg_2 = self._make_MG_unit(block, 512, stride=1, rate=2, is_last=False)
        self.layer4_mg_4 = self._make_MG_unit(block, 512, stride=1, rate=4)

        # self.multi_grid = [1,2,4]
        # self.mg = self._make_MG_unit(block, 512, stride=1, rate=2)
        # test for only have different layer4 and remove aspp

        # self.aspp1 = self.aspp_module(2048, 256, rate=self.rates[0])
        # self.aspp2 = self.aspp_module(2048, 256, rate=self.rates[1])
        # self.aspp3 = self.aspp_module(2048, 256, rate=self.rates[2])
        # self.aspp4 = self.aspp_module(2048, 256, rate=self.rates[3])
        # self.image_pool = self._cbr(in_channel=2048)


        self.g1_reduce = self._cbr2(in_channel=2048,out_channels=256)
        self.classifier_g1 = nn.Linear(256, num_classes)

        self.g2_reduce = self._cbr2(in_channel=2048,out_channels=256)
        self.classifier_g2 = nn.Linear(256, num_classes)

        self.l4_reduce = self._cbr2(in_channel=2048, out_channels=256)
        self.classifier_l4 = nn.Linear(256, num_classes)


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
        # pdb.set_trace()
        if pretrained:
            print('load the pre-trained model.')
            resnet = models.resnet50(pretrained=pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            # pdb.set_trace()
            self.layer4_mg_2.load_state_dict(resnet.layer4.state_dict())
            self.layer4_mg_4.load_state_dict(resnet.layer4.state_dict())
            # torch.sum(self.layer4_mg[0].conv1.weight-resnet.layer4[0].conv1.weight)
            # pdb.set_trace()
        #     self.base.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        #     mg_unit = self.base.layer4 
        #     mg_unit[0].conv2.stride = (1,1)
        #     mg_unit[0].downsample[0].stride = (1,1)
        #     mg_unit[0].conv2.dilation = (self.multi_grid[0]*2,self.multi_grid[0]*2)
        #     mg_unit[0].conv2.padding = (self.multi_grid[0]*2,self.multi_grid[0]*2)

        #     mg_unit[1].conv2.dilation = (self.multi_grid[1]*2,self.multi_grid[1]*2)
        #     mg_unit[1].conv2.padding = (self.multi_grid[1]*2,self.multi_grid[1]*2)

        #     mg_unit[2].conv2.dilation = (self.multi_grid[2]*2,self.multi_grid[2]*2)
        #     mg_unit[2].conv2.padding = (self.multi_grid[2]*2,self.multi_grid[2]*2)
        #     self.basenet = nn.Sequential(self.base.conv1,self.base.bn1,self.base.relu,self.base.maxpool,self.base.layer1,self.base.layer2,self.base.layer3,self.base.layer4)
        # else:
        #     self.basenet = nn.Sequential(self.base.conv1,self.base.bn1,self.base.relu,self.base.maxpool,self.base.layer1,self.base.layer2,self.base.layer3,self.base.layer4)
        # # pdb.set_trace()


            # pdb.set_trace()
    def _cbr(self, in_channel=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        return op_s

    def _cbr2(self, in_channel=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            )
        return op_s

    def aspp_module(self, inplanes, planes, rate):
        op_s = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,dilation=rate, padding=rate),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )
        return op_s

    def _make_layer(self, block, planes, blocks, stride=1, rate=1, is_last=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        # if is_last:
        self.inplanes = planes * block.expansion       
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if not is_last:
            # pdb.set_trace()
            self.inplanes = self.inplanes//2


        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,2], stride=1, rate=1, is_last=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))
        if not is_last:
            # pdb.set_trace()
            self.inplanes = self.inplanes//2
        return nn.Sequential(*layers)

    def forward(self, x):
        # g1 = self.basenet(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)

        l4 = self.layer4(l3)
        g1 = self.layer4_mg_2(l3)
        g2 = self.layer4_mg_4(l3)
        # atrous1 = self.aspp1(g1)
        # atrous2 = self.aspp2(g1)
        # atrous3 = self.aspp3(g1)
        # atrous4 = self.aspp4(g1)
        # image_poo = self.image_pool(g1)
        # image_poo = F.upsample(image_poo, size=atrous4.size()[2:], mode='bilinear')
        # pdb.set_trace()

        # g_all = torch.cat([image_poo,atrous1,atrous2,atrous3,atrous4],1)
        g1_mp = F.avg_pool2d(g1, g1.size()[2:])
        g1_rd = self.g1_reduce(g1_mp)  # 1280->256
        g1_rd = g1_rd.squeeze()
        g1_sm = self.classifier_g1(g1_rd)


        g2_mp =  F.avg_pool2d(g2, g2.size()[2:])
        g2_rd = self.g2_reduce(g2_mp)  # 1280->256
        g2_rd = g2_rd.squeeze()
        g2_sm = self.classifier_g2(g2_rd)


        l4_mp = F.avg_pool2d(l4, l4.size()[2:])
        l4_rd = self.l4_reduce(l4_mp)
        l4_rd = l4_rd.squeeze()
        l4_sm = self.classifier_l4(l4_rd)


        fea_trip = [l4_rd,g1_rd,g2_rd]
        scores = [l4_sm,g1_sm,g2_sm]

        fea_test = [l4_rd,g1_rd,g2_rd]
        fea_for_test = torch.cat(fea_test, 1)
        if self.norm:
            fea_for_test = F.normalize(fea_for_test)
        return scores, fea_trip, fea_for_test


def resnet50_aspp(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_aspp(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101_aspp(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_aspp(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
