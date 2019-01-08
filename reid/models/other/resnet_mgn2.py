import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn import init


import pdb


__all__ = [ 'resnet50_mgn2', 'resnet101_mgn2']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
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


class ResNet_mgn2(nn.Module):

    def __init__(self, block, layers, num_features=0, norm=False, dropout=0, num_classes=751):
        self.inplanes = 64
        super(ResNet_mgn2, self).__init__()
        self.num_features = num_features
        self.norm = norm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3_0 = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=2,is_last=True)
        
        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=2,is_last=False)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=1,is_last=False)
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=1,is_last=True)

        # pdb.set_trace()
        self.classifier_g1 = nn.Linear(512 * block.expansion, num_classes)
        self.g1_reduce = self._cbr2()
        
        self.classifier_g2   = nn.Linear(512 * block.expansion, num_classes)
        self.g2_reduce = self._cbr2()
        self.classifier_p2_1 = nn.Linear(256, num_classes)
        self.p2_1_reduce = self._cbr2()
        self.classifier_p2_2 = nn.Linear(256, num_classes)
        self.p2_2_reduce = self._cbr2()
        self.classifier_p2 = nn.Linear(256*2, num_classes)

        self.classifier_g3   = nn.Linear(512 * block.expansion, num_classes)
        self.g3_reduce = self._cbr2()
        self.classifier_p3_1 = nn.Linear(256, num_classes)
        self.p3_1_reduce = self._cbr2()
        self.classifier_p3_2 = nn.Linear(256, num_classes)
        self.p3_2_reduce = self._cbr2()
        self.classifier_p3_3 = nn.Linear(256, num_classes)
        self.p3_3_reduce = self._cbr2()
        self.classifier_p3 = nn.Linear(256*3, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
                # init.normal(m.weight, std=0.001) # can change to kaiming_normal
                # if m.bias is not None:
                #     init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight,1)
                init.constant(m.bias,0)


    def _cbr(self, in_channel=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Linear(in_channel, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
            )
        return op_s

    def _cbr2(self, in_channel=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            )
        return op_s

    def _make_layer(self, block, planes, blocks, stride=1,is_last=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # if is_last:
        self.inplanes = planes * block.expansion       
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if not is_last:
            # pdb.set_trace()
            self.inplanes = self.inplanes//2


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        g1 = self.layer3_0(x)
        g1 = self.layer4_0(g1)
        g1_mp = F.max_pool2d(g1,g1.size()[2:])
        # pdb.set_trace()
        g1_mp_s = g1_mp.squeeze()
        g1_sm = self.classifier_g1(g1_mp_s)
        g1_rd = self.g1_reduce(g1) # 2048->256
        g1_rd = F.max_pool2d(g1_rd,g1_rd.size()[2:])
        g1_rd = g1_rd.squeeze()



        g2 = self.layer3_1(x)
        g2 = self.layer4_1(g2)
        g2_mp = F.max_pool2d(g2, g2.size()[2:])
        g2_mp_s = g2_mp.squeeze()
        g2_sm = self.classifier_g2(g2_mp_s)
        g2_rd = self.g2_reduce(g2)
        g2_rd = F.max_pool2d(g2_rd,g2_rd.size()[2:])
        g2_rd = g2_rd.squeeze()

        h2 = g2.size(2)//2
        p2_1 = g2[:,:,0:h2,:]
        p2_1_rd = self.p2_1_reduce(p2_1)
        p2_1_rd = F.max_pool2d(p2_1_rd,p2_1_rd.size()[2:])
        p2_1_rd = p2_1_rd.squeeze()
        p2_1_sm = self.classifier_p2_1(p2_1_rd)

        p2_2 = g2[:,:,h2:,:]
        p2_2_rd = self.p2_2_reduce(p2_2)
        p2_2_rd = F.max_pool2d(p2_2_rd,p2_2_rd.size()[2:])
        p2_2_rd = p2_2_rd.squeeze()
        p2_2_sm = self.classifier_p2_2(p2_2_rd)



        g3 = self.layer3_2(x)
        g3 = self.layer4_2(g3)
        g3_mp = F.max_pool2d(g3,g3.size()[2:])
        g3_mp_s = g3_mp.squeeze()
        g3_sm = self.classifier_g3(g3_mp_s)
        g3_rd = self.g3_reduce(g3)
        g3_rd = F.max_pool2d(g3_rd,g3_rd.size()[2:])
        g3_rd = g3_rd.squeeze()

        h3 = g3.size(2)//3

        p3_1 = g3[:,:,0:h3,:]
        p3_1_rd = self.p3_1_reduce(p3_1)
        p3_1_rd = F.max_pool2d(p3_1_rd,p3_1_rd.size()[2:])
        p3_1_rd = p3_1_rd.squeeze()
        p3_1_sm = self.classifier_p3_1(p3_1_rd)

        p3_2 = g3[:,:,h3:h3*2,:]
        p3_2_rd = self.p3_2_reduce(p3_2)
        p3_2_rd = F.max_pool2d(p3_2_rd,p3_2_rd.size()[2:])
        p3_2_rd = p3_2_rd.squeeze()
        p3_2_sm = self.classifier_p3_2(p3_2_rd)

        p3_3 = g3[:,:,h3*2:h3*3,:]
        p3_3_rd = self.p3_3_reduce(p3_3)
        p3_3_rd = F.max_pool2d(p3_3_rd,p3_3_rd.size()[2:])
        p3_3_rd = p3_3_rd.squeeze()
        p3_3_sm = self.classifier_p3_3(p3_3_rd)


        fea_trip = [g1_rd,g2_rd,g3_rd]
        scores = [g1_sm,g2_sm,p2_1_sm,p2_2_sm,g3_sm,p3_1_sm,p3_2_sm,p3_3_sm]

        fea_test = [g1_rd,g2_rd,p2_1_rd,p2_2_rd,g3_rd,p3_1_rd,p3_2_rd,p3_3_rd]
        fea_for_test = torch.cat(fea_test,1)
        if self.norm:
            fea_for_test = F.normalize(fea_for_test)

        return  scores, fea_trip, fea_for_test

def copy_weight_for_branches(model_name, new_model, branch_num=3):
    model_weight = model_zoo.load_url(model_urls[model_name])
    model_keys = model_weight.keys()

    new_model_weight = new_model.state_dict()
    new_model_keys = new_model_weight.keys()
    # new_model_weight = copy.deepcopy(model_weight)
    handled =[]
    for block in new_model_keys:
        # print(block)
        prefix = block.split('.')[0]
        ori_prefix = prefix.split('_')[0]
        suffix = block.split('.')[1:]
        ori_key_to_join = [ori_prefix] + suffix
        ori_key = '.'.join(ori_key_to_join)
        if(ori_prefix == 'layer3') or (ori_prefix =='layer4'):
            for i in range(0,branch_num):
                if ori_key in model_keys:
                    # print(new_model_weight[block].size(),model_weight[ori_key].size())
                    new_model_weight[block] = model_weight[ori_key]
                else:
                    continue
                # pdb.set_trace()
                # handled.append(ori_key)
        elif ori_key in model_keys:
            new_model_weight[block] = model_weight[block]
    save_name = model_name + '_mgn2.tar'
    torch.save(new_model_weight,save_name)


def resnet50_mgn2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mgn2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # pdb.set_trace()
        copy_weight_for_branches('resnet50',model)
        model.load_state_dict(torch.load('./resnet50_mgn2.tar'))

    # resnet50_w = model_zoo.load_url(model_urls['resnet50']) 
    # qqq = model.state_dict()
    # qzq = qqq['layer4_0.0.conv1.weight']
    # bbb = resnet50_w['layer4.0.conv1.weight']
    # pdb.set_trace()
    return model


def resnet101_mgn2(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mgn2(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        copy_weight_for_branches('resnet101',model)
        model.load_state_dict(torch.load('./resnet101_mgn2.tar'))
    return model



