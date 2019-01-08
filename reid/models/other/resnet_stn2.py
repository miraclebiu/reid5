import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn import init


import pdb


__all__ = [ 'resnet50_stn2', 'resnet101_stn2']


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


class ResNet_stn2(nn.Module):

    def __init__(self, block, layers, num_features=0, norm=False, dropout=0, num_classes=751):
        self.inplanes = 64
        super(ResNet_stn2, self).__init__()
        self.num_features = num_features
        self.norm = norm

        self.layer2_maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2_0 = self._make_layer(block, 128, layers[1], stride=2,is_last=False)
        self.layer2_1 = self._make_layer(block, 128, layers[1], stride=2,is_last=True)

        self.layer3_0 = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=2,is_last=True)

        
        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=2,is_last=False)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=1,is_last=True)

        self.se_fc1 = nn.Linear(2048,128)
        self.se_relu = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Linear(128,2048)
        self.se_act = nn.Sigmoid()


        self.classifier_g1 = nn.Linear(256, num_classes)
        self.g1_reduce = self._cbr2()

        self.stn = self.stn_loc_f()
        self.stn_loc = nn.Linear(256,6)

        self.classifier_g2 = nn.Linear(256, num_classes)
        self.g2_reduce = self._cbr2()

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
        self.stn_loc.weight.data.fill_(0)
        self.stn_loc.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0]) # this need to be paid attention


    def stn_loc_f(self):
        op = nn.Sequential(
            nn.Conv2d(2048,256,kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(8,4)),# position can be change
            )
        return op

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
        g1 = self.layer2_0(x)
        l2_maxp = self.layer2_maxpool(g1)  # batch*512*8*4
        l2_side = F.normalize(l2_maxp.pow(2).mean(1).view(l2_maxp.size(0), -1)).view(l2_maxp.size(0), 1,l2_maxp.size(2), l2_maxp.size(3))  # batch*1*8*4

        g1 = self.layer3_0(g1)
        l3_maxp = self.layer3_maxpool(g1)  #batch*1024*8*4
        l3_side = F.normalize(l3_maxp.pow(2).mean(1).view(l3_maxp.size(0),-1)).view(l3_maxp.size(0),1,l3_maxp.size(2),l3_maxp.size(3))  #batch*1*8*4

        g1_l = self.layer4_0(g1)
        l4_maxp = g1_l
        l4_side = F.normalize(l4_maxp.pow(2).mean(1).view(l4_maxp.size(0),-1)).view(l4_maxp.size(0),1,l4_maxp.size(2),l4_maxp.size(3))  #batch*1*8*4

        g1_2 = g1_l*(l2_side+l3_side+l4_side)

        g1_mp = F.avg_pool2d(g1_2, g1_2.size()[2:])
        ##########################se module################################
        g1_sq = g1_mp.squeeze()
        se_1 = self.se_fc1(g1_sq)
        se_relu1 = self.se_relu(se_1)
        se_2 = self.se_fc2(se_relu1)
        se_sig = self.se_act(se_2)
        channel_att = se_sig.unsqueeze(2).unsqueeze(3)
        g1_mp = g1_mp*channel_att

        # pdb.set_trace()
        ###########################se_module_end_for_test###############

        g1_rd = self.g1_reduce(g1_mp)  # 2048->256
        g1_rd = g1_rd.squeeze()
        g1_sm = self.classifier_g1(g1_rd)

        # loc_info = self.stn(g1_l)
        # loc_info = loc_info.view(-1,256)
        # loc_info = self.stn_loc(loc_info)
        # loc_info = loc_info.view(-1,2,3)
        #
        # transf = F.affine_grid(loc_info, x.size())
        # x_tf = F.grid_sample(x, transf)
        # # qzq=torch.sum(x_tf-x)
        # # pdb.set_trace()
        #
        # g2 = self.layer2_1(x_tf)
        # g2 = self.layer3_1(g2)
        # g2 = self.layer4_1(g2)
        # g2_mp = F.avg_pool2d(g2, g2.size()[2:])
        # g2_rd = self.g2_reduce(g2_mp)  # 2048->256
        # g2_rd = g2_rd.squeeze()
        # g2_sm = self.classifier_g2(g2_rd)


        # fea_trip = [g1_rd, g2_rd]
        # scores = [g1_sm, g2_sm]
        #
        # fea_test = [g1_rd, g2_rd]
        fea_trip = [g1_rd]
        scores = [g1_sm]

        fea_test = [g1_rd]
        fea_for_test = torch.cat(fea_test, 1)
        if self.norm:
            fea_for_test = F.normalize(fea_for_test)
        return scores, fea_trip,l2_side,l3_side,l4_side, fea_for_test

def copy_weight_for_branches(model_name, new_model, branch_num=2):
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
        if(ori_prefix == 'layer2') or (ori_prefix == 'layer3') or (ori_prefix =='layer4'):
            for i in range(0,branch_num):
                if ori_key in model_keys:
                    new_model_weight[block] = model_weight[ori_key]
                else:
                    continue
        elif ori_key in model_keys:
            new_model_weight[block] = model_weight[ori_key]
    save_name = model_name + '_stn2.tar'
    torch.save(new_model_weight,save_name)


def resnet50_stn2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_stn2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # pdb.set_trace()
        copy_weight_for_branches('resnet50',model)
        model.load_state_dict(torch.load('./resnet50_stn2.tar'))
    return model


def resnet101_stn2(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_stn2(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        copy_weight_for_branches('resnet101',model)
        model.load_state_dict(torch.load('./resnet101_stn2.tar'))
    return model
