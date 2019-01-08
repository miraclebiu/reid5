import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision
import pdb

if torch.__version__ == '0.3.0.post4':
    from torch.nn.init import constant as init_constant
    from torch.nn.init import kaiming_normal as init_kaiming_normal
    from torch.nn.init import normal as init_normal
else:
    from torch.nn.init import constant_ as init_constant
    from torch.nn.init import kaiming_normal_ as init_kaiming_normal
    from torch.nn.init import normal_ as init_normal


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, groups=groups)
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

def make_layer(block=Bottleneck, inplanes=1024, planes=512, blocks=3, stride=1, groups =3):
    downsample = None
    inplanes = inplanes * groups
    planes = planes * groups
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False, groups=groups),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, groups=groups))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, groups=groups))
    return nn.Sequential(*layers)

def load_conv( new_conv, old_conv, num_bypath):
    new_conv.weight.data.copy_(torch.cat([old_conv.weight.data]*num_bypath,0))
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(torch.cat([old_conv.bias.data]*num_bypath,0))
    


def load_bn(new_bn, old_bn, num_bypath):

    new_bn.bias.data.copy_(torch.cat([old_bn.bias.data]*num_bypath,0))
    new_bn.weight.data.copy_(torch.cat([old_bn.weight.data]*num_bypath,0))
    new_bn.running_mean.copy_(torch.cat([old_bn.running_mean]*num_bypath,0))
    new_bn.running_var.copy_(torch.cat([old_bn.running_var]*num_bypath,0))
  

def parallel_bypath(new_module, old_module, num_bypath):
    if isinstance(new_module, Bottleneck):
        for sub_new_module,sub_old_module in zip(new_module.children(), old_module.children()):
            parallel_bypath(sub_new_module, sub_old_module, num_bypath)

    elif isinstance(new_module, nn.Sequential):
        for sub_new_module,sub_old_module in zip(new_module.children(), old_module.children()):
            parallel_bypath(sub_new_module, sub_old_module, num_bypath)

    elif isinstance(new_module, nn.Conv2d):
        load_conv(new_module,old_module,num_bypath)
        print(new_module)
    elif isinstance(new_module, nn.BatchNorm2d):
        load_bn(new_module,old_module,num_bypath)
        print(new_module)
    elif isinstance(new_module, nn.ReLU):
        return
    else:
        raise RuntimeError('not conv or bn or bottleneck or sequential')


# def parallel_bypath(module, num_bypath, enlarge_rf=True):
#     # 
#     if isinstance(module, Bottleneck):
#         for sub_module in module.children():
#             parallel_bypath(sub_module, num_bypath)
#     elif isinstance(module, nn.Sequential):
#         for sub_module in module.children():
#             parallel_bypath(sub_module, num_bypath)
#     elif isinstance(module, nn.Conv2d):
#         pdb.set_trace()

#         module.in_channels = module.in_channels * num_bypath
#         module.out_channels = module.out_channels * num_bypath
#         module.groups = num_bypath
#         new_weight = [module.weight] * num_bypath
#         module.weight = Parameter(torch.cat(new_weight, 0))
#         if enlarge_rf and module.stride == (2, 2):
#             module.stride = (1, 1)
#         if module.bias is not None :
#             new_bias = [module.bias] * num_bypath
#             module.bias = Parameter((torch.cat(new_bias, 0)))
#         print(module)
#     elif isinstance(module, nn.BatchNorm2d):
#         print(module)
#         module.num_features = module.num_features * num_bypath
#         new_weight = [module.weight] * num_bypath
#         module.weight = Parameter((torch.cat(new_weight, 0)))

#         new_running_mean = [module.running_mean] * num_bypath
#         module.running_mean = torch.cat(new_running_mean, 0)

#         new_running_var = [module.running_var] * num_bypath
#         module.running_var = torch.cat(new_running_var, 0)
#         if module.bias is not None :
#             new_bias = [module.bias] * num_bypath
#             module.bias = Parameter((torch.cat(new_bias, 0)))
#         print(module)
#     elif isinstance(module, nn.ReLU):
#         return
#     else:

#         raise RuntimeError('not conv or bn or bottleneck or sequential')


def _initialize_weights(models):
    for m in models:
        if isinstance(m, list):
            for mini_m in m:
                _initialize_weights(m)
        else:
            if isinstance(m, nn.Sequential):
                _initialize_weights(m.modules())
            elif isinstance(m, nn.Module):
                _initialize_weights(m.modules())
            elif isinstance(m, nn.Conv2d):
                init_kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init_constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init_constant(m.weight, 1)
                init_constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_normal(m.weight, std=0.001)
                if m.bias is not None:
                    init_constant(m.bias, 0)



class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(4,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)/2,(kernel_size[1]-1)/2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).view(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)


        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,5)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).view(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]


        fea = torch.cat(fea_stack, 3)
        return fea

class SelfAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention * scale
        # weighting
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class AttentionMap(nn.Module):
    """Get current layer attention map for one head ."""

    def __init__(self, dropout=0.0):
        super(AttentionMap, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k):
        scale = (k.size(-1)) ** -0.5
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention * scale
        attention = self.softmax(attention)
        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=2048, num_heads=8, dropout=0.0, has_norm=True, compression=1):
        super(MultiHeadAttention, self).__init__()
        self.middle_dim = int(model_dim// compression)

        self.dim_per_head = int(self.middle_dim // num_heads)
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.middle_dim)
        self.linear_v = nn.Linear(model_dim, self.middle_dim)
        self.linear_q = nn.Linear(model_dim, self.middle_dim)

        self.dot_product_attention = SelfAttention(dropout)
        self.has_norm = has_norm

        self.linear_final = nn.Linear(self.middle_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.layer_norm = nn.BatchNorm1d(model_dim) # 128 is 16*8 spatial size, batchnorm is normalize the second channel in models

    def forward(self, query, key, value):
        # residual connect
        residual = value

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        ##### reshape to every head batch * len * dim_per_head
        key = key.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)
        value = value.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)
        query = query.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.transpose(2,1).contiguous().view(batch_size, num_heads*dim_per_head, -1).transpose(2,1)

        # final linear projection
        output = self.linear_final(context)

        output = self.dropout(output)
        output = self.layer_norm(output)
        output = self.relu(output)

        output = residual + output

        return output, attention


class MultiHeadAttention_ca(nn.Module):
    def __init__(self, spatial_size = 16*8,model_dim=512, middle_ss = 32*16 , norm_type='bn'):
        super(MultiHeadAttention_ca, self).__init__()

        self.linear_k = nn.Linear(spatial_size, middle_ss)
        self.linear_v = nn.Linear(spatial_size, middle_ss)
        self.linear_q = nn.Linear(spatial_size, middle_ss)

        self.dot_product_attention = SelfAttention(0.0)
        self.linear_final = nn.Linear(middle_ss, spatial_size)
        if norm_type != 'bn':
            self.layer_norm = nn.LayerNorm(model_dim)
        else:
            self.layer_norm = nn.BatchNorm1d(model_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query, key, value):
        # residual connect
        residual = value

        # scaled dot product attention
        query = self.linear_q(query)
        key   = self.linear_k(key)
        value = self.linear_v(value)

        scale = key.size(-1) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale)

        # final linear projection
        output = self.linear_final(context)
        output = self.layer_norm(output)
        output = self.relu(output)

        output = residual + output

        return output, attention