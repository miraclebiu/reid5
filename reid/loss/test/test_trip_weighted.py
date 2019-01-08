from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb

import numpy as np

def remove_self(pos_dist,numth,num_instances):
    if numth == 0:
        return pos_dist[1:]
    if numth == num_instances-1:
        return pos_dist[:numth]
    else:
        return torch.cat((pos_dist[:numth],pos_dist[numth+1:]),0)

n = 16
targets = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
targets = torch.from_numpy(targets)
input_fea = torch.rand(16,512)
input_fea = Variable(input_fea)
# pdb.set_trace()
dist = torch.pow(input_fea, 2).sum(1).expand(n, n)
dist = dist + dist.t()
dist.addmm_(1, -2, input_fea, input_fea.t())
dist = dist.clamp(min=1e-12).sqrt()
mask = targets.expand(n, n).eq(targets.expand(n, n).t())
normalizer = nn.Softmax()
num_instance=4
for i in range(16):
    print(i)
    positives = dist[i][mask[i]]
    positives_delself = remove_self(positives,i%num_instance,num_instance)
    positives_weighted = normalizer(positives_delself)
    positive = positives_delself * positives_weighted
    print(positives,positives_delself,positives_weighted,positive)
    pdb.set_trace()
