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


rank_loss = 0.0
correct = 0
sort_dists, sort_indice = dist.sort() 
for i in range(n):
    full_index = [i for i in range(num_instances)]
    cur_mask = mask[i]
    intra_class = cur_mask.nonzero()
    assert len(intra_class) == num_instances
    sort_dist, sort_indice = sort_dists[i], sort_indice[i]
    # rank_indice indicate the real index in the sorted dist
    wrong_indice = list()
    for each_ind in intra_class:
        rank_index = (sort_indice == each_ind).nonzero().item()
        if rank_index >= num_instances:
            wrong_indice.append(rank_index)
        else:
            full_index.remove(rank_index)
    left_wrong_num = len(full_index)
    if left_wrong_num == 0:
        correct += 1
        continue
    else:

        ## most difficult pos sample pair with most difficult neg sample
        wrong_indice = sorted(wrong_indice, reverse=True)
        most_wrong_neg = sort_dist[full_index[0]]
        for wpos_ind, wneg_ind in zip(wrong_indice, full_index):
            # multiply the num_wrong is meant to penalize the length of wrong indices

            wpos_dis = sort_dist[wpos_ind]
            wneg_dis = sort_dist[wneg_ind]
            wneg_dis_ratio = (most_wrong_neg - wneg_dis) / (most_wrong_neg + 1e-12)
            # consider the wrong num and the distance
            weight = left_wrong_num * torch.exp(wneg_dis_ratio)

            rank_loss += weight * (wpos_dis - wneg_dis) + margin
            left_wrong_num -= 1


rank_loss = rank_loss / n
prec = correct / n