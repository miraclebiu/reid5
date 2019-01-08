from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation import accuracy

import numpy as np

class Rank_loss(nn.Module):
    def __init__(self, margin1=0.5,margin2=0.3, num_instances=8, alpha=1.0, gamma =1.0,theta=0.1,divide=3):
        super(Rank_loss, self).__init__()
        self.margin1 = margin1
        self.num_instances = num_instances
        self.alpha = alpha 
        self.gamma = gamma
        self.theta = theta
        self.xentropy_loss = nn.CrossEntropyLoss()
    
    
    def forward(self, inputs, targets):
        cls_fea = inputs[0]
        input_fea=inputs[-1] # before_cls_fea
        l2_side = inputs[1]
        l3_side = inputs[2]
        l4_side = inputs[3]

        n = input_fea.size(0)
        num_person=n // self.num_instances
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(input_fea, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, input_fea, input_fea.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # print(dist.mean().data.cpu()[0])
        # For each anchor, find the hardest positive and negative dist[0]
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        rank_loss=0.0
        correct = 0
        for i in range(n):
            full_index = [i for i in range(self.num_instances)]
            cur_dist = dist[i]
            cur_mask = mask[i]
            intra_class = cur_mask.nonzero()
            assert len(intra_class) == self.num_instances
            sort_dist, sort_indice = cur_dist.sort()
            # rank_indice indicate the real index in the sorted dist
            wrong_indice = list()
            for each_ind in intra_class:
                rank_index = (sort_indice == each_ind).nonzero().item()
                if rank_index >=self.num_instances:
                    wrong_indice.append(rank_index)
                else:
                    full_index.remove(rank_index)
            left_wrong_num = len(full_index)
            if left_wrong_num == 0:
                correct +=1
                continue
            else:

                ## most difficult pos sample pair with most difficult neg sample
                wrong_indice = sorted(wrong_indice,reverse=True)
                most_wrong_neg = sort_dist[full_index[0]]
                for wpos_ind, wneg_ind in zip(wrong_indice,full_index):
                    # multiply the num_wrong is meant to penalize the length of wrong indices

                    wpos_dis = sort_dist[wpos_ind]
                    wneg_dis = sort_dist[wneg_ind]
                    wneg_dis_ratio = (most_wrong_neg-wneg_dis) / (most_wrong_neg+1e-12)
                    # consider the wrong num and the distance
                    weight = left_wrong_num * torch.exp(wneg_dis_ratio)

                    rank_loss += weight*(wpos_dis - wneg_dis) + self.margin1
                    left_wrong_num -= 1
                    # print(wpos_ind,wneg_ind)

            # rank_indice indicate the real index in the sorted dist 
            # for example for index7: 4,5,6,7 are same class
            # the distance sort for 4 is 1 dist 0.1
            # the distance sort for 5 is 3 dist 0.3
            # the distance sort for 6 is 2 dist 0.2
            # the distance sort for 7 is 0 dist 0.0
            # which means that 7 is the most likely to 7, then 4, 6, 5
        rank_loss = rank_loss/n


        loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
        loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())
        side_loss = (loss42+loss43)
        xentropy = self.xentropy_loss(cls_fea,targets)

        total_loss = self.alpha * rank_loss + self.gamma * xentropy + self.theta * side_loss
        # pdb.set_trace()

        prec = correct / n
        accuracy_val,  = accuracy(input_fea.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec2 = max(prec,accuracy_val)

        return total_loss, prec2 







