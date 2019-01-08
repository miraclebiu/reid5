from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation import accuracy

import numpy as np

def remove_self(pos_dist,numth,num_instances):
    if numth == 0:
        return pos_dist[1:]
    if numth == num_instances-1:
        return pos_dist[:numth]
    else:
        return torch.cat((pos_dist[:numth],pos_dist[numth+1:]),0)

class Triplet_Weighted_SAC(nn.Module):
    def __init__(self, margin1=0.5,margin2=0.3, num_instances=4, alpha=1.0, gamma =0.5,theta=0.1,divide=3):
        super(Triplet_Weighted_SAC, self).__init__()
        self.margin1 = margin1
        self.margin2  = margin2
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2)
        self.num_instances = num_instances
        self.alpha = alpha 
        self.gamma = gamma
        self.theta = theta
        self.normalizer = nn.Softmax()
        self.xentropy_loss = nn.CrossEntropyLoss()
        self.divide = divide
    
    
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
       
        dist_ap, dist_an = [], []
        dist_ap2, dist_an2 = [], []
        dist_mean = []
        for i in range(n):
            positives = dist[i][mask[i]]
            positive = positives.max()
            # form hard set  remove easy samples
            positives_delself = remove_self(positives,i%self.num_instances,self.num_instances)
            pos_mask = positives_delself > (positives_delself-self.margin2/self.divide)
            pos_set = torch.masked_select(positives_delself, pos_mask)
            mean_pos = pos_set.mean()
            positives_weighted = self.normalizer(pos_set)
            positive2 = pos_set * positives_weighted
            positive2 = positive2.sum()


            negatives = dist[i][mask[i] == 0]
            negative = negatives.min()
            # form hard set  remove easy samples
            neg_mask = negatives < (negative + self.margin2/self.divide)
            neg_set = torch.masked_select(negatives, neg_mask)
            mean_neg = neg_set.mean()
            negatives_weighted = self.normalizer(-neg_set) # minus mark is very important
            negative2 = neg_set * negatives_weighted
            negative2 = negative2.sum()


            dist_ap.append(positive)
            dist_an.append(negative)
            dist_ap2.append(positive2)
            dist_an2.append(negative2)
            dist_mean.append(mean_neg-mean_pos)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        dist_mean = torch.cat(dist_mean)
        dist_mean = dist_mean.mean()
        print("pos_neg_hardSet_mean_dist: {:.2f}".format(dist_mean.data.cpu()[0]))
        # Compute ranking hinge loss
        
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
        loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())
        # loss32 = torch.sqrt((l3_side-l2_side).pow(2).sum())

        # TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
        TripletLoss2 = self.ranking_loss2(dist_an2, dist_ap2, y)
        xentropy = self.xentropy_loss(cls_fea,targets)

        # loss = self.alpha*(TripletLoss1+TripletLoss2) + self.gamma *xentropy + self.theta*(loss42+loss43)
        loss = self.alpha * (TripletLoss2) + self.gamma * xentropy + self.theta * (loss42 + loss43)
        #pdb.set_trace()
        #print('Triplet-Loss is :{},  Cross-Entropy-Loss is :{}'.format(TripletLoss,xentropy))
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        accuracy_val,  = accuracy(input_fea.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec2 = max(prec,accuracy_val)

        return loss, prec2 







