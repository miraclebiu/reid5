from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation import accuracy

class Triplet_SAC(nn.Module):
    def __init__(self, margin1=0, num_instances=4, alpha=1.0, gamma =0.5, theta=0.1, has_sac = True):
        super(Triplet_SAC, self).__init__()
        self.margin1 = margin1
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1)
        self.num_instances = num_instances
        self.alpha = alpha 
        self.gamma = gamma
        self.theta = theta
        self.has_sac = has_sac
        self.xentropy_loss = nn.CrossEntropyLoss()
    
    
    def forward(self, inputs, targets):
        cls_fea = inputs[0]
        input_fea=inputs[-1] # before_cls_fea


        n = input_fea.size(0)
        num_person=n // self.num_instances
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(input_fea, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, input_fea, input_fea.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
       
        dist_ap, dist_an = [], []
        for i in range(n):
            hard_positive = dist[i][mask[i]].max()
            dist_ap.append(hard_positive)

            hard_negative = dist[i][mask[i] == 0].min(0)
            dist_an.append(hard_negative[0])
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
        loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())


        TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
        xentropy = self.xentropy_loss(cls_fea,targets)
        if self.has_sac:
            l2_side = inputs[1]
            l3_side = inputs[2]
            l4_side = inputs[3]
            loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
            loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())
            loss = self.alpha*TripletLoss1 + self.gamma *xentropy + self.theta*(loss42+loss43)
        else:
            loss = self.alpha*TripletLoss1 + self.gamma *xentropy      

        

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        accuracy_val,  = accuracy(input_fea.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec2 = max(prec,accuracy_val)

        return loss, prec2 
