from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation import accuracy

def sac(l4_side,l3_side,l2_side):
    loss42 = torch.sqrt((l4_side - l2_side).pow(2).sum())
    loss43 = torch.sqrt((l4_side - l3_side).pow(2).sum())
    return loss42+loss43


class Quadruplet_SAC(nn.Module):
    def __init__(self, margin1=0, margin2=0, num_instances=4, alpha=1.0, beta=0.5 ,gamma =0.5,theta=0.1):
        super(Quadruplet_SAC, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2)
        self.num_instances = num_instances
        self.alpha = alpha 
        self.beta = beta
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
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
       
        dist_ap, dist_an ,dist_to_get= [], [],[]
        for i in range(n):
            hard_positive = dist[i][mask[i]].max()
            dist_ap.append(hard_positive)

            hard_negative = dist[i][mask[i] == 0].min(0)
            dist_an.append(hard_negative[0])

            negative_negative =hard_negative[1]
            lower_bound = (i//self.num_instances)*self.num_instances
            if (negative_negative >=lower_bound).cpu().data.numpy():
                negative_negative = negative_negative + self.num_instances

            dist_to_get.append(negative_negative)
        dist_ann =torch.cat([dist_an[i.cpu().data.numpy()[0]] for i in dist_to_get])
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        sac_loss = sac(l4_side,l3_side,l2_side)
        # loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
        # loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())
        # loss32 = torch.sqrt((l3_side-l2_side).pow(2).sum())

        TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
        TripletLoss2 = self.ranking_loss2(dist_ann, dist_ap, y)
        # xentropy = self.xentropy_loss(input_fea, targets)
        xentropy = self.xentropy_loss(cls_fea,targets)

        loss = self.alpha*TripletLoss1 + self.beta*TripletLoss2 + self.gamma *xentropy + self.theta*sac_loss
        #pdb.set_trace()
        #print('Triplet-Loss is :{},  Cross-Entropy-Loss is :{}'.format(TripletLoss,xentropy))
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        #pdb.set_trace()
        accuracy_val,  = accuracy(input_fea.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec2 = max(prec,accuracy_val)

        return loss, prec2 
