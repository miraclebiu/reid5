from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from ..evaluation import accuracy

class MGN_loss(nn.Module):
    def __init__(self, margin1=1.2, num_instances=4, alpha=1.0, gamma =1.0,theta=0.1, has_trip = False, has_side = False):
        super(MGN_loss, self).__init__()
        self.margin1 = margin1
        self.num_instances = num_instances
        self.alpha = alpha 
        # self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.has_trip = has_trip
        self.has_side = has_side
        print(self.alpha,self.gamma,self.theta)
        # self.xentropy_loss = nn.CrossEntropyLoss()
    
    
    def forward(self, inputs, targets):
        softmax_out = inputs[0]
        trip_out = inputs[1]
        if len(inputs) > 3 and self.has_side :
            l2_side = inputs[2]
            l3_side = inputs[3]
            l4_side = inputs[4]
            loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
            loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())

        sf_num = len(softmax_out)
        total_cls_loss = 0
        for i in range(0,sf_num):
            # total_cls_loss += self.xentropy_loss(softmax_out[i],targets)
            total_cls_loss += F.cross_entropy(softmax_out[i], targets)

        trip_num = len(trip_out)
        total_trip_loss = 0
        if self.has_trip:
            for i in range(0,trip_num):
                input_fea = trip_out[i]
                n = input_fea.size(0)
                num_person= n // self.num_instances
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
                temp_trip_loss = F.margin_ranking_loss(dist_an,dist_ap,y,self.margin1)
                total_trip_loss += temp_trip_loss
        if self.has_side:
            # pdb.set_trace()
            loss = self.gamma*total_cls_loss + self.alpha*total_trip_loss + self.theta * (loss42 + loss43)
        else:
            loss = self.gamma*total_cls_loss + self.alpha*total_trip_loss
        accuracy_val,  = accuracy(softmax_out[0].data, targets.data)
        prec = accuracy_val[0]
        return loss, prec 
