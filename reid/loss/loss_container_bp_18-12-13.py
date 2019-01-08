from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import pdb
from ..evaluation import accuracy

from pprint import pprint

from easydict import EasyDict as edict

loss_config = edict()
# current_model
loss_config.model = 'resnet_reid'
# single or list to determine whether using mgn_loss or not
loss_config.loss_type = 'single'
# whether has self-attention constraint or not 
loss_config.sac = True
# whether has cross-entropy loss or not 
loss_config.xen = True
# whether has triplet loss or not
loss_config.trip = True
# whether has triplet-weighted loss or not
loss_config.trip_weight = False
#triplet weight combineori triplet list and weight
loss_config.trip_com = False

loss_config.rank_loss = False

# whether has quadrupletl oss or not
loss_config.quad = False
# balance each part of the loss,default params
# alpha for triplet-hard loss 
loss_config.alpha = 1.0
# beta for second part of quadruplet loss
loss_config.beta = 0.5
# gamma for cross-entropy
loss_config.gamma = 0.5
# theta for self-attention constraint loss
loss_config.theta = 0.5
# margin for triplet loss
loss_config.margin1 = 0.5
# margin for quadruplet loss
loss_config.margin2 = 0.3
# margin for triplet-weight loss
loss_config.weight_margin = 2.0
# shrink the radius of the positive set 
loss_config.trip_weight_pos_radius_divide = 2.0
# shrink the radius of the negative set
loss_config.trip_weight_neg_radius_divide = 2.0
# num_instances for each person in a batch
loss_config.num_instances = 8


def merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """

  for k, v in b.items():
    # a must specify keys that are in b
    if k not in a:
        continue

    # the types must match, too
    old_type = type(v)
    new_obj = getattr(a,k)
    if old_type is not type(new_obj):
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(old_type,
                                                       type(new_obj), k))

    # recursively merge dicts
    b[k] = new_obj
  return b



def sac3(l2_side,l3_side,l4_side):
    loss42 = torch.sqrt((l4_side - l2_side).pow(2).sum())
    loss43 = torch.sqrt((l4_side - l3_side).pow(2).sum())
    return loss42+loss43

def sac2(l2_side,l3_side,l4_side):
    loss43 = torch.sqrt((l4_side - l3_side).pow(2).sum())
    return loss43

def remove_self(pos_dist,numth,num_instances):
    if numth == 0:
        return pos_dist[1:]
    if numth == num_instances-1:
        return pos_dist[:numth]
    else:
        return torch.cat((pos_dist[:numth],pos_dist[numth+1:]),0)


def select_triplet(feature,targets, num_instances):
    n = feature.size(0)
    num_person=n // num_instances
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    dist_ap_hard, dist_an_hard = [], []

    for i in range(n):
        positives = dist[i][mask[i]]
        # positives_delself = remove_self(positives, i%num_instances, num_instances)
        positive_hard = positives.max()

      
        negatives = dist[i][mask[i] == 0]
        negative_hard = negatives.min()

        dist_ap_hard.append(positive_hard)
        dist_an_hard.append(negative_hard)


    dist_ap_hard = torch.stack(dist_ap_hard)
    dist_an_hard = torch.stack(dist_an_hard)

    y = dist_an_hard.data.new()
    y.resize_as_(dist_an_hard.data)
    y.fill_(1)
    y = Variable(y)

    return dist_ap_hard, dist_an_hard, y


def select_triplet_exp(feature, targets, num_instances):
    n = feature.size(0)
    fea_dim = feature.size(1)
    scale = fea_dim ** -0.5

    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    dist_ap_hard, dist_an_hard = [], []
    dist_mean = []
    # aaa = feature[0,:]
    # bbb = aaa.pow(2).sum()
    for i in range(n):
        positives = dist[i][mask[i]]
        # positives_delself = remove_self(positives, i%num_instances, num_instances)
        positive_hard = positives.max()
        positive_hard_scale = positive_hard * scale

        negatives = dist[i][mask[i] == 0]
        negative_hard = negatives.min()
        negative_hard_scale = negative_hard * scale

        np_dist = positive_hard_scale - negative_hard_scale
        exp_pos_hard = np_dist.exp()
        positive_hard_exp = exp_pos_hard * positive_hard_scale
        negative_hard_exp = exp_pos_hard * negative_hard_scale
        # dist_mean.append(positive_hard_exp-negative_hard_exp)

        #
        #
        # print(np_dist)
        # print(exp_pos_hard)
        # print(positive_hard_scale)
        # print(negative_hard_scale)
        # pdb.set_trace()

        dist_ap_hard.append(positive_hard_exp)
        dist_an_hard.append(negative_hard_exp)

    dist_ap_hard = torch.stack(dist_ap_hard)
    dist_an_hard = torch.stack(dist_an_hard)
    # dist_mean = torch.stack(dist_mean)
    # print('np_dist_mean',dist_mean.mean().item())

    y = dist_an_hard.data.new()
    y.resize_as_(dist_an_hard.data)
    y.fill_(1)
    y = Variable(y)

    return dist_ap_hard, dist_an_hard, y


# def select_triplet_weighted(feature, targets, num_instances,
#                             normalizer,
#                             pos_margin, pos_radius_divide,
#                             neg_margin, neg_radius_divide, trip_com=False):
#     n = feature.size(0)
#     num_person = n // num_instances
#     # Compute pairwise distance, replace by the official when merged
#     dist = torch.pow(feature, 2).sum(1).expand(n, n)
#     dist = dist + dist.t()
#     dist.addmm_(1, -2, feature, feature.t())
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     # print(dist.mean().data.cpu()[0])
#     # For each anchor, find the hardest positive and negative dist[0]
#     mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#
#     dist_ap_hard, dist_an_hard = [], []
#     dist_ap_weig, dist_an_weig = [], []
#     # dist_mean = []
#     for i in range(n):
#         positives = dist[i][mask[i]]
#         positives_delself = remove_self(positives, i % num_instances, num_instances)
#         positive_hard = positives_delself.max()
#         # form hard set  remove easy samples
#
#         negatives = dist[i][mask[i] == 0]
#         negative_hard = negatives.min()
#
#         dist_ap_hard.append(positive_hard)
#         dist_an_hard.append(negative_hard)
#
#         # form hard set  remove easy samples
#         pos_mask = positives_delself > (positive_hard - pos_margin / pos_radius_divide)
#         pos_set = torch.masked_select(positives_delself, pos_mask)
#         # this normalize need to be modify
#         positives_weight = normalizer(pos_set - positive_hard)  # this operation is for gradient
#         positives_weighted = pos_set * positives_weight
#         positive_weighted = positives_weighted.sum()
#
#         # form hard set  remove easy samples
#         neg_mask = negatives < (negative_hard + neg_margin / neg_radius_divide)
#         neg_set = torch.masked_select(negatives, neg_mask)
#
#         negatives_weight = normalizer(-neg_set + negative_hard)  # -(neg_set-negative_hard)
#         negatives_weighted = neg_set * negatives_weight
#         negative_weighted = negatives_weighted.sum()
#
#         dist_ap_weig.append(positive_weighted)
#         dist_an_weig.append(negative_weighted)
#         # dist_mean.append(mean_neg-mean_pos)
#
#     dist_ap_hard = torch.stack(dist_ap_hard)
#     dist_an_hard = torch.stack(dist_an_hard)
#     dist_ap_weig = torch.stack(dist_ap_weig)
#     dist_an_weig = torch.stack(dist_an_weig)
#     # dist_mean = torch.cat(dist_mean)
#     # dist_mean = dist_mean.mean()
#     # print("pos_neg_hardSet_mean_dist: {:.2f}".format(dist_mean.data.cpu()[0]))
#     y = dist_an_hard.data.new()
#     y.resize_as_(dist_an_hard.data)
#     y.fill_(1)
#     y = Variable(y)
#
#     if trip_com:
#         return [dist_ap_hard, dist_ap_weig], [dist_an_hard, dist_an_weig], y
#     else:
#         return dist_ap_weig, dist_an_weig, y


def select_triplet_weighted(feature, targets, num_instances,
                            normalizer,
                            pos_margin, pos_radius_divide,
                            neg_margin, neg_radius_divide, trip_com=False):
    n = feature.size(0)
    num_person = n // num_instances
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # print(dist.mean().data.cpu()[0])
    # For each anchor, find the hardest positive and negative dist[0]
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    dist_ap_hard, dist_an_hard = [], []
    dist_ap_weig, dist_an_weig = [], []
    dist_mean = []
    for i in range(n):
        positives = dist[i][mask[i]]
        positives_delself = remove_self(positives, i % num_instances, num_instances)
        positive_hard = positives_delself.max()
        # form hard set  remove easy samples

        negatives = dist[i][mask[i] == 0]
        negative_hard = negatives.min()

        dist_ap_hard.append(positive_hard)
        dist_an_hard.append(negative_hard)

        # form hard set  remove easy samples
        # pos_mask = positives_delself > (positive_hard - pos_margin / pos_radius_divide)
        # pos_set = torch.masked_select(positives_delself, pos_mask)
        pos_set = positives_delself
        # this normalize need to be modify
        positives_weight = normalizer(pos_set - positive_hard)  # this operation is for gradient
        positives_weighted = pos_set * positives_weight
        positive_weighted = positives_weighted.sum()

        # form hard set  remove easy samples
        neg_mask = negatives < (negative_hard + neg_margin / neg_radius_divide)
        neg_set = torch.masked_select(negatives, neg_mask)

        negatives_weight = normalizer(-neg_set + negative_hard)  # -(neg_set-negative_hard)
        negatives_weighted = neg_set * negatives_weight
        negative_weighted = negatives_weighted.sum()

        dist_ap_weig.append(positive_weighted)
        dist_an_weig.append(negative_weighted)
        dist_mean.append(positive_weighted-negative_weighted)

    dist_ap_hard = torch.stack(dist_ap_hard)
    dist_an_hard = torch.stack(dist_an_hard)
    dist_ap_weig = torch.stack(dist_ap_weig)
    dist_an_weig = torch.stack(dist_an_weig)
    dist_mean = torch.stack(dist_mean)
    dist_mean = dist_mean.mean()
    print("pos_neg_hardSet_mean_dist: {:.2f}".format(dist_mean))
    y = dist_an_hard.data.new()
    y.resize_as_(dist_an_hard.data)
    y.fill_(1)
    y = Variable(y)

    if trip_com:
        return [dist_ap_hard, dist_ap_weig], [dist_an_hard, dist_an_weig], y
    else:
        return dist_ap_weig, dist_an_weig, y



def select_quadruplet(feature, targets, num_instances):

    n = feature.size(0)
    num_person=n // num_instances
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
   
    dist_ap, dist_an ,dist_ann= [], [],[]
    for i in range(n):
        hard_positive = dist[i][mask[i]].max()
        dist_ap.append(hard_positive)

        hard_negative = dist[i][mask[i] == 0].min(0)
        dist_an.append(hard_negative[0])

        negative_index =hard_negative[1].item()

        mask2 = mask[i].clone()
        lower_bound = (negative_index//num_instances)*num_instances
        upper_bound = (negative_index//num_instances+1)*num_instances
        mask2[lower_bound:upper_bound] = 1

        hard_negative2 = dist[negative_index][mask2 == 0].min()
        dist_ann.append(hard_negative2)
        # print(mask[i])
        # print(mask2)
        # pdb.set_trace()

    dist_ann =torch.stack(dist_ann)
    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    # Compute ranking hinge loss
    
    y = dist_an.data.new()
    y.resize_as_(dist_an.data)
    y.fill_(1)
    y = Variable(y)
    return dist_ap, dist_an, dist_ann, y


def select_quadruplet_weight(feature, targets, num_instances):
    n = feature.size(0)
    num_person = n // num_instances
    fea_dim = feature.size(1)
    scale = fea_dim ** -0.5
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    dist_ap, dist_an, dist_to_get = [], [], []
    for i in range(n):
        hard_positive = dist[i][mask[i]].max()
        dist_ap.append(hard_positive)

        hard_negative = dist[i][mask[i] == 0].min(0)
        dist_an.append(hard_negative[0])

        negative_negative = hard_negative[1]
        lower_bound = (i // num_instances) * num_instances
        if (negative_negative >= lower_bound).cpu().data.numpy():
            negative_negative = negative_negative + num_instances

        dist_to_get.append(negative_negative)
    dist_ann = torch.stack([dist_an[i.cpu().data.numpy()[0]] for i in dist_to_get])
    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    # Compute ranking hinge loss

    y = dist_an.data.new()
    y.resize_as_(dist_an.data)
    y.fill_(1)
    y = Variable(y)
    return dist_ap, dist_an, dist_ann, y



def cal_rank_loss_normalize(feature, targets, num_instances, margin):
    n = feature.size(0)
    fea_dim = feature.size(1)
    scale = fea_dim ** -0.5

    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    ######### if dist is norm then should normalize by divide sqrt(dim)
    dist = dist * scale
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    rank_loss = 0.0
    correct = 0
    # np_dist_mean = 0
    for i in range(n):
        full_index = [i for i in range(num_instances)]
        cur_dist = dist[i]
        cur_mask = mask[i]
        intra_class = cur_mask.nonzero()
        assert len(intra_class) == num_instances
        sort_dist, sort_indice = cur_dist.sort()
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

            ## make the network can adjust easily
            wrong_indice = sorted(wrong_indice, reverse=True)
            most_wrong_neg = sort_dist[full_index[0]]
            # full_index means wrong neg index in the sort list in num_instance's
            # wrong_indice means wrong neg index in the sort list after num_instance's
            for wpos_ind, wneg_ind in zip(wrong_indice, full_index):
                wpos_dis = sort_dist[wpos_ind]
                wneg_dis = sort_dist[wneg_ind]
                pn_real_dis = dist[wpos_ind,wneg_ind]
                # wneg_dis_ratio = (most_wrong_neg - wneg_dis) / (most_wrong_neg + 1e-12)
                # consider the wrong num and the distance
                # weight = torch.exp(wneg_dis_ratio) * torch.exp(wpos_dis - wneg_dis)
                weight = torch.exp(pn_real_dis)
                clamp_dist = torch.clamp(wpos_dis - wneg_dis + margin, min=0.0)

                rank_loss += weight * clamp_dist
                left_wrong_num -= 1
                # np_dist_mean+=(wpos_dis - wneg_dis).item()
    # np_dist_mean = np_dist_mean/n
    # print(np_dist_mean)
    rank_loss = rank_loss / n
    prec = correct / n
    return rank_loss, prec



def cal_rank_loss(feature, targets, num_instances, margin):
    n = feature.size(0)
    # fea_dim = feature.size(1)
    # scale = fea_dim ** -0.5

    dist = torch.pow(feature, 2).sum(1).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, feature, feature.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    ######### if dist is norm then should normalize by divide sqrt(dim)
    # dist = dist * scale
    # print(dist.mean().data.cpu()[0])
    # For each anchor, find the hardest positive and negative dist[0]
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    rank_loss = 0.0
    correct = 0
    for i in range(n):
        full_index = [i for i in range(num_instances)]
        cur_dist = dist[i]
        cur_mask = mask[i]
        intra_class = cur_mask.nonzero()
        assert len(intra_class) == num_instances
        sort_dist, sort_indice = cur_dist.sort()
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
                wpos_dis = sort_dist[wpos_ind]
                wneg_dis = sort_dist[wneg_ind]
                wneg_dis_ratio = (most_wrong_neg - wneg_dis) / (most_wrong_neg + 1e-12)
                # consider the wrong num and the distance
                weight = left_wrong_num* torch.exp(wneg_dis_ratio)

                rank_loss += weight * (wpos_dis - wneg_dis) + margin
                left_wrong_num -= 1
                # print(wpos_ind,wneg_ind)
    rank_loss = rank_loss / n
    prec = correct / n
    return rank_loss, prec


class Loss_Single(nn.Module):
    def __init__(self, config):
        super(Loss_Single, self).__init__()
        self.config = config

        self.xentropy_loss = nn.CrossEntropyLoss()
        self.ranking_loss1 = nn.MarginRankingLoss(margin=config.margin1)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=config.margin2)
        self.pos_divide = config.trip_weight_pos_radius_divide
        self.neg_divide = config.trip_weight_neg_radius_divide
        if config.trip_weight and not config.trip and not config.quad :
            self.ranking_loss2.margin = config.weight_margin


        self.num_instances = config.num_instances
        self.alpha = config.alpha 
        self.beta = config.beta
        self.gamma = config.gamma
        self.theta = config.theta


    
    def forward(self, inputs, targets):

        num_input = len(inputs)
        if num_input == 3 :
            softmax_out = inputs[0]
            trip_out = inputs[1]
            features = inputs[2]
        elif num_input == 6:
            softmax_out = inputs[0]
            trip_out = inputs[1]
            l2_side = inputs[2]
            l3_side = inputs[3]
            l4_side = inputs[4]
            features = inputs[5]

        sac_loss = 0
        xentropy = 0
        TripletLoss1 = 0
        TripletLoss2 = 0
        prec = 0


        if self.config.sac:
            sac_loss = sac3(l2_side,l3_side,l4_side)

        if self.config.xen:
            xentropy = self.xentropy_loss(softmax_out,targets)
        ############ triplet
        if self.config.trip and not self.config.trip_weight and not self.config.quad and not self.config.rank_loss:

            #### print('Using triplet loss')
            dist_ap, dist_an, y = select_triplet(trip_out,targets,
                                                 self.config.num_instances)
            TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
            prec = (dist_an.data > dist_ap.data).sum().item() * 1. / y.size(0)
        #### triplet weight
        if  self.config.trip_weight and not self.config.trip and not self.config.quad and not self.config.rank_loss:
            normalizer = F.softmax()
            dist_ap, dist_an, y = select_triplet_weighted(trip_out,targets,
                                                           self.num_instances,
                                                           normalizer,self.config.trip_com,
                                                           self.config.weight_margin, self.config.trip_weight_pos_radius_divide,
                                                           self.config.weight_margin, self.config.trip_weight_neg_radius_divide)
            if self.config.trip_com:
                #### print('Using triplet and triplet-weighted loss')
                TripletLoss1 = self.ranking_loss1(dist_an[0], dist_ap[0], y)
                TripletLoss2 = self.ranking_loss2(dist_an[1], dist_ap[1], y)
                prec = (dist_an[0].data > dist_ap[0].data).sum().item() * 1. / y.size(0)

            else:
                #### print('Using triplet-weighted loss')
                TripletLoss2 = self.ranking_loss2(dist_an, dist_ap, y)
                prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

                
        ####### quad
        if self.config.quad and not self.config.trip_weight and not self.config.trip and not self.config.rank_loss:
            #### print('Using quadruplet loss')
            dist_ap, dist_an, dist_ann, y = select_quadruplet(trip_out, targets, self.num_instances)
            TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
            TripletLoss2 = self.ranking_loss2(dist_ann, dist_ap, y) 
            prec = prec = (dist_an.data > dist_ap.data).sum().item() * 1. / y.size(0)
        ##### rank
        if self.config.rank_loss and not self.config.trip_weight and not self.config.trip and not self.config.quad:
            rank_loss, prec = cal_rank_loss(trip_out, targets, self.num_instances, self.config.margin1)
            TripletLoss1 = rank_loss

        loss = self.alpha*TripletLoss1 + self.beta*TripletLoss2 + self.gamma *xentropy + self.theta*sac_loss

        accuracy_val,  = accuracy(softmax_out.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec_batch = max(prec,accuracy_val)

        return loss, prec_batch 


class Loss_Multi(nn.Module):
    def __init__(self, config):
        super(Loss_Multi, self).__init__()
        self.config = config        

        self.num_instances = config.num_instances
        self.alpha = config.alpha 
        self.beta = config.beta
        self.gamma = config.gamma
        self.theta = config.theta
        self.margin1 = config.margin1
        self.margin2 = config.margin2
        self.pos_divide = config.trip_weight_pos_radius_divide
        self.neg_divide = config.trip_weight_neg_radius_divide
        if config.trip_weight and not config.trip and not config.quad :
            self.margin2 = config.weight_margin

        # self.xentropy_loss = nn.CrossEntropyLoss()
    
    
    def forward(self, inputs, targets):
        num_input = len(inputs)
        if num_input == 3 :
            softmax_out = inputs[0]
            trip_out = inputs[1]
            features = inputs[2]
        elif num_input == 6:
            softmax_out = inputs[0]
            trip_out = inputs[1]
            l2_side = inputs[2]
            l3_side = inputs[3]
            l4_side = inputs[4]
            features = inputs[5]

        
        sac_loss = 0
        xentropy = 0
        TripletLoss1 = 0
        TripletLoss2 = 0

        prec = 0
        prec_sf = 0

        if self.config.sac:
            # if self.config.model == 'resnet_channel3':
            #     sac_loss = sac2(l2_side, l3_side, l4_side)
            # else:
            #     sac_loss = sac3(l2_side,l3_side,l4_side)
            sac_loss = sac3(l2_side, l3_side, l4_side)

        num_softmax = len(softmax_out)
        # print(num_softmax)
        for i in range(0,num_softmax):
            xentropy += F.cross_entropy(softmax_out[i], targets)
        prec_sf, = accuracy(softmax_out[0].data, targets.data)
        prec_sf = prec_sf[0]

        num_trip = len(trip_out)
        for i in range(0,num_trip):
            if self.config.trip and not self.config.trip_weight and not self.config.quad and not self.config.rank_loss:
                # dist_ap, dist_an, y = select_triplet_exp(trip_out[i],targets,
                #                                      self.num_instances)
                dist_ap, dist_an, y = select_triplet(trip_out[i],targets,
                                                     self.num_instances)
                TripletLoss1 += F.margin_ranking_loss(dist_an, dist_ap, y, self.margin1)
                prec = (dist_an.data > dist_ap.data).sum().item() * 1. / y.size(0)

            if  self.config.trip_weight and not self.config.trip and not self.config.quad and not self.config.rank_loss:
                normalizer = nn.Softmax()
                dist_ap, dist_an, y = select_triplet_weighted(trip_out[i],targets,
                                                               self.num_instances,
                                                               normalizer,self.config.trip_com,
                                                               self.config.weight_margin, self.config.trip_weight_pos_radius_divide,
                                                               self.config.weight_margin, self.config.trip_weight_neg_radius_divide)
                pdb.set_trace()

                if self.config.trip_com:
                    #### print('Using triplet and triplet-weighted loss')
                    TripletLoss1 += F.margin_ranking_loss(dist_an[0], dist_ap[0], y, self.margin1)
                    TripletLoss2 += F.margin_ranking_loss(dist_an[1], dist_ap[1], y, self.margin2)
                    prec = (dist_an[0].data > dist_ap[0].data).sum().item() * 1. / y.size(0)

                else:
                    #### print('Using triplet-weighted loss')
                    TripletLoss2 += F.margin_ranking_loss(dist_an, dist_ap, y, self.margin2)
                    prec = (dist_an.data > dist_ap.data).sum().item() * 1. / y.size(0)


            if self.config.quad and not self.config.trip_weight and not self.config.trip and not self.config.rank_loss:
                dist_ap, dist_an, dist_ann, y = select_quadruplet(trip_out[i], targets, self.num_instances)
                TripletLoss1 += F.margin_ranking_loss(dist_an, dist_ap, y, self.margin1)
                TripletLoss2 += F.margin_ranking_loss(dist_ann, dist_ap, y, self.margin2)
                prec = (dist_an.data > dist_ap.data).sum().item() * 1. / y.size(0)


            if self.config.rank_loss and not self.config.trip_weight and not self.config.trip and not self.config.quad:
                rank_loss, prec = cal_rank_loss_normalize(trip_out[i], targets, self.num_instances, self.config.margin1)
                TripletLoss1 = rank_loss

        if self.config.sac:
            # pdb.set_trace()
            loss = self.alpha * TripletLoss1 + self.beta * TripletLoss2 + self.gamma * xentropy + self.theta * sac_loss
        else:
            loss = self.alpha * TripletLoss1 + self.beta * TripletLoss2 + self.gamma * xentropy
        # pdb.set_trace()
        prec_batch = max(prec, prec_sf)
        return loss, prec_batch


def loss_creator(cfg):
    if cfg.loss_type == 'single':
        return Loss_Single(cfg)
    elif cfg.loss_type =='multi':
        return Loss_Multi(cfg) 
